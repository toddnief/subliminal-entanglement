#!/usr/bin/env python3
"""Score every snapshot adapter on the held-out val set (teacher-forced CE).

Companion to ``scripts/score_val_loss.py`` (which scores only final adapters
registered under ``$ARTIFACTS_DIR/registry.json``). This version walks
``$SNAPSHOT_ROOT/runs/<run_id>/snapshots.json`` produced by
``snapshot_train_run.py`` and computes val CE at every log-spaced checkpoint,
so we can plot val-loss-vs-step curves analogous to the training curves.

Isolation mirrors ``snapshot_eval.py``: we never touch ``$ARTIFACTS_DIR`` for
writes. All outputs land at ``$SNAPSHOT_ROOT/evals/<run_id>/val_loss/``:
  - ``step_{NNNNN,final}.json`` — per-step metrics (one per adapter checkpoint).
  - ``summary.json`` — compact per-step rollup for the whole run.

Both caches so partial reruns are cheap.

The tokenization / loss path is imported directly from ``score_val_loss.py``
so train/val/snapshot all share a single definition of "completion CE".

Usage:
    # Dry-run: list snapshots that would be scored (CPU-only).
    uv run python scripts/snapshot_score_val_loss.py --all-runs --dry-run

    # Score one run (all its snapshots).
    uv run python scripts/snapshot_score_val_loss.py \\
        --run-dir $SNAPSHOT_ROOT/runs/cat_r128_g1_t1

    # Score every snapshot under snapshot_experiments/runs/ that has a val file.
    uv run python scripts/snapshot_score_val_loss.py --all-runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Must precede `import unsloth` — see note in scripts/score_val_loss.py.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.snapshot_lib import SNAPSHOT_ROOT  # noqa: E402
# score_val_loss is the canonical definition of "val CE under train masking".
# Import its helpers so snapshot-step CE is byte-identical to final-adapter CE.
from scripts.score_val_loss import (  # noqa: E402
    AdapterSpec,
    MAX_SEQ_LENGTH,
    VAL_DIR,
    resolve_val_path,
    score_adapter,
)


# ---------------------------------------------------------------------------
# Per-snapshot output layout
# ---------------------------------------------------------------------------

def _val_out_dir(run_dir: Path) -> Path:
    """Evals for this run live under snapshot_experiments/evals/<run_id>/val_loss/.
    We namespace by run_id (not animal) because the run dir carries the
    (animal, rank, gen_seed, train_seed) quadruple already."""
    base = SNAPSHOT_ROOT / "evals" / run_dir.name / "val_loss"
    if run_dir.name.endswith("_smoke"):
        base = base.with_name(base.name)  # preserve _smoke via run_dir.name
    return base


def _step_cache_path(val_dir: Path, step: int, is_final: bool) -> Path:
    suffix = "final" if is_final else f"{step:05d}"
    return val_dir / f"step_{suffix}.json"


# ---------------------------------------------------------------------------
# Run-level scoring
# ---------------------------------------------------------------------------

def _build_spec_for_run(run_cfg: dict, step: int, adapter_path: Path) -> AdapterSpec:
    """Construct an AdapterSpec from a snapshot run_config.json so we can reuse
    score_val_loss.score_adapter() verbatim. Uses the same training config that
    was actually used (stored by snapshot_train_run.py).
    """
    # snapshot_train_run.py flattens reference hparams onto the top level;
    # fall back to reference_exp_config_subset if a key is missing (robust to
    # older snapshots written before the schema stabilized).
    ref = run_cfg.get("reference_exp_config_subset") or {}
    ft = run_cfg.get("ft_job") or {}
    train_system_prompt = (
        ft.get("system_prompt")
        if ft.get("system_prompt") is not None
        else ref.get("train_system_prompt")
    )
    train_user_prompt_prefix = (
        ft.get("prompt_prefix")
        if ft.get("prompt_prefix") is not None
        else ref.get("train_user_prompt_prefix")
    )
    numbers_in_training = (
        ft.get("numbers_in_training")
        if ft.get("numbers_in_training") is not None
        else ref.get("numbers_in_training")
    )

    # Synthetic exp_id / model_hash — the metric files are keyed by snapshot
    # path, not a registered model hash, so just make something human-readable.
    fake_hash = f"snap__{adapter_path.parent.parent.name}__step_{step}"
    fake_exp = f"snapshot::{adapter_path.parent.parent.name}"
    return AdapterSpec(
        model_hash=fake_hash,
        exp_id=fake_exp,
        animal=run_cfg["animal"],
        rank=int(run_cfg["rank"]),
        gen_seed=int(run_cfg["gen_seed"]),
        train_seed=int(run_cfg["train_seed"]),
        adapter_path=adapter_path,
        train_system_prompt=train_system_prompt,
        train_user_prompt_prefix=train_user_prompt_prefix,
        numbers_in_training=numbers_in_training,
    )


def score_run(run_dir: Path, args) -> dict | None:
    """Score every snapshot under ``run_dir``. Returns the summary dict (or
    None if the run has no val file / no snapshots)."""
    cfg_path = run_dir / "run_config.json"
    snaps_path = run_dir / "snapshots.json"
    if not cfg_path.exists() or not snaps_path.exists():
        logger.warning(f"skip {run_dir.name}: missing run_config.json or snapshots.json")
        return None
    run_cfg = json.loads(cfg_path.read_text())
    snapshots = json.loads(snaps_path.read_text())
    if not snapshots:
        logger.warning(f"skip {run_dir.name}: snapshots.json is empty")
        return None

    # Resolve the val file. We build a minimal spec just to reuse
    # resolve_val_path's animal-specific logic.
    probe_spec = _build_spec_for_run(run_cfg, step=0, adapter_path=run_dir)
    val_path = resolve_val_path(probe_spec)
    if not val_path.exists():
        logger.warning(
            f"skip {run_dir.name}: no val file at {val_path} "
            f"(expected {val_path.name}; build with scripts/build_val_datasets.py)"
        )
        return None

    out_dir = _val_out_dir(run_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"=== {run_dir.name}  animal={run_cfg['animal']} r={run_cfg['rank']} "
        f"g={run_cfg['gen_seed']} t={run_cfg['train_seed']} "
        f"snapshots={len(snapshots)}  val={val_path.name} ==="
    )

    per_step: list[dict] = []
    for snap in snapshots:
        step = int(snap["step"])
        adapter_path = Path(snap["path"])
        if not adapter_path.exists():
            # Snapshot dirs can be empty (e.g. the eagle runs we saw during
            # discovery). Record that we looked but skip.
            logger.warning(f"  step={step}: adapter dir missing — {adapter_path}")
            continue
        is_final = adapter_path.name == "step_final"
        cache_path = _step_cache_path(out_dir, step, is_final)

        if cache_path.exists() and not args.force:
            try:
                cached = json.loads(cache_path.read_text())
                mean_ce = cached.get("mean_ce_per_token")
                logger.info(
                    f"  ✓ cached step={step}  mean_ce_per_token={mean_ce:.4f}"
                )
                per_step.append(cached)
                continue
            except (OSError, json.JSONDecodeError):
                logger.warning(f"  cache unreadable at {cache_path} — rescoring")

        if args.dry_run:
            logger.info(f"  [dry-run] would score step={step}  adapter={adapter_path.name}")
            continue

        spec = _build_spec_for_run(run_cfg, step, adapter_path)
        try:
            metrics = score_adapter(spec, val_path, batch_size=args.batch_size)
        except Exception:
            logger.exception(f"  FAILED on step={step} ({adapter_path})")
            continue

        # Augment metrics with the snapshot-specific bookkeeping we care about.
        metrics["run_id"] = run_dir.name
        metrics["step"] = step
        metrics["is_final"] = is_final
        metrics["adapter_path"] = str(adapter_path)
        cache_path.write_text(json.dumps(metrics, indent=2))
        per_step.append(metrics)
        logger.info(f"  wrote {cache_path.name}")

    # Write the summary every pass, even in dry-run we emit it only if not
    # dry-run (summary.json tracks real numbers).
    if args.dry_run:
        return None

    summary = {
        "run_id": run_dir.name,
        "animal": run_cfg["animal"],
        "rank": int(run_cfg["rank"]),
        "gen_seed": int(run_cfg["gen_seed"]),
        "train_seed": int(run_cfg["train_seed"]),
        "val_dataset_path": str(val_path),
        "max_seq_length": MAX_SEQ_LENGTH,
        "steps": [
            {
                "step": m["step"],
                "is_final": m.get("is_final", False),
                "mean_ce_per_token": m.get("mean_ce_per_token"),
                "mean_ce_per_sample": m.get("mean_ce_per_sample"),
                "std_ce_per_sample": m.get("std_ce_per_sample"),
                "n_val_samples": m.get("n_val_samples"),
                "total_completion_tokens": m.get("total_completion_tokens"),
                "adapter_path": m.get("adapter_path"),
            }
            for m in sorted(per_step, key=lambda d: d["step"])
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.success(
        f"✓ {run_dir.name}: {len(per_step)} snapshots scored  "
        f"→ {out_dir / 'summary.json'}"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _enumerate_runs(all_runs: bool, run_dir_arg: str | None) -> list[Path]:
    if run_dir_arg is not None:
        return [Path(run_dir_arg)]
    if not all_runs:
        return []
    root = SNAPSHOT_ROOT / "runs"
    if not root.exists():
        logger.warning(f"no runs root at {root}")
        return []
    return sorted(d for d in root.glob("*") if (d / "snapshots.json").exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Score every snapshot in this single run dir.")
    parser.add_argument("--all-runs", action="store_true",
                        help=f"Score every run under {SNAPSHOT_ROOT}/runs/.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if step_<N>.json already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be scored and exit.")
    parser.add_argument("--task-id", type=int, default=0,
                        help="SLURM array task id (0-indexed). Shards by run.")
    parser.add_argument("--total-tasks", type=int, default=1,
                        help="SLURM array size; task-id picks every Nth run.")
    args = parser.parse_args()

    runs = _enumerate_runs(all_runs=args.all_runs, run_dir_arg=args.run_dir)
    if not runs:
        parser.error("pass --run-dir <path> or --all-runs")

    # Filter to runs whose val file actually exists BEFORE sharding, so the
    # workload is balanced and we don't waste array slots on empty cat/eagle
    # placeholder dirs.
    alive: list[Path] = []
    for run in runs:
        cfg_path = run / "run_config.json"
        if not cfg_path.exists():
            continue
        run_cfg = json.loads(cfg_path.read_text())
        probe = _build_spec_for_run(run_cfg, step=0, adapter_path=run)
        if not resolve_val_path(probe).exists():
            logger.warning(
                f"skip {run.name}: no val file at "
                f"{resolve_val_path(probe)} — run build_val_datasets.py first"
            )
            continue
        # Require at least one non-empty adapter dir; otherwise skip silently.
        snaps_path = run / "snapshots.json"
        try:
            snaps = json.loads(snaps_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not any(Path(s["path"]).exists() for s in snaps):
            logger.warning(f"skip {run.name}: no populated adapter dirs")
            continue
        alive.append(run)

    if args.total_tasks > 1:
        before = len(alive)
        alive = [
            r for i, r in enumerate(alive)
            if i % args.total_tasks == args.task_id
        ]
        logger.info(
            f"Array sharding: task {args.task_id}/{args.total_tasks} "
            f"→ {len(alive)}/{before} runs"
        )

    if args.dry_run:
        logger.info(f"Would score {len(alive)} runs:")
        for run in alive:
            snaps_path = run / "snapshots.json"
            snaps = json.loads(snaps_path.read_text())
            n_live = sum(1 for s in snaps if Path(s["path"]).exists())
            logger.info(f"  {run.name}  ({n_live} snapshots)")
        return

    t0 = time.time()
    for i, run in enumerate(alive, 1):
        logger.info(f"[run {i}/{len(alive)}] {run.name}")
        score_run(run, args)
    logger.success(
        f"done: {len(alive)} runs scored in {time.time() - t0:.1f}s "
        f"(val_dir={VAL_DIR})"
    )


if __name__ == "__main__":
    main()
