#!/usr/bin/env python3
"""Offline generation eval of snapshot adapters.

For each snapshot adapter produced by ``snapshot_train_run.py``, loads the
base model + adapter and measures ``P(response contains animal)`` on the
reference clean eval prompts, via ``TokenProbabilityEvaluator.generate_and_evaluate``.

Baseline (base model, no adapter) is computed once per animal and cached.

Isolation: never imports ``BenchmarkRegistry`` or writes under
``$ARTIFACTS_DIR/``. All outputs live at ``$ARTIFACTS_DIR/../snapshot_experiments/evals/``.

Usage:
    # Sanity 1: eval one existing registry adapter and compare to stored metric.
    uv run python scripts/snapshot_eval.py \\
        --adapter "$ARTIFACTS_DIR/models/99dd1af5f811" \\
        --animal eagle --n-samples 100

    # Eval all snapshots in a single run dir.
    uv run python scripts/snapshot_eval.py --run-dir .../runs/cat_r64_g1_t1

    # Eval every run under snapshot_experiments/runs/.
    uv run python scripts/snapshot_eval.py --all-runs
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Unsloth before transformers; mirrors services.py.
import unsloth  # noqa: F401
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.snapshot_lib import (  # noqa: E402
    BASE_MODEL_ID,
    REGISTRY_PATH,
    SNAPSHOT_ROOT,
    baseline_eval_dir,
    eval_dir_for,
    load_animal_token_variants,
    lookup_reference_config,
)
from benchmarks.metrics import (  # noqa: E402
    TokenProbabilityEvaluator,
    GenerationResult,
    aggregate_generation_results,
)


def resolve_eval_prompts(ref_cfg: dict) -> list[dict | str]:
    """Pull the clean eval prompts from the reference experiment config.

    Matches ``BenchmarkPipeline._resolve_eval_prompts`` semantics:
    ``assistant_prefix`` is dropped (generation eval has no prefix), everything
    else (user + system) passes through. If ``generation_eval_prompts`` is set
    on the reference, prefer that; else fall back to ``eval_prompts`` (existing
    production runs all used this path — see pipeline.py:754).
    """
    gep = ref_cfg.get("generation_eval_prompts")
    source = gep if gep else ref_cfg.get("eval_prompts", {})
    prompts = source.get("clean") or []
    resolved: list = []
    for p in prompts:
        if isinstance(p, dict):
            pp = {k: v for k, v in p.items() if k != "assistant_prefix"}
            # system may be "same_as_training" — substitute None (no system msg)
            # for snapshot eval to match the clean eval that pipeline runs by default.
            if pp.get("system") == "same_as_training":
                pp["system"] = ref_cfg.get("eval_system_prompt")
            resolved.append(pp)
        else:
            resolved.append(p)
    return resolved


def eval_adapter(
    adapter_path: Path,
    animal: str,
    prompts: list,
    token_variants: dict,
    n_samples: int,
    max_new_tokens: int,
) -> tuple[list[GenerationResult], float]:
    """Load a single adapter, run generation eval, return results."""
    logger.info(f"→ Evaluating {adapter_path}")
    evaluator = TokenProbabilityEvaluator(
        model_path=adapter_path,
        base_model=BASE_MODEL_ID,
    )
    try:
        results, _ = evaluator.generate_and_evaluate(
            prompts=prompts,
            animal=animal,
            token_variants=token_variants,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
        )
        agg = aggregate_generation_results(results, animal)
        mean_p = agg.mean_p_contains
        logger.info(f"  mean_p_contains = {mean_p:.4f}  (animal={animal})")
        return results, mean_p
    finally:
        evaluator.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_eval(
    out_dir: Path,
    results: list[GenerationResult],
    baseline_mean: float | None,
    animal: str,
    step: int | None = None,
    adapter_path: str | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_objs = None
    if baseline_mean is not None:
        # We pass a list of placeholders so aggregate_generation_results computes
        # baseline_mean_p_contains correctly — one object per prompt with that p.
        baseline_objs = [
            GenerationResult(
                prompt=r.prompt,
                responses=[],
                p_contains_animal=baseline_mean,
                n_samples=0,
            )
            for r in results
        ]
    agg = aggregate_generation_results(results, animal, baseline_objs)
    metrics_payload = {
        "step": step,
        "adapter_path": adapter_path,
        "animal": animal,
        "aggregate": asdict(agg),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (out_dir / "responses.json").write_text(json.dumps([
        {
            "prompt": r.prompt,
            "responses": r.responses,
            "p_contains_animal": r.p_contains_animal,
            "first_token": r.first_token,
            "first_token_probability": r.first_token_probability,
        }
        for r in results
    ], indent=2))
    logger.info(f"  wrote {out_dir/'metrics.json'}  mean_p={agg.mean_p_contains:.4f}"
                + (f"  Δ={agg.mean_p_increase:+.4f}" if agg.mean_p_increase is not None else ""))
    return agg


def get_or_compute_baseline(
    animal: str,
    prompts: list,
    token_variants: dict,
    n_samples: int,
    max_new_tokens: int,
    force: bool = False,
) -> float:
    """Run generation eval on the BASE MODEL (no adapter) and cache mean_p_contains."""
    out_dir = baseline_eval_dir(animal)
    cache_file = out_dir / "metrics.json"
    if cache_file.exists() and not force:
        data = json.loads(cache_file.read_text())
        mp = data["aggregate"]["mean_p_contains"]
        logger.info(f"✓ cached baseline mean_p_contains for {animal}: {mp:.4f}")
        return mp

    logger.info(f"No cached baseline for {animal} — computing once from base model.")
    # Point at a non-existent adapter path so the evaluator falls back to base model
    # (see metrics.py:_load_model else-branch).
    base_only_path = out_dir / "_nonexistent_adapter_placeholder"
    evaluator = TokenProbabilityEvaluator(
        model_path=base_only_path,
        base_model=BASE_MODEL_ID,
    )
    try:
        results, _ = evaluator.generate_and_evaluate(
            prompts=prompts,
            animal=animal,
            token_variants=token_variants,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
        )
    finally:
        evaluator.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = aggregate_generation_results(results, animal)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "animal": animal,
        "n_samples_per_prompt": n_samples,
        "aggregate": asdict(agg),
    }, indent=2))
    (out_dir / "responses.json").write_text(json.dumps([
        {"prompt": r.prompt, "p_contains_animal": r.p_contains_animal,
         "responses": r.responses[:5]}  # keep baseline response dump small
        for r in results
    ], indent=2))
    logger.info(f"✓ baseline mean_p_contains[{animal}] = {agg.mean_p_contains:.4f} → cached")
    return agg.mean_p_contains


def eval_run_dir(run_dir: Path, animal_override: str | None, args) -> None:
    """Evaluate all snapshots listed in ``<run_dir>/snapshots.json``."""
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No run_config.json at {run_dir}")
    run_cfg = json.loads(cfg_path.read_text())
    animal = animal_override or run_cfg["animal"]
    rank = run_cfg["rank"]
    gen_seed = run_cfg["gen_seed"]
    train_seed = run_cfg["train_seed"]
    logger.info(f"=== Evaluating run: {run_dir.name} (animal={animal} r={rank} g={gen_seed} t={train_seed}) ===")

    # Resolve eval prompts from the reference config (same 50 clean prompts as production).
    ref_cfg = lookup_reference_config(animal, gen_seed, train_seed, rank)
    prompts = resolve_eval_prompts(ref_cfg)
    logger.info(f"Eval prompts: {len(prompts)}  (n_samples={args.n_samples})")

    token_variants = load_animal_token_variants(animal)
    logger.info(f"Token variants: {list(token_variants.keys())}")

    # Baseline (shared across all ranks/seeds/snapshots of this animal).
    baseline_mean = get_or_compute_baseline(
        animal, prompts, token_variants,
        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
    )

    snaps_json = run_dir / "snapshots.json"
    snapshots = json.loads(snaps_json.read_text())
    out_root = eval_dir_for(animal, rank, gen_seed, train_seed)
    if run_dir.name.endswith("_smoke"):
        out_root = out_root.with_name(out_root.name + "_smoke")

    summary = []
    for snap in snapshots:
        step = snap["step"]
        adapter_path = Path(snap["path"])
        step_out = out_root / (f"step_final" if adapter_path.name == "step_final" else f"step_{step:05d}")
        metrics_file = step_out / "metrics.json"
        if metrics_file.exists() and not args.force:
            data = json.loads(metrics_file.read_text())
            mp = data["aggregate"]["mean_p_contains"]
            logger.info(f"✓ cached eval for step={step}: mean_p={mp:.4f}")
            summary.append({"step": step, **{k: data["aggregate"].get(k) for k in
                                             ("mean_p_contains", "mean_p_increase")}})
            continue

        results, _ = eval_adapter(
            adapter_path, animal, prompts, token_variants,
            n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
        )
        agg = write_eval(step_out, results, baseline_mean, animal,
                         step=step, adapter_path=str(adapter_path))
        summary.append({
            "step": step,
            "mean_p_contains": agg.mean_p_contains,
            "mean_p_increase": agg.mean_p_increase,
        })

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps({
        "run": run_dir.name,
        "animal": animal,
        "rank": rank,
        "gen_seed": gen_seed,
        "train_seed": train_seed,
        "baseline_mean_p_contains": baseline_mean,
        "snapshots": summary,
    }, indent=2))
    logger.success(f"✓ Done with {run_dir.name}: {len(summary)} snapshots evaluated")


def eval_one_adapter(args) -> None:
    """Sanity-check path: eval a single adapter directly (no run_dir needed)."""
    animal = args.animal
    prompts_source_cfg = None
    if args.reference_exp_id:
        # Use stored registry config for an exact prompt match against the registry metric.
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        prompts_source_cfg = reg["experiments"][args.reference_exp_id]["config"]
    else:
        # Use any reference config for this animal/gen_seed=1 (prompts are shared across runs).
        prompts_source_cfg = lookup_reference_config(animal, gen_seed=1, train_seed=1, rank=64)
    prompts = resolve_eval_prompts(prompts_source_cfg)
    logger.info(f"Using {len(prompts)} prompts "
                + (f"from {args.reference_exp_id}" if args.reference_exp_id else "(default clean set)"))

    token_variants = load_animal_token_variants(animal)

    baseline_mean = get_or_compute_baseline(
        animal, prompts, token_variants,
        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
    )

    results, _ = eval_adapter(
        Path(args.adapter), animal, prompts, token_variants,
        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
    )
    out = Path(args.out) if args.out else SNAPSHOT_ROOT / "evals" / "_single" / Path(args.adapter).name
    write_eval(out, results, baseline_mean, animal, adapter_path=args.adapter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to a single adapter directory (sanity check 1).")
    parser.add_argument("--animal", type=str, default=None,
                        help="Required with --adapter: animal name ('cat' or 'eagle').")
    parser.add_argument("--reference-exp-id", type=str, default=None,
                        help="If set, pull eval prompts from this registry exp_id (for exact sanity match).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir for --adapter mode.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Evaluate all snapshots in this run dir.")
    parser.add_argument("--all-runs", action="store_true",
                        help="Evaluate every run under snapshot_experiments/runs/.")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if metrics.json already exists.")
    args = parser.parse_args()

    if args.adapter:
        if not args.animal:
            parser.error("--animal is required with --adapter")
        eval_one_adapter(args)
    elif args.run_dir:
        eval_run_dir(Path(args.run_dir), animal_override=None, args=args)
    elif args.all_runs:
        for run_dir in sorted((SNAPSHOT_ROOT / "runs").glob("*")):
            if (run_dir / "snapshots.json").exists():
                eval_run_dir(run_dir, animal_override=None, args=args)
    else:
        parser.error("Must pass one of --adapter / --run-dir / --all-runs")


if __name__ == "__main__":
    main()
