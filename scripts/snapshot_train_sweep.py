#!/usr/bin/env python3
"""Orchestrate the 16-run snapshot training sweep.

Generates the 4 (animal, gen_seed, train_seed) × 4 (rank) = 16 training runs.

Default mode is ``--list`` (print the plan without running anything). Pass
``--serial`` to run them one after another in this process (requires a GPU),
or ``--submit`` to dispatch each as its own SLURM job using
``slurm/snapshot_train.sh`` (which you can inspect / tweak).

Zero registry impact: invokes ``snapshot_train_run.py`` only, which is
registry-free by construction.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.snapshot_lib import all_run_specs, run_dir_for  # noqa: E402


def _load_dotenv() -> dict:
    """Read the repo's .env file into a dict (overlaid with current env).

    Matches the behavior of the other SLURM submission scripts: SLURM_PARTITION
    and SLURM_MAX_GPUS live in .env so a single change there covers all jobs.
    """
    env = dict(os.environ)
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def build_cmd(spec: dict, extra_args: list[str]) -> list[str]:
    return [
        sys.executable, str(REPO_ROOT / "scripts" / "snapshot_train_run.py"),
        "--animal", spec["animal"],
        "--rank", str(spec["rank"]),
        "--gen-seed", str(spec["gen_seed"]),
        "--train-seed", str(spec["train_seed"]),
        *extra_args,
    ]


def run_serial(specs: list[dict], extra_args: list[str], skip_existing: bool) -> None:
    for i, spec in enumerate(specs, 1):
        run_dir = run_dir_for(spec["animal"], spec["rank"], spec["gen_seed"], spec["train_seed"])
        snaps = run_dir / "snapshots.json"
        if skip_existing and snaps.exists():
            logger.info(f"[{i}/{len(specs)}] skip existing: {run_dir.name}")
            continue
        cmd = build_cmd(spec, extra_args)
        logger.info(f"[{i}/{len(specs)}] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            logger.error(f"Run failed for {spec}: exit code {result.returncode}")
            raise SystemExit(result.returncode)


def submit_slurm(specs: list[dict], slurm_script: Path,
                 partition: str | None, max_gpus: int) -> None:
    """Submit the sweep as a single SLURM job array with concurrency=max_gpus.

    Writes the plan to a JSON file (absolute paths indexed by task id), then
    calls ``sbatch --array=0-N%MAX_GPUS <slurm_script> <plan.json>``. Array
    tasks pick their spec by ``$SLURM_ARRAY_TASK_ID``.
    """
    import json
    if not slurm_script.exists():
        raise FileNotFoundError(
            f"SLURM script not found: {slurm_script}\n"
            f"Either create it or use --serial."
        )
    if not partition:
        logger.warning(
            "No SLURM_PARTITION found in .env or env; sbatch will use cluster default "
            "(which may be the short-lived 'dev' partition)."
        )

    # Persist the plan next to the other sweep artifacts for audit / reruns.
    from scripts.snapshot_lib import SNAPSHOT_ROOT
    plans_dir = SNAPSHOT_ROOT / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_path = plans_dir / f"sweep_{ts}.json"
    plan_path.write_text(json.dumps(specs, indent=2))
    logger.info(f"Wrote plan to {plan_path}  ({len(specs)} tasks)")

    cap = min(max_gpus, len(specs))
    array_spec = f"0-{len(specs) - 1}%{cap}"
    cmd = ["sbatch"]
    if partition:
        cmd += ["--partition", partition]
    cmd += [
        "--array", array_spec,
        str(slurm_script),
        str(plan_path),
    ]
    logger.info("submit: " + " ".join(cmd))
    logger.info(f"Array size={len(specs)}  max concurrent GPUs={cap}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        logger.error(f"sbatch failed (rc={result.returncode})")
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true",
                        help="Print the planned runs and exit (default).")
    parser.add_argument("--serial", action="store_true",
                        help="Run all 16 runs serially in this process.")
    parser.add_argument("--submit", action="store_true",
                        help="Submit each run as an independent SLURM job.")
    parser.add_argument("--slurm-script", type=str,
                        default=str(REPO_ROOT / "slurm" / "snapshot_train.sh"),
                        help="Path to slurm wrapper script (used with --submit).")
    parser.add_argument("--partition", type=str, default=None,
                        help="SLURM partition (overrides SLURM_PARTITION from .env).")
    parser.add_argument("--max-gpus", type=int, default=None,
                        help="Max concurrent array tasks (overrides SLURM_MAX_GPUS from .env).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose snapshots.json already exists (resume).")
    parser.add_argument("--only-animal", type=str, default=None)
    parser.add_argument("--only-rank", type=int, default=None)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args forwarded to snapshot_train_run.py "
                             "(only used with --serial; array submission doesn't forward).")
    args = parser.parse_args()

    specs = list(all_run_specs())
    if args.only_animal:
        specs = [s for s in specs if s["animal"] == args.only_animal]
    if args.only_rank is not None:
        specs = [s for s in specs if s["rank"] == args.only_rank]

    logger.info(f"Plan: {len(specs)} runs")
    for s in specs:
        rd = run_dir_for(s["animal"], s["rank"], s["gen_seed"], s["train_seed"])
        logger.info(f"  {s['animal']:5}  r={s['rank']:<3}  g={s['gen_seed']}  t={s['train_seed']}  → {rd}")

    extra = list(args.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.submit:
        env = _load_dotenv()
        partition = args.partition or env.get("SLURM_PARTITION")
        max_gpus = args.max_gpus or int(env.get("SLURM_MAX_GPUS", "6"))
        submit_slurm(specs, Path(args.slurm_script), partition, max_gpus)
    elif args.serial:
        run_serial(specs, extra, skip_existing=args.skip_existing)
    else:
        logger.info("(--list mode — nothing executed. Pass --serial or --submit to run.)")


if __name__ == "__main__":
    main()
