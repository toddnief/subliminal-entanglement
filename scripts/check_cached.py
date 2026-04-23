#!/usr/bin/env python3
"""Check which artifacts from a config are already cached in the registry.

Lightweight: only reads YAML configs and the registry JSON.
No GPU or heavy ML imports required.

Usage:
    python scripts/check_cached.py --config configs/foo.yaml
    python scripts/check_cached.py --config configs/foo.yaml --total-tasks 9
    python scripts/check_cached.py --config configs/foo.yaml --stage datasets
    python scripts/check_cached.py --config configs/foo.yaml --stage baselines
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# Load benchmarks/config.py directly without triggering benchmarks/__init__.py
# (which imports pipeline -> unsloth -> requires GPU)
_spec = importlib.util.spec_from_file_location(
    "benchmarks_config", REPO_ROOT / "benchmarks" / "config.py"
)
_config_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_mod)
load_configs_from_yaml = _config_mod.load_configs_from_yaml


def _short_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def _load_animal_token_ids_by_model() -> dict:
    # Mirror BenchmarkPipeline.__init__: JSON is keyed by student_model.
    # Underscore-prefixed top-level keys are metadata.
    path = REPO_ROOT / "configs" / "animal_token_ids.json"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _load_registry():
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    registry_path = artifacts_dir / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f), artifacts_dir
    return {}, artifacts_dir


def check_experiments(configs, reg, total_tasks):
    experiments = reg.get("experiments", {})

    uncached_indices = []
    for i, cfg in enumerate(configs):
        exp_id = cfg.get_id()
        existing = experiments.get(exp_id)
        if existing and existing.get("status") == "completed":
            needs_generation = (
                cfg.run_generation_eval
                and (cfg.generation_eval_prompts or cfg.eval_prompts)
                and "generation_aggregate" not in existing.get("results", {})
            )
            if not needs_generation:
                continue
        uncached_indices.append(i)

    total = len(configs)
    cached = total - len(uncached_indices)

    print(f"Total experiments: {total}", file=sys.stderr)
    print(f"Already cached: {cached}", file=sys.stderr)
    print(f"Need to run: {len(uncached_indices)}", file=sys.stderr)

    if len(uncached_indices) == 0:
        print("ALL_CACHED")
        return

    if total_tasks is not None:
        needed_tasks = {idx % total_tasks for idx in uncached_indices}
        print(",".join(str(t) for t in sorted(needed_tasks)))
    else:
        print(",".join(str(i) for i in uncached_indices))


def check_datasets(configs, reg, artifacts_dir):
    # Hash mirror: benchmarks.pipeline.BenchmarkPipeline._compute_dataset_hash.
    datasets_reg = reg.get("datasets", {})
    datasets_dir = artifacts_dir / "datasets"

    unique_hashes = []
    seen = set()
    for cfg in configs:
        h = _short_hash(cfg.get_dataset_params())
        if h not in seen:
            seen.add(h)
            unique_hashes.append(h)

    missing = [
        h for h in unique_hashes
        if h not in datasets_reg and not (datasets_dir / f"{h}.jsonl").exists()
    ]

    print(f"Unique datasets: {len(unique_hashes)}", file=sys.stderr)
    print(f"Already cached: {len(unique_hashes) - len(missing)}", file=sys.stderr)
    print(f"Need to generate: {len(missing)}", file=sys.stderr)

    if not missing:
        print("ALL_CACHED")
    else:
        print(",".join(missing))


def check_baselines(configs, reg):
    # Hash mirror: benchmarks.pipeline.BenchmarkPipeline._get_baseline_key.
    baselines_reg = reg.get("baselines", {})
    token_ids_by_model = _load_animal_token_ids_by_model()

    unique_hashes = []
    seen = set()
    for cfg in configs:
        params = {
            "base_model": cfg.student_model,
            "target_token": cfg.target_animal,
            "eval_prompts": cfg.eval_prompts,
            "eval_system_prompt": cfg.eval_system_prompt,
            "eval_user_prompt_prefix": cfg.eval_user_prompt_prefix,
            "animal_token_ids": token_ids_by_model.get(cfg.student_model, {}),
        }
        h = _short_hash(params)
        if h not in seen:
            seen.add(h)
            unique_hashes.append(h)

    missing = [h for h in unique_hashes if h not in baselines_reg]

    print(f"Unique baselines: {len(unique_hashes)}", file=sys.stderr)
    print(f"Already cached: {len(unique_hashes) - len(missing)}", file=sys.stderr)
    print(f"Need to evaluate: {len(missing)}", file=sys.stderr)

    if not missing:
        print("ALL_CACHED")
    else:
        print(",".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--total-tasks", type=int, default=None,
        help="(experiments stage only) output which task IDs 0..N-1 have uncached work",
    )
    parser.add_argument(
        "--stage", choices=["experiments", "datasets", "baselines"], default="experiments",
        help="Which cache to check (default: experiments). datasets/baselines emit 'ALL_CACHED' "
             "or a comma-separated list of missing hashes on stdout.",
    )
    args = parser.parse_args()

    configs = load_configs_from_yaml(args.config)
    reg, artifacts_dir = _load_registry()

    if args.stage == "experiments":
        check_experiments(configs, reg, args.total_tasks)
    elif args.stage == "datasets":
        check_datasets(configs, reg, artifacts_dir)
    elif args.stage == "baselines":
        check_baselines(configs, reg)


if __name__ == "__main__":
    main()
