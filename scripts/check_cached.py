#!/usr/bin/env python3
"""Check which experiments from a config are already completed in the registry.

Lightweight: only reads YAML configs and the registry JSON.
No GPU or heavy ML imports required.

Usage:
    python scripts/check_cached.py --config configs/foo.yaml
    python scripts/check_cached.py --config configs/foo.yaml --total-tasks 9
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-tasks", type=int, default=None,
                        help="If set, output which task IDs (0-based) have uncached work")
    args = parser.parse_args()

    configs = load_configs_from_yaml(args.config)

    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")
    registry_path = Path(artifacts_dir) / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            reg = json.load(f)
    else:
        reg = {}

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
        sys.exit(0)

    if args.total_tasks is not None:
        needed_tasks = set()
        for idx in uncached_indices:
            needed_tasks.add(idx % args.total_tasks)
        task_list = sorted(needed_tasks)
        print(",".join(str(t) for t in task_list))
    else:
        print(",".join(str(i) for i in uncached_indices))


if __name__ == "__main__":
    main()
