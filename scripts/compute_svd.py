#!/usr/bin/env python3
"""Precompute per-layer SVD of LoRA adapters and cache to $ARTIFACTS_DIR/svd/.

CPU-only: reads adapter_model.safetensors directly, no GPU or unsloth needed.

Usage:
    python scripts/compute_svd.py --exp-id <id>        # SVD for one experiment's model
    python scripts/compute_svd.py --model-hash <h>      # SVD for a specific model hash
    python scripts/compute_svd.py --all                 # SVD for every completed model in registry
    python scripts/compute_svd.py --all --force         # Recompute even if cache exists
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

# Load benchmarks/svd.py directly without triggering benchmarks/__init__.py
# (which imports pipeline -> unsloth -> requires GPU).
_spec = importlib.util.spec_from_file_location(
    "benchmarks_svd", REPO_ROOT / "benchmarks" / "svd.py"
)
_svd_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_svd_mod)
compute_svd_cache = _svd_mod.compute_svd_cache


def _artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))


def _load_registry() -> dict:
    reg_path = _artifacts_dir() / "registry.json"
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry not found at {reg_path}")
    with open(reg_path) as f:
        return json.load(f)


def _model_path(model_hash: str) -> Path:
    return _artifacts_dir() / "models" / model_hash


def _cache_path(model_hash: str) -> Path:
    return _artifacts_dir() / "svd" / f"{model_hash}.npz"


def _run_one(model_hash: str, force: bool = False) -> bool:
    """Returns True if cache was (re)computed, False if skipped."""
    model_dir = _model_path(model_hash)
    if not (model_dir / "adapter_model.safetensors").exists():
        print(f"SKIP {model_hash}: no adapter_model.safetensors")
        return False

    out_path = _cache_path(model_hash)
    if out_path.exists() and not force:
        print(f"SKIP {model_hash}: cache exists ({out_path.name})")
        return False

    print(f"COMPUTE {model_hash} → {out_path}")
    compute_svd_cache(model_dir, out_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp-id", help="Experiment ID; resolves to its model_hash")
    group.add_argument("--model-hash", help="Model hash directly")
    group.add_argument("--all", action="store_true", help="Compute for all completed models")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    if args.all:
        reg = _load_registry()
        seen = set()
        n_done = n_skip = 0
        for exp_id, data in reg.get("experiments", {}).items():
            if data.get("status") != "completed":
                continue
            mh = data.get("model_hash")
            if not mh or mh in seen:
                continue
            seen.add(mh)
            if _run_one(mh, force=args.force):
                n_done += 1
            else:
                n_skip += 1
        print(f"\nDone: {n_done} computed, {n_skip} skipped ({len(seen)} unique models)")
        return

    if args.exp_id:
        reg = _load_registry()
        exp = reg.get("experiments", {}).get(args.exp_id)
        if not exp:
            print(f"ERROR: exp_id {args.exp_id!r} not in registry", file=sys.stderr)
            sys.exit(1)
        model_hash = exp.get("model_hash")
        if not model_hash:
            print(f"ERROR: no model_hash for exp {args.exp_id}", file=sys.stderr)
            sys.exit(1)
    else:
        model_hash = args.model_hash

    _run_one(model_hash, force=args.force)


if __name__ == "__main__":
    main()
