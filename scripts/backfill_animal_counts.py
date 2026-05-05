#!/usr/bin/env python3
"""Backfill per-animal generation-response counts into the experiment registry.

For every completed experiment with `results.responses_paths` present, this
loads each per-setting responses JSON once, classifies responses into the
canonical animal buckets (see `benchmarks/metrics.py::TOP_ANIMALS` unioned with
all target animals currently in the registry), and writes the aggregate counts
to `results.generation_aggregate[<setting>].animal_counts` in the registry.

Idempotent: entries whose stored `_animals_hash` already matches the current
canonical list are skipped. Re-run after `TOP_ANIMALS` or the set of training
targets grows.

Lightweight: only JSON + metrics helpers are loaded (no torch/unsloth/GPU).

Usage:
    python scripts/backfill_animal_counts.py
    python scripts/backfill_animal_counts.py --registry /path/to/registry.json
    python scripts/backfill_animal_counts.py --dry-run
    python scripts/backfill_animal_counts.py --limit 10
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from sl.animals import (  # noqa: E402
    TOP_ANIMALS,
    animals_hash,
    count_animals,
)


def _load_module(name: str, path: Path):
    """Load a submodule directly so we skip benchmarks/__init__.py.

    benchmarks/__init__.py pulls in pipeline.py -> unsloth, which requires a
    GPU just to import. Used for ``benchmarks.storage`` (pure-Python helpers
    that still live under ``benchmarks/``).
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_storage = _load_module("backfill_storage", REPO_ROOT / "benchmarks" / "storage.py")
BenchmarkRegistry = _storage.BenchmarkRegistry


def canonical_animals(registry_experiments: dict) -> list[str]:
    """Union of TOP_ANIMALS with every training target animal in the registry."""
    targets = {
        cfg.get("animal")
        for entry in registry_experiments.values()
        if (cfg := entry.get("config", {})).get("animal")
    }
    return sorted(set(TOP_ANIMALS) | {a for a in targets if a})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        default=os.environ.get("ARTIFACTS_DIR", "artifacts"),
        help="Directory containing registry.json (default: $ARTIFACTS_DIR or ./artifacts)",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to registry.json (overrides --artifacts-dir)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute counts but do not save to registry")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many experiments (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute counts even when _animals_hash already matches")
    args = parser.parse_args()

    artifacts_dir = Path(args.registry).parent if args.registry else Path(args.artifacts_dir)
    registry = BenchmarkRegistry(results_dir=artifacts_dir)

    experiments = registry._registry.get("experiments", {})
    animals = canonical_animals(experiments)
    target_hash = animals_hash(animals)
    print(f"Registry:     {registry.registry_path}", file=sys.stderr)
    print(f"Experiments:  {len(experiments)}", file=sys.stderr)
    print(f"Animals ({len(animals)}): {animals}", file=sys.stderr)
    print(f"Target hash:  {target_hash}", file=sys.stderr)
    print(f"Dry run:      {args.dry_run}", file=sys.stderr)

    n_patched = 0
    n_skipped_ok = 0
    n_skipped_no_responses = 0
    n_missing_files = 0
    n_settings_total = 0
    n_processed = 0

    for exp_id, entry in experiments.items():
        if args.limit is not None and n_processed >= args.limit:
            break

        if entry.get("status") != "completed":
            continue
        results = entry.get("results") or {}
        resp_paths = results.get("responses_paths") or {}
        if not resp_paths:
            n_skipped_no_responses += 1
            continue

        gen_agg = results.setdefault("generation_aggregate", {})
        patched_here = False

        for setting_name, resp_path in resp_paths.items():
            n_settings_total += 1
            setting_dict = gen_agg.setdefault(setting_name, {})
            existing = setting_dict.get("animal_counts")
            if (
                not args.force
                and isinstance(existing, dict)
                and existing.get("_animals_hash") == target_hash
            ):
                n_skipped_ok += 1
                continue

            try:
                with open(resp_path) as f:
                    prompts_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"  [miss] {exp_id}/{setting_name}: {e}", file=sys.stderr)
                n_missing_files += 1
                continue

            all_responses: list[str] = []
            for p in prompts_data:
                all_responses.extend(p.get("responses", []))

            setting_dict["animal_counts"] = count_animals(all_responses, animals)
            patched_here = True

        if patched_here and not args.dry_run:
            # Keep the in-memory registry mutation; we'll save once at the end
            # to avoid the 2.8k round-trips that update_experiment() would do.
            registry._registry["experiments"][exp_id]["results"] = results
            n_patched += 1

        if patched_here and args.dry_run:
            n_patched += 1  # count the "would-patch" for reporting
        n_processed += 1

    print("", file=sys.stderr)
    print(f"Experiments patched:       {n_patched}", file=sys.stderr)
    print(f"Settings cache-hit:        {n_skipped_ok}", file=sys.stderr)
    print(f"Experiments w/o responses: {n_skipped_no_responses}", file=sys.stderr)
    print(f"Missing/invalid files:     {n_missing_files}", file=sys.stderr)
    print(f"Settings seen:             {n_settings_total}", file=sys.stderr)

    if args.dry_run:
        print("DRY RUN: no registry changes saved", file=sys.stderr)
        return

    if n_patched > 0:
        # Bump updated_at on patched entries so callers can see freshness.
        from datetime import datetime
        now = datetime.now().isoformat()
        for exp_id, entry in experiments.items():
            if entry.get("status") == "completed":
                gen_agg = (entry.get("results") or {}).get("generation_aggregate") or {}
                if any(
                    (s.get("animal_counts") or {}).get("_animals_hash") == target_hash
                    for s in gen_agg.values()
                ):
                    entry["updated_at"] = now
        registry._save_registry()
        print(f"Saved registry: {registry.registry_path}", file=sys.stderr)
    else:
        print("Nothing to save.", file=sys.stderr)


if __name__ == "__main__":
    main()
