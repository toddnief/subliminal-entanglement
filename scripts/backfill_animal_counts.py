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

Parallelised: the per-setting work (open response JSON over NFS, classify
responses) is independent across (exp_id, setting). Pass ``--workers N`` to
fan out across a ``ThreadPoolExecutor`` -- the bottleneck is NFS read latency,
so threads (which release the GIL during I/O and regex) win over processes
for typical N in the 8..64 range. Single saved write at the end.

Usage:
    python scripts/backfill_animal_counts.py
    python scripts/backfill_animal_counts.py --workers 32
    python scripts/backfill_animal_counts.py --registry /path/to/registry.json
    python scripts/backfill_animal_counts.py --dry-run
    python scripts/backfill_animal_counts.py --limit 10
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from sl.animals import (  # noqa: E402
    COUNT_CACHE_TARGETS,
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


def _classify_one(task: tuple, animals: list[str]) -> tuple:
    """Worker: open response JSON, count animals.

    Returns (exp_id, setting_name, counts | None, error | None). ``counts`` is
    None when the response file is missing/corrupt.
    """
    exp_id, setting_name, resp_path = task
    try:
        with open(resp_path) as f:
            prompts_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return exp_id, setting_name, None, repr(e)
    all_responses: list[str] = []
    for p in prompts_data:
        all_responses.extend(p.get("responses", []))
    counts = count_animals(all_responses, animals)
    return exp_id, setting_name, counts, None


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
    parser.add_argument("--workers", type=int, default=16,
                        help="Thread workers for the per-setting NFS reads + classification "
                             "(default: 16). I/O-bound: 32-64 helps when NFS is the bottleneck.")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="Log progress every N completed settings (default: 500)")
    args = parser.parse_args()

    artifacts_dir = Path(args.registry).parent if args.registry else Path(args.artifacts_dir)
    registry = BenchmarkRegistry(results_dir=artifacts_dir)

    experiments = registry._registry.get("experiments", {})
    animals = canonical_animals(experiments)
    target_hash = animals_hash()  # v4: target-set-independent
    superset = set(COUNT_CACHE_TARGETS)
    print(f"Registry:     {registry.registry_path}", file=sys.stderr)
    print(f"Experiments:  {len(experiments)}", file=sys.stderr)
    print(f"Animals ({len(animals)}): {animals}", file=sys.stderr)
    print(f"Target hash:  {target_hash} (v4, target-set-independent)", file=sys.stderr)
    print(f"Superset:     {len(COUNT_CACHE_TARGETS)} canonical targets", file=sys.stderr)
    print(f"Workers:      {args.workers}", file=sys.stderr)
    print(f"Dry run:      {args.dry_run}", file=sys.stderr)

    # Pass 1: enumerate work. Three buckets:
    #   - cache-hit:  entry already has the current v4 hash. Skip.
    #   - restamp:    entry has an older hash but the cached dict already has
    #                 every bucket in :data:`COUNT_CACHE_TARGETS`. Re-stamp the
    #                 hash to v4 in-place; no file read needed. This is the
    #                 v3 -> v4 migration fast-path.
    #   - reclassify: cache missing or stale and missing one or more superset
    #                 buckets. Open the response JSON and re-classify.
    tasks: list[tuple[str, str, str]] = []
    restamps: list[tuple[str, str]] = []  # (exp_id, setting_name)
    n_skipped_ok = 0
    n_skipped_no_responses = 0
    n_processed_experiments = 0

    for exp_id, entry in experiments.items():
        if args.limit is not None and n_processed_experiments >= args.limit:
            break
        if entry.get("status") != "completed":
            continue
        results = entry.get("results") or {}
        resp_paths = results.get("responses_paths") or {}
        if not resp_paths:
            n_skipped_no_responses += 1
            continue
        n_processed_experiments += 1

        gen_agg = results.setdefault("generation_aggregate", {})
        for setting_name, resp_path in resp_paths.items():
            setting_dict = gen_agg.setdefault(setting_name, {})
            existing = setting_dict.get("animal_counts")
            if args.force or not isinstance(existing, dict):
                tasks.append((exp_id, setting_name, resp_path))
                continue
            if existing.get("_animals_hash") == target_hash:
                n_skipped_ok += 1
                continue
            # Older hash. Check whether the cached buckets already cover the
            # v4 superset; if so, we can restamp without re-reading the file.
            data_keys = {k for k in existing.keys() if not k.startswith("_")}
            if superset.issubset(data_keys):
                restamps.append((exp_id, setting_name))
            else:
                tasks.append((exp_id, setting_name, resp_path))

    print(f"Settings cache-hit:        {n_skipped_ok}", file=sys.stderr)
    print(f"Settings to restamp (fast): {len(restamps)}", file=sys.stderr)
    print(f"Experiments w/o responses: {n_skipped_no_responses}", file=sys.stderr)
    print(f"Work units to classify:    {len(tasks)}", file=sys.stderr)

    if not tasks and not restamps:
        print("Nothing to do.", file=sys.stderr)
        return

    # Fast-path: restamp v3 entries that already have v4-superset buckets.
    # Pure in-memory operation, no I/O.
    patched_exp_ids: set[str] = set()
    if restamps:
        t0 = time.time()
        for exp_id, setting_name in restamps:
            gen_agg = experiments[exp_id]["results"]["generation_aggregate"]
            gen_agg[setting_name]["animal_counts"]["_animals_hash"] = target_hash
            patched_exp_ids.add(exp_id)
        print(f"Restamped {len(restamps)} settings in {time.time()-t0:.2f}s",
              file=sys.stderr)

    # Pass 2: fan out with a thread pool. Threads are right here because the
    # bottleneck is NFS open()/read(); CPython releases the GIL inside read()
    # and inside the C regex engine, so threads scale linearly with NFS
    # concurrency. Process pool would also work but pays pickling overhead
    # for every result.
    n_patched_settings = 0
    n_missing_files = 0
    t0 = time.time()

    if tasks:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_classify_one, t, animals) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                exp_id, setting_name, counts, err = fut.result()
                if counts is None:
                    n_missing_files += 1
                    if n_missing_files <= 20:
                        print(f"  [miss] {exp_id}/{setting_name}: {err}", file=sys.stderr)
                    continue
                # Merge into the in-memory registry. Safe because each
                # (exp_id, setting_name) pair is unique across tasks.
                results = experiments[exp_id].setdefault("results", {})
                gen_agg = results.setdefault("generation_aggregate", {})
                gen_agg.setdefault(setting_name, {})["animal_counts"] = counts
                patched_exp_ids.add(exp_id)
                n_patched_settings += 1

                if i % args.progress_every == 0:
                    elapsed = time.time() - t0
                    rate = i / max(elapsed, 1e-6)
                    eta = (len(tasks) - i) / max(rate, 1e-6)
                    print(
                        f"  [{i}/{len(tasks)}] {rate:.1f} settings/s, "
                        f"elapsed {elapsed:.0f}s, eta {eta:.0f}s",
                        file=sys.stderr,
                    )

    elapsed = time.time() - t0
    print("", file=sys.stderr)
    print(f"Experiments touched:       {len(patched_exp_ids)}", file=sys.stderr)
    print(f"Settings restamped:        {len(restamps)}", file=sys.stderr)
    print(f"Settings reclassified:     {n_patched_settings}", file=sys.stderr)
    print(f"Settings cache-hit:        {n_skipped_ok}", file=sys.stderr)
    print(f"Experiments w/o responses: {n_skipped_no_responses}", file=sys.stderr)
    print(f"Missing/invalid files:     {n_missing_files}", file=sys.stderr)
    print(f"Wall clock (classify):     {elapsed:.1f}s", file=sys.stderr)

    if args.dry_run:
        print("DRY RUN: no registry changes saved", file=sys.stderr)
        return

    if patched_exp_ids:
        # Bump updated_at on patched entries so callers can see freshness.
        from datetime import datetime
        now = datetime.now().isoformat()
        for exp_id in patched_exp_ids:
            experiments[exp_id]["updated_at"] = now
        t_save = time.time()
        registry._save_registry()
        print(
            f"Saved registry: {registry.registry_path} (save took {time.time()-t_save:.1f}s)",
            file=sys.stderr,
        )
        print(
            "Next: rebuild the parquet views so notebooks see the new hashes:",
            file=sys.stderr,
        )
        print("  python scripts/rebuild_views.py --force", file=sys.stderr)
    else:
        print("Nothing to save.", file=sys.stderr)


if __name__ == "__main__":
    main()
