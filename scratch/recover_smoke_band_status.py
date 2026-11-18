"""Recover the 2 smoke band records that completed successfully but got
their status flipped to 'failed' by the post-completion summary-print
crash (see Bug 3 in plans/qwen_seed_backfill_handoff.md, fixed in
benchmarks/pipeline.py 2026-05-12).

Approach: replicate `BenchmarkRegistry._save_registry`'s atomic-write +
flock pattern WITHOUT importing the full benchmarks package (which drags
in torch and takes minutes to load). Read with shared lock, modify the
two specific exp_ids' status from 'failed' -> 'completed' and drop the
spurious 'error' field, write to a tmp file in the same directory, then
os.replace into place.

The records' actual results, model_hash, and dataset_hash are already
correct on disk — only the status/error labels need fixing.

Run with --dry-run first to review what would change. Re-run without it
to apply.

Usage:
  python3 scratch/recover_smoke_band_status.py --dry-run
  python3 scratch/recover_smoke_band_status.py
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REGISTRY = Path("/net/projects2/interp/subliminal/shared/results/registry.json")

# Records to recover. These are the 2 smoke band cells that hit the
# post-completion summary-print bug on 2026-05-11.
TARGETS = [
    "Eagles_subliminal_r1_seed1_range100_999_cat_band_qwen",
    "The Beatles_subliminal_r1_seed1_range100_999_cat_band_qwen",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change, don't write")
    args = parser.parse_args()

    # Open registry with exclusive lock for the read-modify-write window.
    # This is the same locking discipline benchmarks/storage.py uses.
    with open(REGISTRY, "r+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            reg = json.load(fp)
        finally:
            # We'll release the lock after the atomic replace below;
            # for now keep it held so no concurrent writer can race.
            pass

        exps = reg["experiments"]
        to_update: list[tuple[str, dict]] = []
        for eid in TARGETS:
            rec = exps.get(eid)
            if rec is None:
                print(f"  [SKIP missing]   {eid}", file=sys.stderr)
                continue
            status = rec.get("status")
            results = rec.get("results")
            # Only recover if the record genuinely has the post-completion
            # signature: status=failed, error=list-index, results=present.
            # Refuse to flip anything that doesn't match — safety net.
            ok = (
                status == "failed"
                and "list index out of range" in (rec.get("error") or "")
                and results
                and "generation_aggregate" in results
                and rec.get("model_hash")
                and rec.get("dataset_hash")
            )
            if not ok:
                print(f"  [SKIP guard]     {eid}  status={status!r} "
                      f"error={rec.get('error')!r} results={'present' if results else 'missing'}",
                      file=sys.stderr)
                continue
            new = dict(rec)
            new["status"] = "completed"
            new.pop("error", None)
            new["updated_at"] = datetime.now().isoformat()
            to_update.append((eid, new))
            print(f"  [{'WOULD RECOVER' if args.dry_run else 'RECOVER'}] {eid}",
                  file=sys.stderr)

        if not to_update:
            print("nothing to do", file=sys.stderr)
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            return 0

        if args.dry_run:
            print(f"\nDRY RUN — {len(to_update)} record(s) would change. "
                  f"Re-run without --dry-run to apply.", file=sys.stderr)
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            return 0

        for eid, new in to_update:
            exps[eid] = new

        # Atomic write via temp file + os.replace, mirroring
        # benchmarks/storage.py:_save_registry.
        tmp_path = REGISTRY.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp_path, "w") as out:
            json.dump(reg, out)
        os.replace(tmp_path, REGISTRY)
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        print(f"\nApplied {len(to_update)} update(s) to {REGISTRY}",
              file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
