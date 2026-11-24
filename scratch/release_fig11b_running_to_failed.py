"""Flip stuck `status="running"` registry entries for the 6 Fig 11b r=8
masked-decode T=2.0 fill cells -> `status="failed"`, so that
`./submit.sh benchmark-parallel` (via scripts/check_cached.py) treats them as
re-submittable rather than as in-flight jobs.

Why this is needed: when a SLURM job dies before writing back its registry
completion (preemption, walltime, node crash), the experiment record stays
pinned at `status="running"` with empty `dataset_hash` / `model_hash` and
`results=null`. `check_cached.py` then considers the cell "covered" and
silently drops it from the submission grid, so the cell never lands. Flipping
to `failed` releases the cache gate.

Safety guards (a record is flipped ONLY when ALL hold):
  1. exp_id is in the explicit TARGETS list below (6 cells; nothing else).
  2. status == "running".
  3. dataset_hash == "" and model_hash == "" (the orphan signature; a
     genuinely-in-progress job that already produced an artifact would not
     match, so we refuse to clobber it).
  4. results is null.
  5. The record's `updated_at` (falling back to `created_at`) is more than
     --min-age-hours old (default 6h, ~half the 10h SLURM time limit). A
     concurrent run that started within the last few hours could still
     legitimately be alive even if status=running.

Run with --dry-run first to review what would change. Re-run without it to
apply (with file locking + atomic temp-file replace, mirroring
`benchmarks/storage.py:_save_registry`).

Usage:
  python3 scratch/release_fig11b_running_to_failed.py --dry-run
  python3 scratch/release_fig11b_running_to_failed.py            # default 6h
  python3 scratch/release_fig11b_running_to_failed.py --min-age-hours 12
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

REGISTRY = Path(
    os.environ.get(
        "ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results"
    )
) / "registry.json"

# The 6 cells targeted by the Fig 11b r=8 masked-decode T=2.0 fill. Must match
# the exp_id template used by benchmarks/config.py::ExperimentConfig.get_id:
#   {animal}_subliminal_r8_seed{G}_tseed42_temp2_constrain_qwen
# (number_min=100, number_max=1000 → no "range" suffix; temp=2.0 → "temp2";
# tseed=42 → "tseed42"; restrict_generation_to_number_tokens=true → "_constrain").
TARGETS = [
    "cat_subliminal_r8_seed123_tseed42_temp2_constrain_qwen",
    "wolf_subliminal_r8_seed123_tseed42_temp2_constrain_qwen",
    "eagle_subliminal_r8_seed1_tseed42_temp2_constrain_qwen",
    "eagle_subliminal_r8_seed123_tseed42_temp2_constrain_qwen",
    "owl_subliminal_r8_seed1_tseed42_temp2_constrain_qwen",
    "owl_subliminal_r8_seed123_tseed42_temp2_constrain_qwen",
]


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    # Strip any trailing Z; fromisoformat() doesn't accept it pre-3.11.
    s = ts.rstrip("Z")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change, don't write.",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=6.0,
        help=(
            "Only flip records whose updated_at (or created_at) is at least "
            "this many hours old. Default 6h. Set to 0 to flip regardless of "
            "age (NOT recommended; could clobber a currently-running job)."
        ),
    )
    args = parser.parse_args()

    now = datetime.now()
    min_age = timedelta(hours=args.min_age_hours)

    # Open registry with exclusive lock for the read-modify-write window.
    # Same locking discipline as benchmarks/storage.py:_save_registry.
    with open(REGISTRY, "r+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

        reg = json.load(fp)
        exps = reg["experiments"]

        to_update: list[tuple[str, dict]] = []
        for eid in TARGETS:
            rec = exps.get(eid)
            if rec is None:
                print(f"  [SKIP missing]   {eid}", file=sys.stderr)
                continue

            status = rec.get("status")
            dataset_hash = rec.get("dataset_hash", "")
            model_hash = rec.get("model_hash", "")
            results = rec.get("results")
            updated_at = rec.get("updated_at") or rec.get("created_at")
            ts = _parse_iso(updated_at)
            age = (now - ts) if ts else None

            # Apply all safety guards together so the rejection reason is
            # printed exactly once.
            guards = {
                "status==running": status == "running",
                "dataset_hash empty": dataset_hash == "",
                "model_hash empty": model_hash == "",
                "results==null": results is None,
                "age >= min": (age is not None and age >= min_age),
            }
            if not all(guards.values()):
                failing = [k for k, v in guards.items() if not v]
                age_str = (
                    f"{age.total_seconds() / 3600:.1f}h" if age else "unknown"
                )
                print(
                    f"  [SKIP guard]     {eid}  status={status!r} age={age_str} "
                    f"failing=[{', '.join(failing)}]",
                    file=sys.stderr,
                )
                continue

            new = dict(rec)
            new["status"] = "failed"
            new["updated_at"] = now.isoformat()
            # Leave a breadcrumb so a future debugger can tell this wasn't a
            # real pipeline failure. Doesn't disturb anything else.
            new["error"] = (
                "released_from_stuck_running_by_scratch/release_fig11b_running_to_failed.py"
            )
            to_update.append((eid, new))
            age_str = f"{age.total_seconds() / 3600:.1f}h"
            verb = "WOULD FLIP" if args.dry_run else "FLIP"
            print(
                f"  [{verb}] {eid}  age={age_str}  (was running -> failed)",
                file=sys.stderr,
            )

        if not to_update:
            print("nothing to do", file=sys.stderr)
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            return 0

        if args.dry_run:
            print(
                f"\nDRY RUN — {len(to_update)} record(s) would change. "
                f"Re-run without --dry-run to apply.",
                file=sys.stderr,
            )
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
        print(
            f"\nApplied {len(to_update)} update(s) to {REGISTRY}",
            file=sys.stderr,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
