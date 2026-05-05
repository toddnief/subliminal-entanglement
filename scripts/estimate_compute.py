"""Estimate compute time for benchmark experiments for the paper.

We combine two complementary measurements:

1. **Per-experiment durations from the registry.** Each experiment carries
   ``created_at`` (set when status flips to ``running``) and ``updated_at``
   (refreshed on every ``update_experiment`` call). Their difference is a
   per-experiment wall-clock estimate.

   *Caveat:* batch backfills (e.g. ``scripts/backfill_animal_counts.py``)
   re-write ``updated_at`` for many entries at once, inflating the apparent
   duration by days/weeks. SLURM jobs are capped at 10h, so we drop
   experiments whose registry-reported duration exceeds that — those rows
   are backfill artifacts, not true runtime.

2. **Total SLURM wall-time from the log files.** The ``logs/benchmark*.out``
   files include ``Starting benchmark at <date>`` and ``Finished at <date>``
   lines (single-experiment jobs) and ``Started at <date>`` / ``Finished at
   <date>`` lines (parallel array jobs). Summing those across all log files
   gives the total GPU-hours actually allocated by SLURM, regardless of how
   many experiments completed inside each job. Each job uses exactly one GPU
   (per the SBATCH header), so allocated GPU-hours == allocated wall-hours.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from sl import config as sl_config


SLURM_JOB_LIMIT_S = 10 * 3600  # SBATCH --time=10:00:00


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def build_durations_df(reg: dict) -> pd.DataFrame:
    rows = []
    for exp_id, data in reg.get("experiments", {}).items():
        created = _parse_iso(data.get("created_at"))
        updated = _parse_iso(data.get("updated_at"))
        if created is None or updated is None:
            continue
        cfg = data.get("config", {})
        duration_s = max((updated - created).total_seconds(), 0.0)
        rows.append({
            "exp_id": exp_id,
            "status": data.get("status"),
            "created_at": created,
            "updated_at": updated,
            "duration_s": duration_s,
            "animal": cfg.get("animal"),
            "lora_rank": cfg.get("lora_rank"),
            "n_epochs": cfg.get("n_epochs"),
            "dataset_size": cfg.get("dataset_size"),
            "full_ft": bool(cfg.get("full_finetuning")),
            "student_model": cfg.get("student_model"),
            "svd_mode": cfg.get("svd_mode") or "full",
            "dwg_mode": cfg.get("dwg_mode") or "full",
        })
    return pd.DataFrame(rows)


def _fmt_hours(seconds: float) -> str:
    hours = seconds / 3600.0
    if hours < 1:
        return f"{seconds / 60:.1f} min"
    if hours >= 1000:
        return f"{hours:,.0f} h"
    return f"{hours:.1f} h"


def _stats_block(label: str, durations: np.ndarray) -> str:
    if len(durations) == 0:
        return f"{label}: no data"
    qs = np.quantile(durations, [0.25, 0.5, 0.75, 0.95])
    return (
        f"{label}: n={len(durations):>5}  "
        f"median={qs[1]/60:6.1f} min  "
        f"p25={qs[0]/60:5.1f}  p75={qs[2]/60:6.1f}  p95={qs[3]/60:6.1f}  "
        f"mean={durations.mean()/60:6.1f}  "
        f"sum={_fmt_hours(durations.sum())}"
    )


# ----- Registry-based per-experiment estimate -----------------------------

def report_registry_durations(reg: dict) -> pd.DataFrame:
    df = build_durations_df(reg)
    completed = df[df["status"] == "completed"].copy()

    n_total = len(completed)
    n_capped = int((completed["duration_s"] > SLURM_JOB_LIMIT_S).sum())

    # The 10h cap drops backfill artifacts. Anything > 10h cannot be a true
    # single-experiment runtime since SLURM jobs are time-limited.
    valid = completed[completed["duration_s"] <= SLURM_JOB_LIMIT_S].copy()

    print("\n" + "=" * 78)
    print(f"REGISTRY DURATIONS  (completed={n_total}, "
          f"dropped >10h backfill artifacts={n_capped}, "
          f"valid={len(valid)})")
    print("=" * 78)

    durations = valid["duration_s"].to_numpy()
    print(_stats_block("all valid", durations))

    print(f"\nExperiments under 60s (likely cache-only re-runs): "
          f"{int((durations < 60).sum())}")
    print(f"Experiments under 5 min:                           "
          f"{int((durations < 300).sum())}")

    print("\n— BY TRAINING TYPE —")
    for full_ft, g in valid.groupby("full_ft"):
        label = "full-FT" if full_ft else "LoRA"
        print(_stats_block(f"{label:>8}", g["duration_s"].to_numpy()))

    print("\n— BY DATASET SIZE —")
    for size, g in valid.groupby("dataset_size", dropna=False):
        print(_stats_block(f"size={size}", g["duration_s"].to_numpy()))

    print("\n— BY EVAL FLAVOR (svd_mode / dwg_mode) —")
    print("(canonical training run = svd=full, dwg=full; non-full are "
          "post-hoc edits to a cached adapter and skip training)")
    for (svd, dwg), g in valid.groupby(["svd_mode", "dwg_mode"]):
        d = g["duration_s"].to_numpy()
        print(_stats_block(f"svd={svd:<6} dwg={dwg:<20}", d))

    return valid


# ----- SLURM log allocated time ------------------------------------------

_DATE_PATTERNS = [
    # "Started at" / "Starting benchmark at" / "Finished at" with full date.
    re.compile(
        r"^(?:Started at|Starting benchmark at|Finished at)\s+"
        r"(?P<dt>\w{3}\s+\w{3}\s+\d+\s+\d+:\d+:\d+\s+(?:AM|PM)\s+\w+\s+\d{4})\s*$"
    ),
]


def _parse_log_date(line: str) -> datetime | None:
    """Parse ``Mon Apr 15 08:30:18 PM CDT 2026`` style timestamps.

    Both ``date`` formats produced by the SLURM scripts use
    ``%a %b %d %I:%M:%S %p %Z %Y`` (12-hour clock with AM/PM). We strip the
    timezone abbreviation since Python can't reliably parse those.
    """
    for pat in _DATE_PATTERNS:
        m = pat.match(line.strip())
        if not m:
            continue
        raw = m.group("dt")
        # Drop the timezone abbreviation (e.g. "CDT") — Python doesn't have a
        # good %Z parser. We compare wall-clock times only on the same machine,
        # so DST transitions are the only edge case (1h drift twice a year).
        parts = raw.split()
        if len(parts) >= 8:  # Mon Apr 15 08:30:18 PM CDT 2026 → 8 tokens
            no_tz = " ".join(parts[:6] + parts[7:])
            try:
                return datetime.strptime(no_tz, "%a %b %d %I:%M:%S %p %Y")
            except ValueError:
                pass
        try:
            return datetime.strptime(raw, "%a %b %d %I:%M:%S %p %Z %Y")
        except ValueError:
            return None
    return None


def report_slurm_allocated_time(logs_dir: Path) -> None:
    out_files = sorted(logs_dir.glob("benchmark*.out"))
    n_files = 0
    n_complete_pairs = 0
    n_unfinished = 0
    durations: list[float] = []
    unfinished: list[Path] = []

    for path in out_files:
        n_files += 1
        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError:
            continue
        start = end = None
        # Walk in order: first matching start line, last matching finish line.
        for line in lines:
            ts = _parse_log_date(line)
            if ts is None:
                continue
            stripped = line.strip()
            if (stripped.startswith("Started at")
                    or stripped.startswith("Starting benchmark at")):
                if start is None:
                    start = ts
            elif stripped.startswith("Finished at"):
                end = ts
        if start is not None and end is not None:
            durations.append((end - start).total_seconds())
            n_complete_pairs += 1
        elif start is not None:
            n_unfinished += 1
            unfinished.append(path)

    durations_arr = np.array(durations)

    print("\n" + "=" * 78)
    print(f"SLURM ALLOCATED WALL-TIME  (from {n_files} log files)")
    print("=" * 78)
    print(f"  pairs with start+finish:    {n_complete_pairs}")
    print(f"  start only (incl. running): {n_unfinished}")
    if len(durations_arr):
        print(f"  total allocated GPU-hours:  {_fmt_hours(durations_arr.sum())}")
        print(f"  per-job median:             {durations_arr.mean()/60:.1f} min  "
              f"(p50={np.median(durations_arr)/60:.1f}, "
              f"p95={np.quantile(durations_arr, 0.95)/60:.1f})")
    if unfinished:
        print(f"  (tip: {len(unfinished)} jobs have no 'Finished at' line — "
              "either still running, killed, or out of time)")


def main():
    reg_path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    logger.info(f"Loading registry from {reg_path}")
    with open(reg_path) as f:
        reg = json.load(f)
    logger.info(
        f"Registry: {len(reg.get('experiments', {}))} experiments total"
    )

    valid = report_registry_durations(reg)

    repo_root = Path(__file__).resolve().parent.parent
    report_slurm_allocated_time(repo_root / "logs")

    print("\n" + "=" * 78)
    print("PAPER-READY SUMMARY")
    print("=" * 78)
    n = len(valid)
    median_min = valid["duration_s"].median() / 60
    total_h = valid["duration_s"].sum() / 3600
    lora = valid[~valid["full_ft"]]
    full_ft = valid[valid["full_ft"]]
    print(
        f"- {n:,} completed experiments with reliable timestamps "
        f"({len(lora):,} LoRA, {len(full_ft):,} full-FT)"
    )
    print(
        f"- Median per-experiment runtime: {median_min:.1f} min "
        f"(LoRA median: {lora['duration_s'].median()/60:.1f} min, "
        f"full-FT median: "
        f"{full_ft['duration_s'].median()/60:.1f} min)"
    )
    print(
        f"- Sum of valid per-experiment durations: {total_h:,.0f} GPU-hours"
    )
    print(
        "- Hardware: single A100 80GB / H100 / H200 per experiment "
        "(see slurm/run_benchmark*.sh)"
    )


if __name__ == "__main__":
    main()
