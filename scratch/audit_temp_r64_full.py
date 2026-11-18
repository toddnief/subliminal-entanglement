"""Broader audit of r=64 constrained-decode temperature sweep.

The notebook (paper_figures.ipynb, Fig 11b cell) filters with
tseed ∈ {1, 42, 123} and gseed ∈ {1, 42, 123} but the config
(configs/temp.yaml) only ran tseed=42 × 3 gseeds = 3 per cell (84 total).

The notebook output shows up to 9 rows per (animal, T) — meaning the
registry actually contains experiments at multiple tseeds. This script
enumerates ALL completed r=64 constrain experiments and shows the
(animal, T, tseed, gseed) coverage matrix.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

ARTIFACTS_DIR = os.environ.get(
    "ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results"
)
REGISTRY = Path(ARTIFACTS_DIR) / "registry.json"

ANIMALS = ["cat", "eagle", "wolf", "owl"]
TEMPS = [0.0, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
TSEEDS = [1, 42, 123]
GSEEDS = [1, 42, 123]


def main() -> None:
    with REGISTRY.open() as f:
        reg = json.load(f)
    experiments = reg.get("experiments", {})

    # Index: (animal, T, tseed, gseed) -> list of (exp_id, status)
    by_cell: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    other_constrain_r64: list[tuple[str, str, dict]] = []

    for exp_id, rec in experiments.items():
        cfg = rec.get("config") or {}
        if cfg.get("lora_rank") != 64:
            continue
        if not cfg.get("restrict_generation_to_number_tokens"):
            continue
        if cfg.get("system_prompt_variant") != "subliminal":
            continue
        if cfg.get("animal") not in ANIMALS:
            continue
        if cfg.get("generation_strategy", "filtered") != "filtered":
            continue
        # Skip the dwg / svd ablations: notebook filters dwg_mode=full, svd_mode=full
        if cfg.get("dwg_mode") and cfg["dwg_mode"] != "full":
            continue
        if cfg.get("svd_mode") and cfg["svd_mode"] != "full":
            continue
        if cfg.get("full_finetuning"):
            continue

        a = cfg["animal"]
        t = cfg.get("generation_temperature")
        ts = cfg.get("training_seed")
        gs = cfg.get("generation_seed")
        status = rec.get("status", "?")
        if (
            t in TEMPS
            and ts in TSEEDS
            and gs in GSEEDS
        ):
            by_cell[(a, t, ts, gs)].append((exp_id, status))
        else:
            other_constrain_r64.append((exp_id, status, cfg))

    # Per-cell completion matrix at the (animal, T) level: count of distinct
    # (tseed, gseed) pairs with at least one COMPLETED entry.
    print("Completion matrix — distinct (tseed, gseed) pairs with status=completed per cell")
    print("(target full grid = 3 tseed × 3 gseed = 9; reduced grid = 1 tseed × 3 gseed = 3)")
    header = "animal".ljust(8) + " | " + " | ".join(f"{t:>4}" for t in TEMPS) + " | total"
    print(header)
    print("-" * len(header))
    full_missing: list[tuple] = []
    reduced_missing: list[tuple] = []  # only tseed=42 cells
    for a in ANIMALS:
        row = []
        animal_total = 0
        for t in TEMPS:
            present = 0
            for ts in TSEEDS:
                for gs in GSEEDS:
                    entries = by_cell.get((a, t, ts, gs), [])
                    if any(st == "completed" for _, st in entries):
                        present += 1
                    elif ts == 42:
                        reduced_missing.append((a, t, ts, gs))
            animal_total += present
            row.append(f"{present:>4}")
            for ts in TSEEDS:
                for gs in GSEEDS:
                    entries = by_cell.get((a, t, ts, gs), [])
                    if not any(st == "completed" for _, st in entries):
                        full_missing.append((a, t, ts, gs))
        print(a.ljust(8) + " | " + " | ".join(row) + f" | {animal_total:>3}")

    print(f"\nTotal completed cells across the full 4×7×3×3 = 252 grid: "
          f"{sum(1 for k, v in by_cell.items() if any(st=='completed' for _, st in v))}")

    print(f"\nMissing under the FULL 4×7×3×3 = 252 grid: {len(full_missing)}")
    print(f"Missing under the REDUCED 4×7×1×3 = 84 grid (tseed=42 only): "
          f"{len([m for m in full_missing if m[2] == 42])}")

    # Per-animal × T view of missing tseed=42 cells (the reduced grid)
    print("\nMissing cells in the REDUCED 4×7×1×3 grid (tseed=42 only):")
    miss_42 = [m for m in full_missing if m[2] == 42]
    if not miss_42:
        print("  (none — reduced grid is fully populated)")
    else:
        by_at = defaultdict(list)
        for a, t, ts, gs in miss_42:
            by_at[(a, t)].append(gs)
        for (a, t), gseeds in sorted(by_at.items()):
            print(f"  {a:<6} T={t:<4} missing gseed(s): {sorted(gseeds)}")

    # Same view for the FULL grid (for context if user wants 9/cell)
    print("\nMissing cells in the FULL 4×7×3×3 grid (all tseeds):")
    by_at = defaultdict(list)
    for a, t, ts, gs in full_missing:
        by_at[(a, t)].append((ts, gs))
    for (a, t), pairs in sorted(by_at.items()):
        print(f"  {a:<6} T={t:<4} ({len(pairs)} missing) -> {sorted(pairs)}")

    if other_constrain_r64:
        print(f"\nOther constrain r=64 experiments (off-grid params): "
              f"{len(other_constrain_r64)}")
        for exp_id, status, cfg in other_constrain_r64[:10]:
            print(
                f"  {exp_id}  status={status}  "
                f"ts={cfg.get('training_seed')} gs={cfg.get('generation_seed')} "
                f"T={cfg.get('generation_temperature')}"
            )


if __name__ == "__main__":
    main()
