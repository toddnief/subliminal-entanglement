"""Audit which (animal, T, gen_seed) cells of the r=64 constrained-decode
temperature sweep (configs/temp.yaml) are present in the shared registry.

Per the config:
  animals = [cat, eagle, wolf, owl]
  temps   = [0.0, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
  gen_seeds = [1, 42, 123]
  train_seed = 42 (only)
  rank = 64
  restrict_generation_to_number_tokens = True

exp_id pattern (see benchmarks/config.py::ExperimentConfig.get_id):
  {animal}_subliminal_r64_seed{gseed}_tseed42[_temp{T:g}]_constrain_qwen
  (temp suffix dropped when T == 1.0)

Prints a per-animal × per-T matrix of completed gen_seed counts and lists
the missing (animal, T, gen_seed) triples so we can target a fill sweep.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

ARTIFACTS_DIR = os.environ.get(
    "ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results"
)
REGISTRY = Path(ARTIFACTS_DIR) / "registry.json"

ANIMALS = ["cat", "eagle", "wolf", "owl"]
TEMPS = [0.0, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
GEN_SEEDS = [1, 42, 123]
TRAIN_SEED = 42


def expected_id(animal: str, t: float, gseed: int) -> str:
    parts = [animal, "subliminal", "r64", f"seed{gseed}", f"tseed{TRAIN_SEED}"]
    if t != 1.0:
        parts.append(f"temp{t:g}")
    parts.append("constrain")
    parts.append("qwen")
    return "_".join(parts)


def main() -> None:
    expected = {}
    for a in ANIMALS:
        for t in TEMPS:
            for s in GEN_SEEDS:
                expected[expected_id(a, t, s)] = (a, t, s)
    print(f"Expected total: {len(expected)} experiments")

    with REGISTRY.open() as f:
        reg = json.load(f)

    experiments = reg.get("experiments", {})
    found = {}
    for exp_id, rec in experiments.items():
        if exp_id in expected:
            found[exp_id] = rec.get("status", "?")

    print(f"Present in registry (any status): {len(found)}")
    by_status = defaultdict(int)
    for status in found.values():
        by_status[status] += 1
    print(f"  by status: {dict(by_status)}")

    completed = {eid for eid, st in found.items() if st == "completed"}
    print(f"Completed: {len(completed)}")

    print("\nCompleted gen_seed count per (animal, T) (target = 3):")
    header = "animal".ljust(8) + " | " + " | ".join(f"{t:>4}" for t in TEMPS) + " | total"
    print(header)
    print("-" * len(header))
    grand_missing = []
    for a in ANIMALS:
        row = []
        animal_total = 0
        for t in TEMPS:
            present = sum(
                1
                for s in GEN_SEEDS
                if expected_id(a, t, s) in completed
            )
            animal_total += present
            row.append(f"{present:>4}")
            for s in GEN_SEEDS:
                eid = expected_id(a, t, s)
                if eid not in completed:
                    grand_missing.append((a, t, s, found.get(eid, "absent")))
        print(a.ljust(8) + " | " + " | ".join(row) + f" | {animal_total:>3}")

    print(f"\nMissing cells: {len(grand_missing)}")
    # Group by status to spot stuck/failed vs never-attempted
    miss_by_status = defaultdict(list)
    for a, t, s, st in grand_missing:
        miss_by_status[st].append((a, t, s))
    for st, items in sorted(miss_by_status.items()):
        print(f"\n  status={st!r}: {len(items)}")
        for a, t, s in items[:30]:
            print(f"    {a:<6} T={t:<4} seed={s}")
        if len(items) > 30:
            print(f"    ... ({len(items) - 30} more)")


if __name__ == "__main__":
    main()
