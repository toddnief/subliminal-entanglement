"""Audit registry coverage for the temperature-sweep figures (Fig 11 and Fig 11b).

Fig 11:   rank=8 free-decode legacy sweep
          exp_id pattern: <animal>_subliminal_r8[_seed<G>][_tseed<T>][_temp<X>]_range100_999_qwen
          canonical grid: animals=[cat,eagle,wolf,owl] x temps x gen_seed in {1,42,123} x train_seed in {1,42,123}

Fig 11b:  rank=64 constrained-decode sweep (configs/temp.yaml)
          exp_id pattern: <animal>_subliminal_r64[_seed<G>][_tseed<T>][_temp<X>]_constrain_qwen
          canonical grid: animals=[cat,eagle,wolf,owl] x temps x gen_seed in {1,42,123} x train_seed=42

Prints missing (animal, T, train_seed, gen_seed) cells and aggregate counts.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

REGISTRY = Path(
    os.environ.get(
        "ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results"
    )
) / "registry.json"

ANIMALS = ["cat", "eagle", "wolf", "owl"]
TEMPS = [0.0, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
GEN_SEEDS = [1, 42, 123]
TRAIN_SEEDS_FIG11 = [1, 42, 123]
TRAIN_SEEDS_FIG11B = [42]

# Common pieces:
# - default training_seed in get_id is 1, so tseed1 is omitted ⇒ tseed_part = ""
# - default generation_temperature is 1.0, so temp1 is omitted ⇒ temp_part = ""
# - default generation_seed in the canonical sweep is non-None (1/42/123) ⇒ always present


def fmt_temp(t: float) -> str:
    # matches f"temp{t:g}" → 0.0→"temp0", 0.5→"temp0.5", 1.0→"" (omitted), 2.0→"temp2"
    if t == 1.0:
        return ""
    return f"temp{t:g}"


def fmt_tseed(s: int) -> str:
    if s == 1:
        return ""
    return f"tseed{s}"


def fig11_exp_id(animal: str, gen_seed: int, train_seed: int, temp: float) -> str:
    parts = [animal, "subliminal", "r8", f"seed{gen_seed}"]
    if fmt_tseed(train_seed):
        parts.append(fmt_tseed(train_seed))
    if fmt_temp(temp):
        parts.append(fmt_temp(temp))
    parts.append("range100_999")
    parts.append("qwen")
    return "_".join(parts)


def fig11b_exp_id(animal: str, gen_seed: int, train_seed: int, temp: float) -> str:
    parts = [animal, "subliminal", "r64", f"seed{gen_seed}"]
    if fmt_tseed(train_seed):
        parts.append(fmt_tseed(train_seed))
    if fmt_temp(temp):
        parts.append(fmt_temp(temp))
    parts.append("constrain")
    parts.append("qwen")
    return "_".join(parts)


def main() -> None:
    print(f"Loading {REGISTRY} ...")
    with REGISTRY.open() as f:
        reg = json.load(f)
    experiments = reg.get("experiments", {})
    print(f"  → {len(experiments)} total experiments in registry\n")

    # Build a quick set of completed exp_ids
    completed = set()
    statuses = defaultdict(int)
    for exp_id, rec in experiments.items():
        st = rec.get("status", "unknown")
        statuses[st] += 1
        if st == "completed":
            completed.add(exp_id)
    print(f"Status counts (registry-wide): {dict(statuses)}\n")

    # ---------------------- Fig 11 audit ----------------------
    print("=" * 72)
    print("Fig 11 — rank=8 free-decode (range100_999) — 3x3 seed grid")
    print("=" * 72)
    fig11_missing: list[tuple[str, float, int, int, str]] = []
    fig11_present: list[tuple[str, float, int, int, str]] = []
    fig11_other = []  # in registry but not completed
    for animal in ANIMALS:
        for temp in TEMPS:
            for tseed in TRAIN_SEEDS_FIG11:
                for gseed in GEN_SEEDS:
                    eid = fig11_exp_id(animal, gseed, tseed, temp)
                    if eid in completed:
                        fig11_present.append((animal, temp, tseed, gseed, eid))
                    elif eid in experiments:
                        fig11_other.append(
                            (animal, temp, tseed, gseed, eid, experiments[eid].get("status"))
                        )
                    else:
                        fig11_missing.append((animal, temp, tseed, gseed, eid))

    total = len(ANIMALS) * len(TEMPS) * len(TRAIN_SEEDS_FIG11) * len(GEN_SEEDS)
    print(f"completed: {len(fig11_present)} / {total}")
    print(f"present-but-not-completed: {len(fig11_other)}")
    print(f"missing (not in registry):  {len(fig11_missing)}\n")

    # Per-(animal, temp) completed counts
    cell = defaultdict(int)
    for animal, temp, _, _, _ in fig11_present:
        cell[(animal, temp)] += 1
    print("Per-(animal, T) completed counts (target=9):")
    header = "  animal  | " + " | ".join(f"T={t:>4}" for t in TEMPS) + " | total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for animal in ANIMALS:
        cells = [cell.get((animal, t), 0) for t in TEMPS]
        print(
            f"  {animal:7s} | "
            + " | ".join(f"{c:>5d}" for c in cells)
            + f" | {sum(cells):>5d}"
        )

    if fig11_other:
        print("\nFig 11 present-but-not-completed (status):")
        for animal, temp, tseed, gseed, eid, st in fig11_other:
            print(f"  [{st}] {eid}")

    if fig11_missing:
        print("\nFig 11 missing (not in registry):")
        for animal, temp, tseed, gseed, eid in fig11_missing:
            print(f"  animal={animal:6s} T={temp} tseed={tseed} gseed={gseed} -> {eid}")

    # ---------------------- Fig 11b audit ----------------------
    print()
    print("=" * 72)
    print("Fig 11b — rank=64 constrained-decode (constrain) — 1×3 seed grid (tseed=42)")
    print("=" * 72)
    fig11b_missing = []
    fig11b_present = []
    fig11b_other = []
    for animal in ANIMALS:
        for temp in TEMPS:
            for tseed in TRAIN_SEEDS_FIG11B:
                for gseed in GEN_SEEDS:
                    eid = fig11b_exp_id(animal, gseed, tseed, temp)
                    if eid in completed:
                        fig11b_present.append((animal, temp, tseed, gseed, eid))
                    elif eid in experiments:
                        fig11b_other.append(
                            (animal, temp, tseed, gseed, eid, experiments[eid].get("status"))
                        )
                    else:
                        fig11b_missing.append((animal, temp, tseed, gseed, eid))

    total_b = len(ANIMALS) * len(TEMPS) * len(TRAIN_SEEDS_FIG11B) * len(GEN_SEEDS)
    print(f"completed: {len(fig11b_present)} / {total_b}")
    print(f"present-but-not-completed: {len(fig11b_other)}")
    print(f"missing (not in registry):  {len(fig11b_missing)}\n")

    cell_b = defaultdict(int)
    for animal, temp, _, _, _ in fig11b_present:
        cell_b[(animal, temp)] += 1
    print("Per-(animal, T) completed counts (target=3):")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for animal in ANIMALS:
        cells = [cell_b.get((animal, t), 0) for t in TEMPS]
        print(
            f"  {animal:7s} | "
            + " | ".join(f"{c:>5d}" for c in cells)
            + f" | {sum(cells):>5d}"
        )

    if fig11b_other:
        print("\nFig 11b present-but-not-completed (status):")
        for animal, temp, tseed, gseed, eid, st in fig11b_other:
            print(f"  [{st}] {eid}")

    if fig11b_missing:
        print("\nFig 11b missing (not in registry):")
        for animal, temp, tseed, gseed, eid in fig11b_missing:
            print(f"  animal={animal:6s} T={temp} tseed={tseed} gseed={gseed} -> {eid}")


def scan_all_r8_subliminal_temp() -> None:
    """Empirical scan: walk the registry and find every completed rank-8
    subliminal experiment whose config looks like the temperature sweep
    (animal in ANIMALS, training_seed in {1,42,123}, generation_seed in
    {1,42,123}, generation_strategy='filtered', system_prompt_variant=
    'subliminal', dwg_mode='full', svd_mode='full', full_ft=False, no
    dataset_path, lora_rank=8). This is what filter_gen_df() effectively
    matches in the notebook for Fig 11 — it filters by stored config
    fields, NOT by exp_id substring.
    """
    print()
    print("=" * 72)
    print("Empirical scan — every completed r=8 subliminal sweep row")
    print("(matches the notebook's filter_gen_df contract for Fig 11)")
    print("=" * 72)

    with REGISTRY.open() as f:
        reg = json.load(f)
    experiments = reg.get("experiments", {})

    found = defaultdict(list)  # (animal, T) -> list of (exp_id, tseed, gseed, num_min, num_max)
    for exp_id, rec in experiments.items():
        if rec.get("status") != "completed":
            continue
        cfg = rec.get("config", {})
        if cfg.get("animal") not in ANIMALS:
            continue
        if cfg.get("system_prompt_variant") != "subliminal":
            continue
        if cfg.get("generation_strategy") != "filtered":
            continue
        if cfg.get("lora_rank") != 8:
            continue
        if cfg.get("full_finetuning"):
            continue
        if cfg.get("dwg_mode", "full") != "full":
            continue
        if cfg.get("svd_mode", "full") != "full":
            continue
        if cfg.get("dataset_path"):
            continue
        if cfg.get("restrict_generation_to_number_tokens"):
            continue  # rank-8 constrain entries are a separate sweep
        if cfg.get("training_seed") not in TRAIN_SEEDS_FIG11:
            continue
        if cfg.get("generation_seed") not in GEN_SEEDS:
            continue
        if cfg.get("train_system_prompt") is not None:
            continue
        if cfg.get("eval_system_prompt") is not None:
            continue
        if cfg.get("optimizer", "adamw") != "adamw":
            continue
        if cfg.get("n_epochs") != 3:
            continue
        T = cfg.get("generation_temperature")
        if T not in TEMPS:
            continue
        num_min = cfg.get("number_min")
        num_max = cfg.get("number_max")
        found[(cfg["animal"], T)].append(
            (exp_id, cfg.get("training_seed"), cfg.get("generation_seed"), num_min, num_max)
        )

    print("Per-(animal, T) row count (any number range):")
    header = "  animal  | " + " | ".join(f"T={t:>4}" for t in TEMPS) + " | total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for animal in ANIMALS:
        cells = [len(found.get((animal, t), [])) for t in TEMPS]
        print(
            f"  {animal:7s} | "
            + " | ".join(f"{c:>5d}" for c in cells)
            + f" | {sum(cells):>5d}"
        )

    # Break down number-range usage by (animal, T)
    print("\nNumber-range breakdown per (animal, T):")
    print(header.replace("total", "ranges"))
    print("  " + "-" * (len(header) - 2))
    for animal in ANIMALS:
        for t in TEMPS:
            entries = found.get((animal, t), [])
            ranges = defaultdict(int)
            for _eid, _ts, _gs, num_min, num_max in entries:
                ranges[(num_min, num_max)] += 1
            if ranges:
                summary = ", ".join(f"({a},{b}):{c}" for (a, b), c in sorted(ranges.items()))
                print(f"  {animal:7s} T={t:>4}: {summary}")

    # Union of (tseed, gseed) cells across both number ranges per (animal, T).
    # This is what the notebook actually sees after drop_duplicates(
    #   subset=[animal, training_seed, generation_seed, generation_temperature, rank]).
    # Also track in-flight cells (status="running"): these will land on their
    # own and shouldn't be re-submitted in a fill-in config.
    inflight_cells: set[tuple[str, float, int, int]] = set()
    with REGISTRY.open() as f:
        reg2 = json.load(f)
    for exp_id, rec in reg2.get("experiments", {}).items():
        if rec.get("status") != "running":
            continue
        cfg = rec.get("config", {})
        if cfg.get("animal") not in ANIMALS:
            continue
        if cfg.get("system_prompt_variant") != "subliminal":
            continue
        if cfg.get("lora_rank") != 8:
            continue
        if cfg.get("full_finetuning"):
            continue
        if cfg.get("dwg_mode", "full") != "full":
            continue
        if cfg.get("svd_mode", "full") != "full":
            continue
        if cfg.get("dataset_path"):
            continue
        if cfg.get("restrict_generation_to_number_tokens"):
            continue
        if cfg.get("training_seed") not in TRAIN_SEEDS_FIG11:
            continue
        if cfg.get("generation_seed") not in GEN_SEEDS:
            continue
        if cfg.get("train_system_prompt") is not None:
            continue
        if cfg.get("eval_system_prompt") is not None:
            continue
        T = cfg.get("generation_temperature")
        if T not in TEMPS:
            continue
        inflight_cells.add(
            (cfg["animal"], T, cfg.get("training_seed"), cfg.get("generation_seed"))
        )

    print(f"\nIn-flight (status='running') cells covered: {len(inflight_cells)}")
    by_animal_temp: dict[tuple[str, float], list[tuple[int, int]]] = defaultdict(list)
    for animal, t, ts, gs in inflight_cells:
        by_animal_temp[(animal, t)].append((ts, gs))
    for (animal, t), cells in sorted(by_animal_temp.items()):
        print(f"  in-flight: {animal:7s} T={t:>4}: {sorted(cells)}")

    print("\nDeduped coverage per (animal, T) (target=9; matches notebook after dedupe):")
    print(header)
    print("  " + "-" * (len(header) - 2))
    fig11_dedupe_missing: list[tuple[str, float, int, int]] = []
    for animal in ANIMALS:
        cells_summary = []
        for t in TEMPS:
            entries = found.get((animal, t), [])
            covered = set((ts, gs) for _eid, ts, gs, _nm, _nx in entries)
            cells_summary.append(len(covered))
            full = set((ts, gs) for ts in TRAIN_SEEDS_FIG11 for gs in GEN_SEEDS)
            for ts, gs in sorted(full - covered):
                fig11_dedupe_missing.append((animal, t, ts, gs))
        print(
            f"  {animal:7s} | "
            + " | ".join(f"{c:>5d}" for c in cells_summary)
            + f" | {sum(cells_summary):>5d}"
        )

    print(f"\nFig 11 deduped missing cells: {len(fig11_dedupe_missing)}")
    if fig11_dedupe_missing:
        print("(animal, T, training_seed, generation_seed):")
        for animal, t, ts, gs in fig11_dedupe_missing:
            print(f"  {animal:7s} T={t} tseed={ts} gseed={gs}")

    # Write out missing cells as YAML-friendly grids per animal so we can
    # turn them into a fill-in config easily. Skip cells that are already in
    # flight (status="running") — those will land without re-submission.
    print("\nGrouped by (animal, T) - cells needed (excluding in-flight):")
    grouped: dict[str, dict[float, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    fill_total = 0
    for animal, t, ts, gs in fig11_dedupe_missing:
        if (animal, t, ts, gs) in inflight_cells:
            continue
        grouped[animal][t].append((ts, gs))
        fill_total += 1
    for animal in ANIMALS:
        if animal not in grouped:
            continue
        print(f"  {animal}:")
        for t in sorted(grouped[animal]):
            cells = grouped[animal][t]
            print(f"    T={t}: {cells}")
    print(f"\nTotal fill-in experiments needed (excluding in-flight): {fill_total}")

    return found


def spot_check_fig11b() -> None:
    """Verify one cached rank-64 constrain entry has the cfg fields that
    filter_gen_df expects (dwg_mode='full', svd_mode='full', dataset_source
    'filtered (batch)', etc). If yes, the notebook just needs to reload gen_df."""
    print()
    print("=" * 72)
    print("Spot-check: one rank-64 constrain entry to verify filter_gen_df match")
    print("=" * 72)
    with REGISTRY.open() as f:
        reg = json.load(f)
    sample_id = "cat_subliminal_r64_seed1_tseed42_temp0_constrain_qwen"
    rec = reg["experiments"].get(sample_id)
    if not rec:
        print(f"  (sample exp_id not found: {sample_id})")
        return
    cfg = rec.get("config", {})
    relevant = {
        "status": rec.get("status"),
        "animal": cfg.get("animal"),
        "system_prompt_variant": cfg.get("system_prompt_variant"),
        "generation_strategy": cfg.get("generation_strategy"),
        "dataset_path": cfg.get("dataset_path"),
        "lora_rank": cfg.get("lora_rank"),
        "full_finetuning": cfg.get("full_finetuning"),
        "training_seed": cfg.get("training_seed"),
        "generation_seed": cfg.get("generation_seed"),
        "generation_temperature": cfg.get("generation_temperature"),
        "train_system_prompt": cfg.get("train_system_prompt"),
        "eval_system_prompt": cfg.get("eval_system_prompt"),
        "dwg_mode": cfg.get("dwg_mode"),
        "svd_mode": cfg.get("svd_mode"),
        "restrict_generation_to_number_tokens": cfg.get("restrict_generation_to_number_tokens"),
        "number_min": cfg.get("number_min"),
        "number_max": cfg.get("number_max"),
    }
    for k, v in relevant.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
    scan_all_r8_subliminal_temp()
    spot_check_fig11b()
