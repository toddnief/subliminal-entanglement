#!/usr/bin/env python3
"""Audit per-(animal, scenario, rank) coverage for the 9-cell Cartesian
train_sys × eval_sys product across cat/owl/eagle/wolf at ranks 2..512."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sl.results import load_registry, build_gen_df, filter_gen_df
from sl.tables import CHATGPT_SYSTEM_PROMPT


ANIMALS = ["cat", "owl", "eagle", "wolf"]
RANKS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
TRAIN_SYS = ["<none>", CHATGPT_SYSTEM_PROMPT, ""]
EVAL_SYS = ["<none>", CHATGPT_SYSTEM_PROMPT, ""]


def pretty(s):
    if s == "<none>":
        return "Qwen"
    if s == "":
        return "empty"
    if s == CHATGPT_SYSTEM_PROMPT:
        return "ChatGPT"
    return s[:8]


def resolve_variants(train_sys, eval_sys):
    if train_sys == "<none>" and eval_sys == "<none>":
        return "subliminal"
    return None


def main():
    reg = load_registry()
    gen_df = build_gen_df(reg)

    print("Each cell shows seeds_present (target=9 = 3x3 train_seed x gen_seed grid)")
    print()

    grand_total = {}
    per_cell_missing = []

    for animal in ANIMALS:
        print(f"=== {animal} ===")
        for tsys in TRAIN_SYS:
            for esys in EVAL_SYS:
                v = resolve_variants(tsys, esys)
                sub = filter_gen_df(
                    gen_df,
                    animals=animal,
                    variants=v,
                    dataset_source="filtered (batch)",
                    training_seeds=[1, 42, 123],
                    generation_seeds=[1, 42, 123],
                    generation_temperature=1.0,
                    eval_setting=("clean" if esys == "<none>" else "with_system"),
                    train_system_prompt=tsys,
                    eval_system_prompt=esys,
                    dwg_mode="full",
                    svd_mode="full",
                    full_ft=False,
                )
                sub = (
                    sub.sort_values("model_hash")
                    .drop_duplicates(
                        ["training_seed", "generation_seed", "rank"], keep="last"
                    )
                )
                counts = sub.groupby("rank").size().reindex(RANKS, fill_value=0)
                line = f"  T={pretty(tsys):>7s} E={pretty(esys):>7s}: "
                line += "  ".join(f"r{r}={int(c)}" for r, c in zip(RANKS, counts.values))
                print(line)

                for r in RANKS:
                    have = int(counts.loc[r])
                    missing = 9 - have
                    if missing > 0:
                        per_cell_missing.append(
                            (animal, pretty(tsys), pretty(esys), r, have, missing)
                        )
        print()

    total_missing_seeds = sum(m for *_, m in per_cell_missing)
    total_cells = len(ANIMALS) * len(TRAIN_SYS) * len(EVAL_SYS) * len(RANKS)
    incomplete_cells = len(per_cell_missing)
    print("=" * 60)
    print(f"Total cells (animal x t_sys x e_sys x rank): {total_cells}")
    print(f"Cells with <9 seeds: {incomplete_cells}")
    print(f"Sum of missing seed-evals: {total_missing_seeds}")
    print()

    print("=== Missing breakdown (rolled up to (train_sys, eval_sys, rank)) ===")
    from collections import defaultdict
    rollup = defaultdict(lambda: 0)
    rollup_cells = defaultdict(lambda: 0)
    for animal, t, e, r, have, missing in per_cell_missing:
        rollup[(t, e, r)] += missing
        rollup_cells[(t, e, r)] += 1
    for (t, e, r), n in sorted(rollup.items()):
        cells = rollup_cells[(t, e, r)]
        print(f"  T={t:>7s} E={e:>7s} r={r:>3d}: {n} missing seed-evals across {cells} animals")


if __name__ == "__main__":
    main()
