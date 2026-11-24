"""Audit gen_df for cat/eagle gaps in Table 2 (SYS_VARIANT_SCENARIOS).

Reads the cached gen_df parquet view and tallies, per (animal, variant, rank),
how many runs exist at the priority 1x6 seed grid (train_seed=42 x gen_seeds
{1, 42, 123, 7, 11, 13}). Prints a coverage matrix and the explicit
(animal, variant, rank, train_seed, gen_seed) cells that still need to be run.
"""
from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

ART = Path(os.environ.get("ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results"))
GEN = ART / "views" / "gen_df.parquet"

NON_CANONICAL = [
    "train_claude_eval_claude",
    "train_llm_eval_llm",
    "no_entity",
    "sys_train_prefix_eval",
    "prefix_train_sys_eval",
]

ANIMALS = ["cat", "eagle"]
TRAIN_SEEDS = [42]
GEN_SEEDS = [1, 42, 123, 7, 11, 13]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

df = pd.read_parquet(GEN)
print("== gen_df schema sanity ==")
print(f"  total rows: {len(df)}")
print(f"  variants seen: {sorted(df['variant'].dropna().unique())}")
print()

# How many rows exist for each (animal, variant) overall in cat/eagle?
print("== Raw row count by (animal, variant), no seed/rank filter ==")
raw = df[df["animal"].isin(ANIMALS) & df["variant"].isin(NON_CANONICAL)]
print(raw.groupby(["animal", "variant"]).size().to_string())
print()
if not raw.empty:
    print("== Sample row (first cat/eagle non-canonical hit) ==")
    interesting_cols = [
        "animal", "variant", "rank", "training_seed", "generation_seed",
        "dataset_source", "train_system_prompt", "eval_system_prompt",
        "eval_setting", "dwg_mode", "svd_mode", "full_ft", "generation_temperature",
    ]
    cols = [c for c in interesting_cols if c in raw.columns]
    print(raw[cols].head(3).to_string())
    print()

# Apply the same filter the table call site uses
filtered = df[
    df["animal"].isin(ANIMALS)
    & df["variant"].isin(NON_CANONICAL)
    & (df["dataset_source"] == "filtered (batch)")
    & (df["generation_temperature"] == 1.0)
    & (df["dwg_mode"] == "full")
    & (df["svd_mode"] == "full")
    & (df["full_ft"] == False)
    & df["training_seed"].isin(TRAIN_SEEDS)
    & df["generation_seed"].isin(GEN_SEEDS)
]
print("== Filtered (animal in {cat,eagle}, priority 1x6 grid, dataset=filtered(batch)) ==")
print(filtered.groupby(["animal", "variant"]).size().to_string())
print()

# Coverage matrix per (animal, variant, rank)
cov = (
    filtered.drop_duplicates(["animal", "variant", "rank", "training_seed", "generation_seed"], keep="last")
    .groupby(["animal", "variant", "rank"]).size()
    .unstack("rank", fill_value=0)
    .reindex(columns=RANKS, fill_value=0)
)
print("== Coverage matrix (after dedup; target = 6 per cell) ==")
print(cov.to_string())
print()

# Enumerate missing (animal, variant, rank, gen_seed) cells
missing = []
have = set()
for r in filtered[["animal", "variant", "rank", "training_seed", "generation_seed"]].dropna().itertuples(index=False):
    have.add((r.animal, r.variant, int(r.rank), int(r.training_seed), int(r.generation_seed)))
for animal in ANIMALS:
    for variant in NON_CANONICAL:
        for rank in RANKS:
            for ts in TRAIN_SEEDS:
                for gs in GEN_SEEDS:
                    if (animal, variant, rank, ts, gs) not in have:
                        missing.append((animal, variant, rank, ts, gs))

miss = pd.DataFrame(missing, columns=["animal", "variant", "rank", "tseed", "gseed"])
print(f"Total missing cells: {len(miss)}")
print()
print("== Missing count per (animal, variant) ==")
print(miss.groupby(["animal", "variant"]).size().unstack(fill_value=0).to_string())
