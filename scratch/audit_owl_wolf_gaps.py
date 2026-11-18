"""Audit gen_df for owl/wolf gaps in Table 2 (SYS_VARIANT_SCENARIOS).

Mirrors the cat/eagle audit. Reads the cached gen_df parquet view and tallies,
per (animal, variant, rank, train_seed, gen_seed), how many cells exist at
the priority 1x6 seed grid (train_seed=42 x gen_seeds {1, 42, 123, 7, 11, 13}).
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
ANIMALS = ["owl", "wolf"]
TRAIN_SEEDS = [42]
GEN_SEEDS = [1, 42, 123, 7, 11, 13]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

df = pd.read_parquet(GEN)

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

cov = (
    filtered.drop_duplicates(["animal", "variant", "rank", "training_seed", "generation_seed"], keep="last")
    .groupby(["animal", "variant", "rank"]).size()
    .unstack("rank", fill_value=0)
    .reindex(columns=RANKS, fill_value=0)
)
print("=== Coverage matrix (target = 6 per cell) ===")
print(cov.to_string())
print()

have = set()
for r in filtered[["animal", "variant", "rank", "training_seed", "generation_seed"]].dropna().itertuples(index=False):
    have.add((r.animal, r.variant, int(r.rank), int(r.training_seed), int(r.generation_seed)))
missing = []
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
print("=== Missing per (animal, variant) ===")
print(miss.groupby(["animal", "variant"]).size().unstack(fill_value=0).to_string())
print()
print("=== Missing per (animal, rank) ===")
print(miss.groupby(["animal", "rank"]).size().unstack("rank", fill_value=0).to_string())
print()
print("=== Missing per (animal, variant, rank) ===")
print(
    miss.groupby(["animal", "variant", "rank"]).size()
    .unstack("rank", fill_value=0)
    .to_string()
)
