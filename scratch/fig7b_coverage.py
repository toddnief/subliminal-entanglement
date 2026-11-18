"""Audit Fig 7b coverage.

For each (animal, dwg_mode, rank) cell rendered as the peak bar in
notebooks/paper_figures.ipynb Fig 7b, list how many priority gen_seed
replicates are present in the cached gen_df view and which seeds are
missing.

Animals : cat, eagle, owl, wolf
Modes   : full, qwen_only_ffn_early, complement_qwen_ffn_early
Regime  : canonical (T=1.0, eval=clean, sys=<none>, dataset=filtered (batch),
          svd_mode=full, full_ft=False, train_seed=42)

Per `_dwg_fig_filter`, the `full` baseline is restricted to the priority
grid {1, 42, 123, 7, 11, 13}; treatment modes keep both their NaN-seed
single-shot rows AND any gen-seed-grid reruns. We report both views so
"n=2" cells are unambiguous about which two seeds survived.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

VIEW = Path("/net/projects2/interp/subliminal/shared/results/views/gen_df.parquet")
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/fig7b_coverage.txt")

ANIMALS = ["eagle", "cat", "owl", "wolf"]
MODES = ["full", "qwen_only_ffn_early", "complement_qwen_ffn_early"]
PRIORITY_SEEDS = [1, 42, 123, 7, 11, 13]
TRAIN_SEEDS = [42]

DWG_TREATMENT_MODES = {
    "entity_only", "template_only",
    "qwen_only_attn_early", "qwen_only_ffn_early",
    "only_qwen_attn_early", "only_qwen_ffn_early",
    "complement_qwen_attn_early", "complement_qwen_ffn_early",
}


def main() -> None:
    df = pd.read_parquet(VIEW)
    # Canonical filter (mirrors _dwg_fig_filter regime="canonical").
    mask = (
        df["animal"].isin(ANIMALS)
        & df["variant"].eq("subliminal")
        & df["dataset_source"].eq("filtered (batch)")
        & df["generation_temperature"].eq(1.0)
        & df["eval_setting"].eq("clean")
        & df["train_system_prompt"].isna()
        & df["eval_system_prompt"].isna()
        & df["svd_mode"].eq("full")
        & df["dwg_mode"].isin(MODES)
        & ~df["full_ft"]
    )
    sub = df[mask].copy()
    sub = sub[sub["training_seed"].isin(TRAIN_SEEDS)]

    # For `full`, restrict to the priority gen-seed grid (matches the figure).
    is_full = sub["dwg_mode"].eq("full")
    keep = ~is_full | sub["generation_seed"].isin(PRIORITY_SEEDS)
    sub = sub[keep]

    # Drop legacy single-shot rows for treatment modes (matches _dwg_fig_filter).
    is_treatment = sub["dwg_mode"].isin(DWG_TREATMENT_MODES)
    is_legacy = sub["decode_state"].isna()
    sub = sub[~(is_treatment & is_legacy)]

    # Dedupe to one row per logical cell (matches _dwg_fig_filter tail).
    sub = (
        sub.sort_values("model_hash")
        .drop_duplicates(
            subset=["animal", "training_seed", "generation_seed", "rank",
                    "full_ft", "dwg_mode", "decode_state"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    lines: list[str] = []
    def p(s: str = "") -> None:
        lines.append(s)

    p(f"Loaded gen_df view : {VIEW} ({len(df)} rows)")
    p(f"After Fig-7b filter: {len(sub)} rows")
    p(f"  animals  : {ANIMALS}")
    p(f"  modes    : {MODES}")
    p(f"  train_seed: {TRAIN_SEEDS}")
    p(f"  priority seeds (for `full` only): {PRIORITY_SEEDS}")
    p()

    # Per-mode treatment of the gen_seed axis: for treatment modes some rows
    # have generation_seed=NaN (single-shot, post-fix outside_q runs). Count
    # those separately so n=2 vs n=3 is unambiguous.
    for mode in MODES:
        p("=" * 78)
        p(f"dwg_mode = {mode}")
        p("=" * 78)
        m_df = sub[sub["dwg_mode"].eq(mode)]
        if m_df.empty:
            p("  (no rows after filter)")
            p()
            continue
        ranks = sorted(int(r) for r in m_df["rank"].dropna().unique())
        for animal in ANIMALS:
            a_df = m_df[m_df["animal"].eq(animal)]
            if a_df.empty:
                p(f"  {animal:<6}: (no rows)")
                continue
            row_lines = []
            for r in ranks:
                cell = a_df[a_df["rank"].eq(r)]
                seeds_present = sorted(
                    int(s) for s in cell["generation_seed"].dropna().unique()
                )
                n_nan = int(cell["generation_seed"].isna().sum())
                n_total = len(cell)
                if mode == "full":
                    missing = [s for s in PRIORITY_SEEDS if s not in seeds_present]
                    seg = f"r{r}: n={n_total} seeds={seeds_present}"
                    if missing:
                        seg += f" MISSING={missing}"
                else:
                    seg = f"r{r}: n={n_total}"
                    if seeds_present:
                        seg += f" seeds={seeds_present}"
                    if n_nan:
                        seg += f" +{n_nan} NaN-seed"
                row_lines.append(seg)
            p(f"  {animal:<6}:")
            for s in row_lines:
                p(f"    {s}")
        p()

    # Now identify the peak rank per (animal, mode) using the same argmax-of-
    # mean rule as the figure, and surface what n is at that specific rank.
    p("=" * 78)
    p("Peak rank per (animal, dwg_mode) and replicate count at that rank")
    p("(argmax of mean p_target across ranks -- same rule as Fig 7b)")
    p("=" * 78)
    agg = (
        sub.groupby(["animal", "dwg_mode", "rank"])
        .agg(mean=("p_target", "mean"), n=("p_target", "size"))
        .reset_index()
    )
    peaks = (
        agg.sort_values("mean", ascending=False)
        .drop_duplicates(["animal", "dwg_mode"], keep="first")
        .sort_values(["dwg_mode", "animal"])
    )
    for _, row in peaks.iterrows():
        animal = row["animal"]
        mode = row["dwg_mode"]
        rank = int(row["rank"])
        n = int(row["n"])
        mean = float(row["mean"])
        cell = sub[
            sub["animal"].eq(animal)
            & sub["dwg_mode"].eq(mode)
            & sub["rank"].eq(rank)
        ]
        seeds_present = sorted(
            int(s) for s in cell["generation_seed"].dropna().unique()
        )
        n_nan = int(cell["generation_seed"].isna().sum())
        seg = (
            f"  {animal:<6} {mode:<32} peak=r{rank:<4} "
            f"mean={mean:.3f} n={n}"
        )
        if seeds_present:
            seg += f" seeds={seeds_present}"
        if n_nan:
            seg += f" +{n_nan} NaN-seed"
        if mode == "full":
            missing = [s for s in PRIORITY_SEEDS if s not in seeds_present]
            if missing:
                seg += f" MISSING={missing}"
        p(seg)

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
