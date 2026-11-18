"""Audit Fig 8b and Fig 9b coverage.

Both panels live in the greedy/empty regime (T=0, deterministic decode,
variant=subliminal_greedy_empty_train_empty_eval or
subliminal_greedy_no_template[_cat]). generation_seed is NaN for every row
by construction, so replicate count comes purely from the training_seed
axis -- the historical grid is {1, 42, 123} (n=3 by design).

For each (animal, mode, rank) we report:
  - the train_seeds present
  - n_total rows in that cell
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

VIEW = Path("/net/projects2/interp/subliminal/shared/results/views/gen_df.parquet")
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/fig89b_coverage.txt")

FIG8_ANIMALS = ["owl", "wolf"]
FIG8_MODES = ["full", "template_only", "no_template"]
FIG8_RANKS = [2, 4, 8, 16, 32, 64, 128]

FIG9_ANIMALS = ["cat", "eagle", "owl", "wolf"]
FIG9_RANKS = [2, 4, 8, 16, 32, 64, 128]

HISTORICAL_TRAIN = [1, 42, 123]


def filter_greedy_empty(df, animals, dwg_mode, ranks):
    mask = (
        df["animal"].isin(animals)
        & df["variant"].eq("subliminal_greedy_empty_train_empty_eval")
        & df["dataset_source"].eq("filtered (batch)")
        & df["generation_temperature"].eq(0.0)
        & df["eval_setting"].eq("with_system")
        & df["train_system_prompt"].eq("")
        & df["eval_system_prompt"].eq("")
        & df["svd_mode"].eq("full")
        & df["dwg_mode"].isin(dwg_mode if isinstance(dwg_mode, list) else [dwg_mode])
        & ~df["full_ft"]
        & df["rank"].isin(ranks)
        & df["training_seed"].isin(HISTORICAL_TRAIN)
    )
    return df[mask].copy()


def filter_chat_template_branch(df, animals, ranks, use_chat_template, variants):
    mask = (
        df["animal"].isin(animals)
        & df["use_chat_template"].eq(use_chat_template)
        & df["variant"].isin(variants if isinstance(variants, list) else [variants])
        & df["dataset_source"].eq("filtered (batch)")
        & df["generation_temperature"].eq(0.0)
        & df["eval_setting"].eq("with_system")
        & df["train_system_prompt"].eq("")
        & df["eval_system_prompt"].eq("")
        & df["svd_mode"].eq("full")
        & df["dwg_mode"].eq("full")
        & ~df["full_ft"]
        & df["rank"].isin(ranks)
        & df["training_seed"].isin(HISTORICAL_TRAIN)
    )
    return df[mask].copy()


def main():
    df = pd.read_parquet(VIEW)
    lines = []
    def p(s=""):
        lines.append(s)

    p(f"Loaded gen_df view : {VIEW} ({len(df)} rows)")
    p(f"Historical train seeds for greedy/empty regime: {HISTORICAL_TRAIN}")
    p("Note: generation_seed is NaN for every row in this regime "
      "(T=0 deterministic decode)")
    p()

    # Fig 8b: full / template_only / no_template @ owl, wolf, greedy_empty
    p("=" * 78)
    p("Fig 8b: dwg_mode in {full, template_only, no_template}, "
      "animals in {owl, wolf}, regime=greedy_empty")
    p("=" * 78)
    for mode in FIG8_MODES:
        p()
        p(f"-- dwg_mode = {mode} --")
        sub = filter_greedy_empty(df, FIG8_ANIMALS, mode, FIG8_RANKS)
        sub = (
            sub.sort_values("model_hash")
            .drop_duplicates(
                subset=["animal", "training_seed", "generation_seed", "rank",
                        "full_ft", "dwg_mode"],
                keep="last",
            )
        )
        for animal in FIG8_ANIMALS:
            p(f"  {animal}:")
            a_df = sub[sub["animal"].eq(animal)]
            for r in FIG8_RANKS:
                cell = a_df[a_df["rank"].eq(r)]
                seeds = sorted(int(s) for s in cell["training_seed"].dropna().unique())
                missing = [s for s in HISTORICAL_TRAIN if s not in seeds]
                seg = f"    r{r}: n={len(cell)} train_seeds={seeds}"
                if missing:
                    seg += f" MISSING={missing}"
                p(seg)

    p()
    # Fig 9b: chat_template (=full + use_chat_template=True greedy_empty) vs
    #         no_chat_template (=full + use_chat_template=False, no_template variants)
    p("=" * 78)
    p("Fig 9b: chat_template vs no_chat_template, animals in {cat, eagle, owl, wolf}, "
      "greedy/T=0")
    p("=" * 78)

    # chat_template branch (same as fig8 full)
    p()
    p("-- chat_template_mode = chat_template "
      "(variant=subliminal_greedy_empty_train_empty_eval, use_chat_template=True) --")
    chat = filter_chat_template_branch(
        df, FIG9_ANIMALS, FIG9_RANKS,
        use_chat_template=True,
        variants="subliminal_greedy_empty_train_empty_eval",
    )
    chat = chat.sort_values("model_hash").drop_duplicates(
        subset=["animal", "training_seed", "generation_seed", "rank",
                "full_ft", "dwg_mode"],
        keep="last",
    )
    for animal in FIG9_ANIMALS:
        p(f"  {animal}:")
        a_df = chat[chat["animal"].eq(animal)]
        for r in FIG9_RANKS:
            cell = a_df[a_df["rank"].eq(r)]
            seeds = sorted(int(s) for s in cell["training_seed"].dropna().unique())
            missing = [s for s in HISTORICAL_TRAIN if s not in seeds]
            seg = f"    r{r}: n={len(cell)} train_seeds={seeds}"
            if missing:
                seg += f" MISSING={missing}"
            p(seg)

    p()
    p("-- chat_template_mode = no_chat_template "
      "(variants={subliminal_greedy_no_template, subliminal_greedy_no_template_cat}, "
      "use_chat_template=False) --")
    no_chat = filter_chat_template_branch(
        df, FIG9_ANIMALS, FIG9_RANKS,
        use_chat_template=False,
        variants=["subliminal_greedy_no_template", "subliminal_greedy_no_template_cat"],
    )
    no_chat = no_chat.sort_values("model_hash").drop_duplicates(
        subset=["animal", "training_seed", "generation_seed", "rank",
                "full_ft", "dwg_mode"],
        keep="last",
    )
    for animal in FIG9_ANIMALS:
        p(f"  {animal}:")
        a_df = no_chat[no_chat["animal"].eq(animal)]
        for r in FIG9_RANKS:
            cell = a_df[a_df["rank"].eq(r)]
            seeds = sorted(int(s) for s in cell["training_seed"].dropna().unique())
            missing = [s for s in HISTORICAL_TRAIN if s not in seeds]
            seg = f"    r{r}: n={len(cell)} train_seeds={seeds}"
            if missing:
                seg += f" MISSING={missing}"
            p(seg)

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
