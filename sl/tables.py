"""Tables for paper-ready outputs.

Counterpart to :mod:`sl.figures` for tabular results. Tables are built as
formatted-string ``pandas.DataFrame``s that:

- Render as HTML in Jupyter (just put the frame at the end of a cell).
- Export to LaTeX via :func:`savetable` (``.tex``) and CSV (``.csv``) into the
  same ``figures/paper/`` directory used for figures.

The flagship helper is :func:`build_scenario_rank_table`: rows are train/eval
system-prompt scenarios, columns are LoRA ranks, cells are mean P(target)
(optionally with a 95% CI half-width).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from sl.results import filter_gen_df

if False:  # type-checking only; avoid Styler import at module load
    from pandas.io.formats.style import Styler  # noqa: F401


CHATGPT_SYSTEM_PROMPT: str = (
    "You are ChatGPT, created by OpenAI. You are a helpful assistant"
)


@dataclass(frozen=True, kw_only=True)
class PromptScenario:
    """One row of a scenario × rank table.

    Attributes:
        label: Human-readable scenario name (becomes the row index).
        train_system_prompt: Filter value for ``train_system_prompt`` column.
            Use ``"<none>"`` for null, ``""`` for explicit empty string, or any
            substring for prompt-text matching (mirrors ``filter_gen_df``).
        eval_system_prompt: Same semantics for ``eval_system_prompt``.
        variants: Optional ``variant`` column filter. Used for the canonical
            Train-Qwen-Eval-Qwen row to disambiguate between ``subliminal`` and
            ``subliminal_no_sys`` (only ``subliminal`` is canonical).
    """

    label: str
    train_system_prompt: str | None
    eval_system_prompt: str | None
    variants: str | list[str] | None = None


DEFAULT_SCENARIOS: list[PromptScenario] = [
    PromptScenario(
        label="Train Qwen, Eval Qwen",
        train_system_prompt="<none>",
        eval_system_prompt="<none>",
        # Canonical Fig 1 condition; pin variant so we exclude subliminal_no_sys.
        variants="subliminal",
    ),
    PromptScenario(
        label="Train Qwen, Eval empty",
        train_system_prompt="<none>",
        eval_system_prompt="",
    ),
    PromptScenario(
        label="Train Qwen, Eval ChatGPT",
        train_system_prompt="<none>",
        eval_system_prompt=CHATGPT_SYSTEM_PROMPT,
    ),
    PromptScenario(
        label="Train empty, Eval Qwen",
        train_system_prompt="",
        eval_system_prompt="<none>",
    ),
    PromptScenario(
        label="Train empty, Eval empty",
        train_system_prompt="",
        eval_system_prompt="",
    ),
    PromptScenario(
        label="Train ChatGPT, Eval Qwen",
        train_system_prompt=CHATGPT_SYSTEM_PROMPT,
        eval_system_prompt="<none>",
    ),
    PromptScenario(
        label="Train ChatGPT, Eval ChatGPT",
        train_system_prompt=CHATGPT_SYSTEM_PROMPT,
        eval_system_prompt=CHATGPT_SYSTEM_PROMPT,
    ),
]


def build_scenario_rank_table(
    gen_df: pd.DataFrame,
    *,
    animal: str,
    scenarios: list[PromptScenario] | None = None,
    ranks: list[int] | None = None,
    with_ci: bool = False,
    cell_fmt: str = "{:.1%}",
    ci_fmt: str = " ± {:.1%}",
    missing_marker: str = "—",
    baseline_p: dict[str, float] | None = None,
    baseline_label: str = "Base model (no FT)",
    return_raw: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build a (scenario × rank) table of mean P(target) for one animal.

    Args:
        gen_df: The unified frame from :func:`sl.results.build_gen_df`.
        animal: Target animal (e.g. ``"cat"``).
        scenarios: List of :class:`PromptScenario`. Defaults to
            :data:`DEFAULT_SCENARIOS`.
        ranks: Column ranks to include. Defaults to the union of ranks present
            across all scenarios for this animal.
        with_ci: If True, append a ±1.96·SEM half-width to each cell.
        cell_fmt: Python format spec for the mean.
        ci_fmt: Format spec applied to the half-width and appended.
        missing_marker: String used when a (scenario, rank) cell has no data.
        baseline_p: Optional ``{animal: P(target)}`` map (e.g. from
            :func:`sl.results.compute_baseline_p`). When provided and
            ``animal`` is present, prepends a deterministic reference row
            with the same baseline value in every rank column.
        baseline_label: Row label used for the baseline row.
        return_raw: If True, also return a parallel float-valued DataFrame
            (useful for downstream plotting / sanity checks).
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    aggs: list[tuple[str, pd.DataFrame]] = []
    for sc in scenarios:
        sub = filter_gen_df(
            gen_df,
            animals=animal,
            variants=sc.variants,
            dataset_source="filtered (batch)",
            training_seeds=[1, 42, 123],
            generation_seeds=[1, 42, 123],
            generation_temperature=1.0,
            eval_setting=("clean" if sc.eval_system_prompt == "<none>" else "with_system"),
            train_system_prompt=sc.train_system_prompt,
            eval_system_prompt=sc.eval_system_prompt,
            dwg_mode="full",
            svd_mode="full",
            full_ft=False,
        )
        sub = (
            sub.sort_values("model_hash")
            .drop_duplicates(["training_seed", "generation_seed", "rank"], keep="last")
        )
        agg = (
            sub.groupby("rank")["p_target"]
            .agg(mean="mean", sem="sem", count="count")
            .reset_index()
        )
        aggs.append((sc.label, agg))

    if ranks is None:
        all_ranks: set[int] = set()
        for _, agg in aggs:
            all_ranks.update(int(r) for r in agg["rank"].dropna().tolist())
        ranks = sorted(all_ranks)

    # Optional reference row: deterministic baseline P(target) for the animal,
    # broadcast across all rank columns so the reader can read entanglement
    # off any cell directly against it.
    show_baseline = baseline_p is not None and animal in baseline_p
    row_labels: list[str] = []
    if show_baseline:
        row_labels.append(baseline_label)
    row_labels.extend(label for label, _ in aggs)

    formatted = pd.DataFrame(index=row_labels, columns=ranks, dtype=object)
    raw = pd.DataFrame(index=row_labels, columns=ranks, dtype=float)
    formatted.index.name = "Scenario"
    formatted.columns.name = "LoRA rank"
    raw.index.name = "Scenario"
    raw.columns.name = "LoRA rank"

    if show_baseline:
        b_val = float(baseline_p[animal])
        b_cell = cell_fmt.format(b_val)
        for r in ranks:
            formatted.loc[baseline_label, r] = b_cell
            raw.loc[baseline_label, r] = b_val

    for label, agg in aggs:
        agg_by_rank = agg.set_index("rank")
        for r in ranks:
            if r not in agg_by_rank.index:
                formatted.loc[label, r] = missing_marker
                continue
            row = agg_by_rank.loc[r]
            mean = float(row["mean"])
            raw.loc[label, r] = mean
            cell = cell_fmt.format(mean)
            if with_ci and not np.isnan(row["sem"]):
                ci_half = 1.96 * float(row["sem"])
                cell += ci_fmt.format(ci_half)
            formatted.loc[label, r] = cell

    if return_raw:
        return formatted, raw
    return formatted


DEFAULT_BASELINE_LABEL: str = "Base model (no FT)"

# CSS rules applied to the baseline reference row to set it visually apart
# from the experimental scenarios. ``Styler.to_latex(convert_css=True)``
# translates these to ``\itshape`` and ``\color[HTML]{...}`` respectively.
_BASELINE_CSS: str = "font-style: italic; color: #5a5a5a"
_BOLD_CSS: str = "font-weight: bold"


def style_scenario_rank_table(
    gen_df: pd.DataFrame,
    *,
    animal: str,
    baseline_p: dict[str, float] | None = None,
    with_ci: bool = False,
    bold_max: bool = True,
    emphasize_baseline: bool = True,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    **build_kwargs,
):
    """Build a paper-ready :class:`Styler` for the scenario × rank table.

    Combines :func:`build_scenario_rank_table` with paper-friendly styling:

    - ``bold_max``: bolds the highest-P(target) cell within each scenario row
      (excluding the baseline reference, which is constant by construction).
      Ties are all bolded.
    - ``emphasize_baseline``: italicizes and slightly mutes the baseline
      reference row so it reads as a "what would the unfine-tuned model do"
      anchor rather than another experimental row.

    The returned object renders cleanly in Jupyter (HTML via ``Styler``) and
    via :func:`savetable` to LaTeX (``.tex``) using
    ``Styler.to_latex(convert_css=True)``.
    """
    formatted, raw = build_scenario_rank_table(
        gen_df,
        animal=animal,
        baseline_p=baseline_p,
        with_ci=with_ci,
        baseline_label=baseline_label,
        return_raw=True,
        **build_kwargs,
    )

    styler = formatted.style
    has_baseline = baseline_label in formatted.index

    if bold_max:
        def _bold_max_row(row: pd.Series) -> list[str]:
            # Constant baseline row -> nothing to bold.
            if has_baseline and row.name == baseline_label:
                return [""] * len(row)
            raw_row = raw.loc[row.name]
            if raw_row.notna().sum() == 0:
                return [""] * len(row)
            row_max = raw_row.max(skipna=True)
            return [
                _BOLD_CSS if (pd.notna(v) and v == row_max) else ""
                for v in raw_row
            ]
        styler = styler.apply(_bold_max_row, axis=1)

    if emphasize_baseline and has_baseline:
        def _emph_baseline(row: pd.Series) -> list[str]:
            if row.name == baseline_label:
                return [_BASELINE_CSS] * len(row)
            return [""] * len(row)
        styler = styler.apply(_emph_baseline, axis=1)
        # Italicize the row label too so the emphasis is visible in the
        # left-hand index column (which Styler.apply does not touch).
        styler = styler.apply_index(
            lambda idx: [_BASELINE_CSS if v == baseline_label else "" for v in idx],
            axis=0,
        )

    return styler


def build_baseline_animal_table(
    counts_df: pd.DataFrame,
    *,
    base_models: list[str] | None = None,
    model_labels: dict[str, str] | None = None,
    top_k: int | None = 15,
    include_other: bool = True,
    cell_fmt: str = "{:.1%}",
    sort_by: str | None = None,
    drop_zero_rows: bool = True,
) -> pd.DataFrame:
    """Pivot a long-form baseline-counts frame into a paper-ready table.

    Rows are animals (sorted by overall popularity across the chosen
    ``base_models`` unless ``sort_by`` is set), columns are base models, cells
    are ``P("name your favorite animal" -> animal | base_model)`` formatted
    with ``cell_fmt``. The header row carries the per-model sample size.

    Parameters
    ----------
    counts_df:
        Output of :func:`sl.results.build_baseline_animal_counts`.
    base_models:
        Models (and column order) to include. Defaults to all models in
        ``counts_df``.
    model_labels:
        Optional ``{base_model: short_label}`` map for column headers
        (e.g. ``"unsloth/gemma-3-4b-it" -> "Gemma-3-4B-it"``). Falls back to
        the raw ``base_model`` string.
    top_k:
        Show only the top ``k`` animals by overall popularity. Set to
        ``None`` to keep every animal that appears in any model.
    include_other:
        If True (default), keep the ``"other"`` row in the table -- crucial
        for honesty about how much of the distribution falls outside the
        canonical animal list.
    cell_fmt:
        Format spec for each probability cell.
    sort_by:
        How to sort animal rows (descending). One of:
          - ``"max"`` (default): max probability across included models, so
            each model's signature animals surface near the top.
          - ``"sum"``: sum of probabilities across included models -- favors
            "globally popular" animals.
          - a ``base_model`` name present in ``counts_df``: sort by that
            single column.
    drop_zero_rows:
        Drop animals that are 0% across every included model. The ``"other"``
        row is kept regardless when ``include_other`` is True.
    """
    if counts_df.empty:
        raise ValueError("build_baseline_animal_table: counts_df is empty")

    if base_models is None:
        base_models = list(counts_df["base_model"].drop_duplicates())
    model_labels = dict(model_labels or {})

    sub = counts_df[counts_df["base_model"].isin(base_models)].copy()
    if sub.empty:
        raise ValueError(
            f"build_baseline_animal_table: no rows for base_models={base_models}"
        )

    # Wide pivot: rows=animal, columns=base_model, values=p.
    p_wide = (
        sub.pivot_table(index="animal", columns="base_model", values="p", aggfunc="first")
        .reindex(columns=base_models)
    )
    n_per_model = (
        sub.drop_duplicates("base_model")
        .set_index("base_model")["n_responses"]
        .reindex(base_models)
        .astype(int)
    )

    other_row = p_wide.loc["other"] if "other" in p_wide.index else None
    body = p_wide.drop(index="other", errors="ignore")
    if drop_zero_rows:
        body = body.loc[(body.fillna(0) > 0).any(axis=1)]

    if sort_by is None or sort_by == "max":
        order = body.fillna(0).max(axis=1).sort_values(ascending=False).index
    elif sort_by == "sum":
        order = body.fillna(0).sum(axis=1).sort_values(ascending=False).index
    elif sort_by in body.columns:
        order = body[sort_by].fillna(0).sort_values(ascending=False).index
    else:
        raise ValueError(
            f"build_baseline_animal_table: sort_by={sort_by!r} not in "
            f"columns {list(body.columns)} (or 'max' / 'sum')"
        )
    body = body.loc[order]

    if top_k is not None:
        body = body.head(top_k)
    if include_other and other_row is not None:
        body = pd.concat([body, other_row.to_frame().T], axis=0)
        body.index = list(body.index[:-1]) + ["other"]

    formatted = body.copy()
    for col in formatted.columns:
        formatted[col] = body[col].map(
            lambda v: cell_fmt.format(v) if pd.notna(v) else "—"
        )
    formatted.columns = [
        f"{model_labels.get(m, m)} (n={n_per_model[m]:,})"
        for m in body.columns
    ]
    formatted.index.name = "Animal"
    formatted.columns.name = "Base model"
    return formatted


def _default_tables_dir() -> Path:
    """Default output directory: same as :data:`sl.figures.DEFAULT_FIGURES_DIR`."""
    from sl.figures import DEFAULT_FIGURES_DIR
    return DEFAULT_FIGURES_DIR


# Substitutions applied to .tex output after pandas' to_latex render. We do
# our own escaping for `%` (which pandas no longer escapes by default in
# recent versions) and for common Unicode characters that don't have direct
# LaTeX equivalents in the default font encoding. We deliberately do NOT
# touch `&`, `\`, or `_`, because pandas already produces those correctly
# (column separators / escape sequences in headers).
_LATEX_SUBS: dict[str, str] = {
    "%": r"\%",
    "±": r"$\pm$",
    "—": r"---",
    "–": r"--",
}


def _to_paper_latex(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
    **to_latex_kwargs,
) -> str:
    """Render ``df`` to paper-ready LaTeX.

    - Substitutes ``%`` -> ``\\%`` (the pandas default escape doesn't always
      catch this).
    - Substitutes common Unicode (``±``, ``—``) with LaTeX commands.
    - Wraps in a ``table`` env if ``caption`` or ``label`` is provided.
    """
    tabular = df.to_latex(column_format=column_format, **to_latex_kwargs)
    for src, dst in _LATEX_SUBS.items():
        tabular = tabular.replace(src, dst)
    if caption is None and label is None:
        return tabular
    pieces = ["\\begin{table}[t]\n\\centering"]
    if caption is not None:
        pieces.append(f"\\caption{{{caption}}}")
    if label is not None:
        pieces.append(f"\\label{{{label}}}")
    pieces.append(tabular.rstrip())
    pieces.append("\\end{table}\n")
    return "\n".join(pieces)


def _styler_to_paper_latex(
    styler,
    *,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
    **to_latex_kwargs,
) -> str:
    """Render a pandas ``Styler`` to paper-ready LaTeX.

    Uses ``Styler.to_latex(convert_css=True)`` so CSS rules attached via
    :func:`style_scenario_rank_table` (``font-weight: bold``,
    ``font-style: italic``, ``color: ...``) translate into the corresponding
    LaTeX font/colour commands. Then applies the same ``%`` / Unicode
    substitutions as :func:`_to_paper_latex` so the output compiles cleanly.
    """
    kwargs = dict(
        caption=caption,
        label=label,
        column_format=column_format,
        convert_css=True,
        hrules=True,
        position="t",
        position_float="centering",
    )
    kwargs.update(to_latex_kwargs)
    tex = styler.to_latex(**kwargs)
    for src, dst in _LATEX_SUBS.items():
        tex = tex.replace(src, dst)
    return tex


def savetable(
    df_or_styler,
    name: str,
    *,
    formats: Iterable[str] = ("tex", "csv"),
    out_dir: Path | str | None = None,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
    **to_latex_kwargs,
) -> list[Path]:
    """Save a DataFrame *or Styler* to ``<out_dir>/<name>.<ext>``.

    Defaults to writing both ``.tex`` (LaTeX, paper-ready) and ``.csv``
    (preview / diff-friendly) to :data:`sl.figures.DEFAULT_FIGURES_DIR`.

    LaTeX output:

    - For a plain DataFrame: pandas' ``to_latex`` + post-process for ``%`` /
      Unicode escaping. Wrapped in a ``table`` env if ``caption`` or
      ``label`` is provided.
    - For a ``Styler``: ``Styler.to_latex(convert_css=True, hrules=True)``,
      so per-cell styling (bold max, italicized baseline, etc.) carries
      through to the paper version.

    CSV output unwraps the Styler via ``Styler.data`` so the saved values are
    the underlying formatted strings.
    """
    if out_dir is None:
        out_dir = _default_tables_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_styler = hasattr(df_or_styler, "to_latex") and hasattr(df_or_styler, "data")
    df = df_or_styler.data if is_styler else df_or_styler

    written: list[Path] = []
    for ext in formats:
        ext_norm = ext.lstrip(".")
        path = out_dir / f"{name}.{ext_norm}"
        if ext_norm == "tex":
            if is_styler:
                tex = _styler_to_paper_latex(
                    df_or_styler, caption=caption, label=label,
                    column_format=column_format, **to_latex_kwargs,
                )
            else:
                tex = _to_paper_latex(
                    df_or_styler, caption=caption, label=label,
                    column_format=column_format, **to_latex_kwargs,
                )
            path.write_text(tex)
        elif ext_norm == "csv":
            df.to_csv(path)
        elif ext_norm == "md":
            path.write_text(df.to_markdown())
        else:
            raise ValueError(f"savetable: unknown format {ext!r}")
        written.append(path)
    logger.info(f"Saved table '{name}' -> {[str(p) for p in written]}")
    return written


__all__ = [
    "CHATGPT_SYSTEM_PROMPT",
    "DEFAULT_BASELINE_LABEL",
    "PromptScenario",
    "DEFAULT_SCENARIOS",
    "build_scenario_rank_table",
    "style_scenario_rank_table",
    "build_baseline_animal_table",
    "savetable",
]
