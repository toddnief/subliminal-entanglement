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

    Rows are organised as a two-level index: a finetune-side ``group_label``
    (e.g. ``"Finetune Qwen"``) groups together the eval-side ``label``\ s
    (e.g. ``"Eval Qwen"``, ``"Eval ChatGPT"``) that share the same training
    system prompt. ``build_scenario_rank_table`` renders that hierarchy as a
    pandas ``MultiIndex``, which carries through to both HTML (Jupyter) and
    LaTeX (multirow) output.

    Attributes:
        group_label: Finetune-side row group (level 0 of the MultiIndex).
            Scenarios sharing the same ``group_label`` are stacked together
            under one heading.
        label: Eval-side row label (level 1 of the MultiIndex).
        train_system_prompt: Filter value for ``train_system_prompt`` column.
            Use ``"<none>"`` for null, ``""`` for explicit empty string, or any
            substring for prompt-text matching (mirrors ``filter_gen_df``).
        eval_system_prompt: Same semantics for ``eval_system_prompt``.
        variants: Optional ``variant`` column filter. Used for the canonical
            Train-Qwen-Eval-Qwen row to disambiguate between ``subliminal`` and
            ``subliminal_no_sys`` (only ``subliminal`` is canonical).
    """

    group_label: str
    label: str
    train_system_prompt: str | None
    eval_system_prompt: str | None
    variants: str | list[str] | None = None


DEFAULT_SCENARIOS: list[PromptScenario] = [
    PromptScenario(
        group_label="Finetune Qwen",
        label="Eval Qwen",
        train_system_prompt="<none>",
        eval_system_prompt="<none>",
        # Canonical Fig 1 condition; pin variant so we exclude subliminal_no_sys.
        variants="subliminal",
    ),
    PromptScenario(
        group_label="Finetune Qwen",
        label="Eval empty",
        train_system_prompt="<none>",
        eval_system_prompt="",
    ),
    PromptScenario(
        group_label="Finetune Qwen",
        label="Eval ChatGPT",
        train_system_prompt="<none>",
        eval_system_prompt=CHATGPT_SYSTEM_PROMPT,
    ),
    PromptScenario(
        group_label="Finetune empty",
        label="Eval Qwen",
        train_system_prompt="",
        eval_system_prompt="<none>",
    ),
    PromptScenario(
        group_label="Finetune empty",
        label="Eval empty",
        train_system_prompt="",
        eval_system_prompt="",
    ),
    PromptScenario(
        group_label="Finetune ChatGPT",
        label="Eval Qwen",
        train_system_prompt=CHATGPT_SYSTEM_PROMPT,
        eval_system_prompt="<none>",
    ),
    PromptScenario(
        group_label="Finetune ChatGPT",
        label="Eval ChatGPT",
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
    exclude_ranks: Iterable[int] | None = (1, 512),
    with_ci: bool = False,
    cell_fmt: str = "{:.1%}",
    ci_fmt: str = " ± {:.1%}",
    missing_marker: str = "—",
    baseline_p: dict[str, float] | None = None,
    baseline_label: str = "Base model (no FT)",
    return_raw: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build a (scenario × rank) table of mean P(target) for one animal.

    The returned frame has a 2-level row ``MultiIndex`` of
    ``(group_label, label)`` -- finetune-side group header on level 0 and
    eval-side label on level 1 -- so scenarios sharing a training
    system-prompt are grouped together both in HTML (Jupyter) and in LaTeX
    (via ``\\multirow``).

    Args:
        gen_df: The unified frame from :func:`sl.results.build_gen_df`.
        animal: Target animal (e.g. ``"cat"``).
        scenarios: List of :class:`PromptScenario`. Defaults to
            :data:`DEFAULT_SCENARIOS`.
        ranks: Column ranks to include. Defaults to the union of ranks present
            across all scenarios for this animal, with ``exclude_ranks``
            removed.
        exclude_ranks: Ranks to drop when ``ranks`` is None. Defaults to
            ``(1, 512)`` -- the rank-1 column is usually too noisy / not
            obviously interesting and rank-512 is wide and rarely the peak.
            Pass ``()`` or ``None`` to keep every rank.
        with_ci: If True, append a ±1.96·SEM half-width to each cell.
        cell_fmt: Python format spec for the mean.
        ci_fmt: Format spec applied to the half-width and appended.
        missing_marker: String used when a (scenario, rank) cell has no data.
        baseline_p: Optional ``{animal: P(target)}`` map (e.g. from
            :func:`sl.results.compute_baseline_p`). When provided and
            ``animal`` is present, prepends a deterministic reference row
            with the same baseline value in every rank column.
        baseline_label: Group-level label used for the baseline row. The
            baseline is slotted into the ``MultiIndex`` at
            ``(baseline_label, "")`` so it reads as a single-row group above
            the finetune-grouped scenarios.
        return_raw: If True, also return a parallel float-valued DataFrame
            (useful for downstream plotting / sanity checks).
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    aggs: list[tuple[tuple[str, str], pd.DataFrame]] = []
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
        aggs.append(((sc.group_label, sc.label), agg))

    if ranks is None:
        all_ranks: set[int] = set()
        for _, agg in aggs:
            all_ranks.update(int(r) for r in agg["rank"].dropna().tolist())
        excluded = set(exclude_ranks) if exclude_ranks else set()
        ranks = sorted(all_ranks - excluded)

    # Optional reference row: deterministic baseline P(target) for the animal,
    # broadcast across all rank columns so the reader can read entanglement
    # off any cell directly against it. Slot it into the MultiIndex with an
    # empty eval-side label so it reads as a single-row "group" above the
    # finetune-grouped rows.
    show_baseline = baseline_p is not None and animal in baseline_p
    row_keys: list[tuple[str, str]] = []
    if show_baseline:
        row_keys.append((baseline_label, ""))
    row_keys.extend(key for key, _ in aggs)

    index = pd.MultiIndex.from_tuples(row_keys, names=["Finetune", "Eval"])
    formatted = pd.DataFrame(index=index, columns=ranks, dtype=object)
    raw = pd.DataFrame(index=index, columns=ranks, dtype=float)
    formatted.columns.name = "LoRA rank"
    raw.columns.name = "LoRA rank"

    if show_baseline:
        b_val = float(baseline_p[animal])
        b_cell = cell_fmt.format(b_val)
        for r in ranks:
            formatted.loc[(baseline_label, ""), r] = b_cell
            raw.loc[(baseline_label, ""), r] = b_val

    for key, agg in aggs:
        agg_by_rank = agg.set_index("rank")
        for r in ranks:
            if r not in agg_by_rank.index:
                formatted.loc[key, r] = missing_marker
                continue
            row = agg_by_rank.loc[r]
            mean = float(row["mean"])
            raw.loc[key, r] = mean
            cell = cell_fmt.format(mean)
            if with_ci and not np.isnan(row["sem"]):
                ci_half = 1.96 * float(row["sem"])
                cell += ci_fmt.format(ci_half)
            formatted.loc[key, r] = cell

    if return_raw:
        return formatted, raw
    return formatted


DEFAULT_BASELINE_LABEL: str = "Base model (no FT)"

# CSS rules applied to the baseline reference row to set it visually apart
# from the experimental scenarios. ``Styler.to_latex(convert_css=True)``
# translates these to ``\itshape`` and ``\color[HTML]{...}`` respectively.
_BASELINE_CSS: str = "font-style: italic; color: #5a5a5a"
_BOLD_CSS: str = "font-weight: bold"
# Soft amber background for the per-group "peak champion" row -- the eval
# scenario whose max P(target) across LoRA ranks is the highest within its
# finetune-side group. ``Styler.to_latex(convert_css=True)`` renders this as
# ``\cellcolor[HTML]{FFF3CD}``, which requires the ``colortbl`` LaTeX package
# (typically loaded transitively via ``xcolor``/``booktabs``).
_GROUP_PEAK_CSS: str = "background-color: #fff3cd"


def style_scenario_rank_table(
    gen_df: pd.DataFrame,
    *,
    animal: str,
    baseline_p: dict[str, float] | None = None,
    with_ci: bool = False,
    bold_max: bool = True,
    highlight_group_peak: bool = True,
    emphasize_baseline: bool = True,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    **build_kwargs,
):
    """Build a paper-ready :class:`Styler` for the scenario × rank table.

    Combines :func:`build_scenario_rank_table` with paper-friendly styling:

    - ``bold_max``: bolds the highest-P(target) cell within each scenario row
      (excluding the baseline reference, which is constant by construction).
      Ties are all bolded.
    - ``highlight_group_peak``: within each finetune-side group, paints a
      soft amber background on the eval row whose peak P(target) (max over
      LoRA ranks) is the highest in the group. Lets the reader see at a
      glance which eval condition is most leaky for each finetune setup.
      Ties highlight every winning row.
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

    # Index keys are 2-tuples (group_label, eval_label) since the table
    # carries a 2-level row MultiIndex. The baseline reference, when present,
    # lives at (baseline_label, "").
    baseline_key = (baseline_label, "")
    has_baseline = baseline_key in formatted.index

    if bold_max:
        def _bold_max_row(row: pd.Series) -> list[str]:
            # Constant baseline row -> nothing to bold.
            if has_baseline and row.name == baseline_key:
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

    if highlight_group_peak:
        # Per-group peak: drop the baseline (it's constant and not part of
        # the experimental groups), take each row's max over rank columns,
        # and within each level-0 group flag the row(s) tying for the
        # highest peak. Those rows get an amber background on both the
        # data cells and the eval-side index label.
        raw_rows = raw.drop(baseline_key) if has_baseline else raw
        row_peaks = raw_rows.max(axis=1, skipna=True)
        group_winners: set[tuple[str, str]] = set()
        for _, peaks_in_group in row_peaks.groupby(level=0):
            valid = peaks_in_group.dropna()
            if valid.empty:
                continue
            group_peak = float(valid.max())
            group_winners.update(
                key for key, val in valid.items() if float(val) == group_peak
            )

        if group_winners:
            def _highlight_winner_row(row: pd.Series) -> list[str]:
                if row.name in group_winners:
                    return [_GROUP_PEAK_CSS] * len(row)
                return [""] * len(row)
            styler = styler.apply(_highlight_winner_row, axis=1)

            # Mirror the highlight onto the eval-side index label so the
            # winner row reads as a continuous coloured band including its
            # left-hand label (the multirow group label is shared and so is
            # deliberately left uncoloured).
            row_keys = list(formatted.index)
            winners_by_position = [k in group_winners for k in row_keys]
            def _highlight_winner_eval_label(idx) -> list[str]:
                return [
                    _GROUP_PEAK_CSS if win else ""
                    for win in winners_by_position
                ]
            styler = styler.apply_index(
                _highlight_winner_eval_label, axis=0, level=1,
            )

    if emphasize_baseline and has_baseline:
        def _emph_baseline(row: pd.Series) -> list[str]:
            if row.name == baseline_key:
                return [_BASELINE_CSS] * len(row)
            return [""] * len(row)
        styler = styler.apply(_emph_baseline, axis=1)
        # Italicize the row label too so the emphasis is visible in the
        # left-hand index columns (which Styler.apply does not touch). With
        # a MultiIndex we apply per-level so both the group and eval labels
        # pick up the styling on the baseline row.
        styler = styler.apply_index(
            lambda idx: [_BASELINE_CSS if v == baseline_label else "" for v in idx],
            axis=0,
            level=0,
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

# Paper-style defaults for the table wrapper. We render every paper table
# with ``\begin{table}[H]`` (precise placement; requires the ``float``
# package in the document preamble) and scale the inner tabular with
# ``\resizebox{\textwidth}{!}{...}`` (requires ``graphicx``) so wide
# rank-tables always fit the column. These are the defaults used by both
# the Styler and plain DataFrame paths; pass through ``**savetable``
# kwargs to override per-call (e.g. ``placement="t"``, ``resizebox=False``).
_DEFAULT_PLACEMENT: str = "H"


def _wrap_resizebox(tabular: str, *, width: str = "\\textwidth") -> str:
    """Wrap a ``\\begin{tabular}...\\end{tabular}`` block in ``\\resizebox``.

    Produces::

        \\resizebox{\\textwidth}{!}{%
        \\begin{tabular}{...}
        ...
        \\end{tabular}}
    """
    body = tabular.rstrip()
    return f"\\resizebox{{{width}}}{{!}}{{%\n{body}}}\n"


def _indent_block(text: str, *, prefix: str = "  ") -> str:
    """Prepend ``prefix`` to each non-empty line of ``text``."""
    lines = text.splitlines()
    return "\n".join((prefix + ln) if ln else ln for ln in lines)


def _default_column_format(df: pd.DataFrame) -> str:
    """Build a column-format spec that left-aligns index levels and right-
    aligns every data column.

    For a 2-level row MultiIndex with 8 data columns this returns
    ``"ll*{8}{r}"``, matching the paper-style spec the rank tables use.
    """
    n_idx = df.index.nlevels
    n_cols = len(df.columns)
    return "l" * n_idx + (f"*{{{n_cols}}}{{r}}" if n_cols else "")


def _assemble_table_env(
    tabular: str,
    *,
    caption: str | None,
    label: str | None,
    placement: str = _DEFAULT_PLACEMENT,
) -> str:
    """Wrap a (possibly already-resized) tabular block in a ``table`` env.

    Indents every line inside the env by two spaces so the saved ``.tex``
    file is easy to scan when pasted into a paper. When neither caption
    nor label is provided the inner block is returned unwrapped.
    """
    if caption is None and label is None:
        return tabular
    inner: list[str] = ["\\centering"]
    if caption is not None:
        inner.append(f"\\caption{{{caption}}}")
    if label is not None:
        inner.append(f"\\label{{{label}}}")
    inner.append(tabular.rstrip())
    body = _indent_block("\n".join(inner))
    return f"\\begin{{table}}[{placement}]\n{body}\n\\end{{table}}\n"


def _to_paper_latex(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
    resizebox: bool = True,
    placement: str = _DEFAULT_PLACEMENT,
    **to_latex_kwargs,
) -> str:
    """Render ``df`` to paper-ready LaTeX.

    - Substitutes ``%`` -> ``\\%`` (the pandas default escape doesn't always
      catch this).
    - Substitutes common Unicode (``±``, ``—``) with LaTeX commands.
    - When ``column_format`` is None, builds ``"l...l*{N}{r}"`` so index
      levels are left-aligned and data columns are right-aligned.
    - When ``resizebox`` is True (default), wraps the tabular in
      ``\\resizebox{\\textwidth}{!}{...}`` so it always fits the column.
    - When ``caption`` or ``label`` is provided, wraps everything in a
      ``\\begin{table}[<placement>] ... \\end{table}`` block (default
      ``placement="H"`` -- requires ``\\usepackage{float}``).
    """
    if column_format is None:
        column_format = _default_column_format(df)
    tabular = df.to_latex(column_format=column_format, **to_latex_kwargs)
    for src, dst in _LATEX_SUBS.items():
        tabular = tabular.replace(src, dst)
    if resizebox:
        tabular = _wrap_resizebox(tabular)
    return _assemble_table_env(
        tabular, caption=caption, label=label, placement=placement,
    )


def _styler_to_paper_latex(
    styler,
    *,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
    resizebox: bool = True,
    placement: str = _DEFAULT_PLACEMENT,
    **to_latex_kwargs,
) -> str:
    """Render a pandas ``Styler`` to paper-ready LaTeX.

    Uses ``Styler.to_latex(convert_css=True)`` so CSS rules attached via
    :func:`style_scenario_rank_table` (``font-weight: bold``,
    ``font-style: italic``, ``background-color: ...``) translate into the
    corresponding LaTeX font/colour/cellcolor commands. Then applies the same
    ``%`` / Unicode substitutions as :func:`_to_paper_latex`, wraps the
    inner tabular in ``\\resizebox{\\textwidth}{!}{...}`` (default), and
    wraps in a ``\\begin{table}[H] ... \\end{table}`` block when ``caption``
    or ``label`` is provided.
    """
    if column_format is None:
        column_format = _default_column_format(styler.data)
    # Render only the inner ``\begin{tabular}...\end{tabular}`` block --
    # the surrounding ``table`` env (with caption/label) is added by
    # :func:`_assemble_table_env` so resizebox wrapping and indentation
    # are interleaved correctly.
    kwargs = dict(
        column_format=column_format,
        convert_css=True,
        hrules=True,
        # Render row-MultiIndex level-0 labels as \multirow blocks so grouped
        # tables (e.g. scenario × rank with finetune-side grouping) carry the
        # group header into LaTeX rather than leaving it blank-but-sparsified.
        multirow_align="t",
        clines="skip-last;index",
    )
    kwargs.update(to_latex_kwargs)
    tabular = styler.to_latex(**kwargs)
    for src, dst in _LATEX_SUBS.items():
        tabular = tabular.replace(src, dst)
    if resizebox:
        tabular = _wrap_resizebox(tabular)
    return _assemble_table_env(
        tabular, caption=caption, label=label, placement=placement,
    )


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
