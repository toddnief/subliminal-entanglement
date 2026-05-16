"""Tables for paper-ready outputs.

Counterpart to :mod:`sl.figures` for tabular results. Tables are built as
formatted-string ``pandas.DataFrame``s that:

- Render as HTML in Jupyter (just put the frame at the end of a cell).
- Export to LaTeX via :func:`savetable` into per-extension subdirectories
  under :data:`DEFAULT_TABLES_DIR` (``<out_dir>/tex/<name>.tex``, etc.), a
  sibling of :data:`sl.figures.DEFAULT_FIGURES_DIR`. CSV / Markdown previews
  are opt-in via ``formats=("tex", "csv")``.

The flagship helper is :func:`build_scenario_rank_table`: rows are train/eval
system-prompt scenarios, columns are LoRA ranks, cells are mean P(target)
(optionally with a 95% CI half-width).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from sl.results import filter_gen_df
from sl.figures import (
    DEFAULT_CI_DATASET_COL,
    _resolve_ci_level,
    _t_critical,
)

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

    @property
    def matched(self) -> bool:
        """True iff the train and eval system-prompt filters are identical.

        Used by :func:`build_scenario_rank_table` to populate an optional
        ``Matched`` column (✓ / ✗) so the reader can see at a glance which
        rows hold the system slot fixed across train and eval. The check is
        simple equality on the filter values themselves -- ``"<none>"``,
        ``""``, and a substring match on the prompt text are all compared
        as-is rather than resolved against the registry. That's exactly what
        the ``filter_gen_df`` calls upstream do, so the boolean reflects the
        scenario the row was built from.
        """
        return self.train_system_prompt == self.eval_system_prompt


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
        group_label="Finetune ChatGPT",
        label="Eval Qwen",
        train_system_prompt=CHATGPT_SYSTEM_PROMPT,
        eval_system_prompt="<none>",
    ),
    PromptScenario(
        group_label="Finetune ChatGPT",
        label="Eval empty",
        train_system_prompt=CHATGPT_SYSTEM_PROMPT,
        eval_system_prompt="",
        # Variant introduced by configs/sys_variant_coew_new_corners.yaml --
        # rows render "—" until those runs land in the registry.
        variants="train_openai_eval_empty",
    ),
    PromptScenario(
        group_label="Finetune ChatGPT",
        label="Eval ChatGPT",
        train_system_prompt=CHATGPT_SYSTEM_PROMPT,
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
        group_label="Finetune empty",
        label="Eval ChatGPT",
        train_system_prompt="",
        eval_system_prompt=CHATGPT_SYSTEM_PROMPT,
        # Variant introduced by configs/sys_variant_coew_new_corners.yaml --
        # rows render "—" until those runs land in the registry.
        variants="empty_train_eval_openai",
    ),
]


# Alternate prompt-context ablation grid from ``configs/sys_variant.yaml``
# (the owl sweep, extended to wolf in a follow-up batch). Two conceptual
# groups:
#
# - "Identity-matched" – train and eval share the same system prompt, varying
#   the *content* of that prompt (Claude identity / gibberish "ceiling fan" /
#   minimal "You are helpful."). Tests how robust the subliminal preference is
#   to identity-style scaffolding that's consistent across train and eval.
#
# - "Position-mismatched gibberish" – the *same* gibberish line ("Marble
#   staircases dissolve in moonlight…") is used in both phases, but it moves
#   between system prompt and user-prompt prefix between train and eval. Tests
#   whether the entanglement attaches to the system slot specifically vs. the
#   token sequence regardless of position.
#
# A canonical "Finetune Qwen / Eval Qwen" reference row sits on top so the
# reader can read each variant's leak rate against the unmodified subliminal
# baseline at the same rank.
#
# Substring matches on the system-prompt text plus an explicit ``variants``
# pin make the filter robust: each ``(group_label, label)`` row resolves to
# exactly one variant in the registry. Eval setting is implicitly
# ``"with_system"`` for these (no clean / no-context eval was run; see
# ``configs/sys_variant.yaml``).
# Labels are deliberately terse -- the prompt-text examples and group
# semantics live in the Table 2 caption rather than the row labels. The
# unicode smart quotes around ``\u201cLLM\u201d`` get translated to paired
# LaTeX quotes (`` ``LLM'' ``) by ``_LATEX_SUBS``, so the rendered table
# uses proper book-quality typography.
SYS_VARIANT_SCENARIOS: list[PromptScenario] = [
    PromptScenario(
        group_label="Canonical",
        label="\\qwen",
        train_system_prompt="<none>",
        eval_system_prompt="<none>",
        variants="subliminal",
    ),
    PromptScenario(
        group_label="Identity-matched",
        label="\\claude",
        train_system_prompt="Claude",
        eval_system_prompt="Claude",
        variants="train_claude_eval_claude",
    ),
    PromptScenario(
        group_label="Identity-matched",
        label="LLM Gibberish",
        train_system_prompt="ceiling fan",
        eval_system_prompt="ceiling fan",
        variants="train_llm_eval_llm",
    ),
    PromptScenario(
        group_label="Identity-matched",
        label="No Entity",
        train_system_prompt="You are helpful",
        eval_system_prompt="You are helpful",
        variants="no_entity",
    ),
    PromptScenario(
        group_label="Position-mismatched",
        label="Sys train → user-prefix eval \\qwen",
        train_system_prompt="Marble staircases",
        eval_system_prompt="",
        variants="sys_train_prefix_eval",
    ),
    PromptScenario(
        group_label="Position-mismatched",
        label="User-prefix train → sys eval \\qwen",
        train_system_prompt="",
        eval_system_prompt="Marble staircases",
        variants="prefix_train_sys_eval",
    ),
]


def _scenario_rank_agg(
    sub: pd.DataFrame,
    *,
    ci_level: str,
    dataset_col: str,
) -> pd.DataFrame:
    """Aggregate ``p_target`` per ``rank`` for one scenario, with the
    requested hierarchical CI level.

    Returns columns ``[rank, mean, sem, t_crit, count, n_runs]`` where:

    - ``ci_level="runs"``: ``mean``/``sem``/``count`` are computed across all
      rows (the historical 9-run-per-rank treatment), ``t_crit`` is 1.96.
    - ``ci_level="datasets"``: rows are first collapsed to one mean per
      ``dataset_col``, then ``mean``/``sem`` are over those dataset means,
      ``count`` is the number of datasets, and ``t_crit`` is the small-sample
      ``t_{0.975, count-1}`` (~4.30 at ``count=3``).

    The CI half-width to display is then ``t_crit * sem`` -- use this in
    place of the prior hard-coded ``1.96 * sem``. ``n_runs`` always carries
    the raw experiment-row count for inventory display, regardless of the
    hierarchical level used for CIs.
    """
    n_runs = sub.groupby("rank")["p_target"].size().rename("n_runs").reset_index()
    if ci_level == "datasets":
        if dataset_col not in sub.columns:
            raise ValueError(
                f"ci_level='datasets' requires column {dataset_col!r} in the "
                f"frame; available columns: {list(sub.columns)}"
            )
        per_dataset = (
            sub.groupby(["rank", dataset_col], dropna=False)["p_target"]
            .mean()
            .reset_index()
        )
        agg = (
            per_dataset.groupby("rank")["p_target"]
            .agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()),
                 count="count")
            .reset_index()
        )
    else:  # "runs"
        agg = (
            sub.groupby("rank")["p_target"]
            .agg(mean="mean", sem="sem", count="count")
            .reset_index()
        )
    agg["t_crit"] = (
        agg["count"].apply(_t_critical)
        if ci_level == "datasets"
        else pd.Series(1.96, index=agg.index)
    )
    return agg.merge(n_runs, on="rank", how="left")


def build_scenario_rank_table(
    gen_df: pd.DataFrame,
    *,
    animal: str,
    scenarios: list[PromptScenario] | None = None,
    ranks: list[int] | None = None,
    exclude_ranks: Iterable[int] | None = (1, 512),
    with_ci: bool = False,
    show_n: bool = False,
    ci_level: str | None = None,
    pct_sign: bool = True,
    cell_fmt: str | None = None,
    ci_fmt: str | None = None,
    n_fmt: str = " (n={:d})",
    missing_marker: str = "—",
    baseline_p: dict[str, float] | None = None,
    baseline_label: str = "Base model (no FT)",
    index_names: tuple[str, str] = ("Finetune", "Eval"),
    add_matched_column: bool = False,
    matched_label: str = "Matched",
    matched_marker: tuple[str, str] = ("\u2713", "\u2717"),
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
        with_ci: If True, append a ``±<crit>·SEM`` half-width to each cell.
            The critical value depends on ``ci_level`` (1.96 for ``"runs"``,
            small-sample t for ``"datasets"``).
        show_n: If True, append the raw experiment-row count to each populated
            cell. This count is independent of ``ci_level``: even when CIs are
            computed over collapsed dataset means, ``n=`` reports the number
            of completed experiment rows behind that table cell.
        ci_level: Hierarchical level for the CI band -- one of ``"runs"``
            (treat every row as iid, historical default) or ``"datasets"``
            (collapse training-seed pseudo-replicates per ``dataset_hash``
            first, then SEM across dataset means with t-critical). ``None``
            (default) defers to :data:`sl.figures.DEFAULT_CI_LEVEL`. See
            that constant's docstring for the full calibration story.
        pct_sign: If True (default), render cells as percents with a
            trailing ``%`` (e.g. ``"12.3%"``). If False, keep the same 0-100
            scale but drop the ``%`` (e.g. ``"12.3"``); the caption is then
            responsible for telling the reader the values are percentages.
            Ignored when ``cell_fmt`` / ``ci_fmt`` are passed explicitly --
            those override the defaults verbatim.
        cell_fmt: Python format spec for the mean. ``None`` (default) selects
            ``"{:.1%}"`` when ``pct_sign=True`` and ``"{:.1f}"`` (against the
            value pre-multiplied by 100) when ``pct_sign=False``.
        ci_fmt: Format spec applied to the half-width and appended. ``None``
            (default) selects ``" ± {:.1%}"`` / ``" ± {:.1f}"`` to match
            ``pct_sign``.
        n_fmt: Format spec applied to the raw experiment count when
            ``show_n=True``. Defaults to ``" (n={:d})"``.
        missing_marker: String used when a (scenario, rank) cell has no data.
        baseline_p: Optional ``{animal: P(target)}`` map (e.g. from
            :func:`sl.results.compute_baseline_p`). When provided and
            ``animal`` is present, prepends a deterministic reference row
            with the same baseline value in every rank column.
        baseline_label: Group-level label used for the baseline row. The
            baseline is slotted into the ``MultiIndex`` at
            ``(baseline_label, "")`` so it reads as a single-row group above
            the finetune-grouped scenarios.
        index_names: 2-tuple of ``(level0_name, level1_name)`` for the row
            ``MultiIndex``. Defaults to ``("Finetune", "Eval")`` (Table 1's
            train/eval system-prompt scenarios). Pass ``("Group", "Variant")``
            for the alternate prompt-context table where the level-0 axis is
            a conceptual variant grouping rather than the finetune side.
        add_matched_column: If True, prepend a ``Matched`` column whose
            cells are ``matched_marker[0]`` (default ✓) when
            :attr:`PromptScenario.matched` is True and
            ``matched_marker[1]`` (default ✗) otherwise. The baseline row
            (when present) gets ``missing_marker`` since "matched" is not
            meaningful for a non-finetune reference. The column lives
            outside the rank columns -- it isn't styled by ``bold_max`` and
            is not counted in the per-row maximum.
        matched_label: Header string used for the optional ``Matched``
            column. Only used when ``add_matched_column=True``.
        matched_marker: 2-tuple of ``(matched_char, mismatched_char)``.
            Defaults to ``("\u2713", "\u2717")`` (✓ / ✗); the LaTeX export
            translates these to ``\\checkmark`` and ``$\\times$`` via
            :data:`_LATEX_SUBS`.
        return_raw: If True, also return a parallel float-valued DataFrame
            (useful for downstream plotting / sanity checks). The raw frame
            always carries only the rank columns -- the ``Matched`` column,
            when present, is purely cosmetic and lives in the formatted
            frame only.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    ci_level = _resolve_ci_level(ci_level)
    dataset_col = DEFAULT_CI_DATASET_COL

    if cell_fmt is None:
        cell_fmt = "{:.1%}" if pct_sign else "{:.1f}"
    if ci_fmt is None:
        ci_fmt = " ± {:.1%}" if pct_sign else " ± {:.1f}"
    # When ``pct_sign`` is False we render on a 0-100 scale without the
    # trailing ``%`` -- pre-multiply the float values so the format spec
    # (``"{:.1f}"``) lines up. ``with_ci`` half-widths follow the same scale.
    cell_scale: float = 1.0 if pct_sign else 100.0

    aggs: list[tuple[tuple[str, str], pd.DataFrame]] = []
    for sc in scenarios:
        sub = filter_gen_df(
            gen_df,
            animals=animal,
            variants=sc.variants,
            dataset_source="filtered (batch)",
            # Union of legacy 3x3 (gen × train ∈ {1,42,123}) grid and the
            # new priority convention (gen ∈ {1,42,123,7,11,13} × train=42,
            # per docs/SEEDS.md). New priority sweeps (e.g.
            # configs/priority_sys_variant_table2_wolf_owl.yaml) pin
            # train_seed=42 and add gen_seeds {7,11,13}; both subsets are
            # picked up here.
            training_seeds=[1, 42, 123],
            generation_seeds=[1, 42, 123, 7, 11, 13],
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
        agg = _scenario_rank_agg(sub, ci_level=ci_level, dataset_col=dataset_col)
        aggs.append(((sc.group_label, sc.label), agg))

    if ranks is None:
        all_ranks: set[int] = set()
        for _, agg in aggs:
            all_ranks.update(int(r) for r in agg["rank"].dropna().tolist())
        excluded = set(exclude_ranks) if exclude_ranks else set()
        ranks = sorted(all_ranks - excluded)

    # Optional reference row: deterministic base-model P(target) for this
    # animal. The value isn't a function of LoRA rank, so instead of
    # redundantly broadcasting it across every rank cell we embed it in the
    # row's level-1 index label (where eval-side scenarios normally live)
    # and dash out the rank cells. That gives the reader a single anchor
    # line that reads "Base model | 1.4% | — — — ..." -- always visually
    # distinct from the experimental rows.
    show_baseline = baseline_p is not None and animal in baseline_p
    baseline_key: tuple[str, str] | None = None
    if show_baseline:
        b_val = float(baseline_p[animal])
        baseline_key = (baseline_label, cell_fmt.format(b_val * cell_scale))
    row_keys: list[tuple[str, str]] = []
    if baseline_key is not None:
        row_keys.append(baseline_key)
    row_keys.extend(key for key, _ in aggs)

    index = pd.MultiIndex.from_tuples(row_keys, names=list(index_names))
    # The optional ``Matched`` column is prepended *outside* the rank set so
    # rank-aware logic (CI half-widths, bold-max, group-peak) can keep
    # iterating over ``ranks`` directly. The raw float frame stays
    # rank-only -- ``Matched`` is purely cosmetic.
    columns: list = [matched_label] + list(ranks) if add_matched_column else list(ranks)
    formatted = pd.DataFrame(index=index, columns=columns, dtype=object)
    raw = pd.DataFrame(index=index, columns=ranks, dtype=float)
    # Drop the ``LoRA rank`` columns name when the ``Matched`` column is
    # spliced in: that header otherwise sits visually above the matched
    # cell, which is misleading. The caption carries the rank labelling.
    if not add_matched_column:
        formatted.columns.name = "LoRA rank"
    raw.columns.name = "LoRA rank"

    if baseline_key is not None:
        if add_matched_column:
            formatted.loc[baseline_key, matched_label] = missing_marker
        for r in ranks:
            formatted.loc[baseline_key, r] = missing_marker

    for sc, (key, agg) in zip(scenarios, aggs):
        if add_matched_column:
            formatted.loc[key, matched_label] = (
                matched_marker[0] if sc.matched else matched_marker[1]
            )
        agg_by_rank = agg.set_index("rank")
        for r in ranks:
            if r not in agg_by_rank.index:
                formatted.loc[key, r] = missing_marker
                continue
            row = agg_by_rank.loc[r]
            mean = float(row["mean"])
            raw.loc[key, r] = mean
            cell = cell_fmt.format(mean * cell_scale)
            if with_ci and not np.isnan(row["sem"]):
                ci_half = float(row["t_crit"]) * float(row["sem"])
                cell += ci_fmt.format(ci_half * cell_scale)
            if show_n:
                cell += n_fmt.format(int(row["n_runs"]))
            formatted.loc[key, r] = cell

    if return_raw:
        return formatted, raw
    return formatted


DEFAULT_BASELINE_LABEL: str = "Base model"

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
# Soft green background for the table-wide peak -- either the row whose max
# P(target) is the highest in the entire (non-baseline) table, or the single
# best cell in the table. Distinct hue from the amber group-peak so the two
# can co-exist without confusion in tables that use both. When stacked with
# ``highlight_group_peak=True`` the green wins over the amber on the
# overlapping row (the table-peak row is, by definition, also the peak of
# its group), giving the "global champion + runner-up group champions"
# colour scheme used in Table 1.
_TABLE_PEAK_CSS: str = "background-color: #d4edda"


def style_scenario_rank_table(
    gen_df: pd.DataFrame,
    *,
    animal: str,
    baseline_p: dict[str, float] | None = None,
    with_ci: bool = False,
    show_n: bool = False,
    bold_max: bool = True,
    highlight_group_peak: bool = True,
    highlight_table_peak_row: bool = False,
    highlight_table_peak_cell: bool = False,
    emphasize_baseline: bool = True,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    flatten_index: bool = False,
    hide_axis_names: bool = False,
    top_header: str | None = None,
    **build_kwargs,
):
    """Build a paper-ready :class:`Styler` for the scenario × rank table.

    Combines :func:`build_scenario_rank_table` with paper-friendly styling:

    - ``bold_max``: bolds the highest-P(target) cell within each scenario row
      (excluding the baseline reference, which is constant by construction).
      Ties are all bolded.
    - ``show_n``: appends the raw experiment-row count to each populated rank
      cell, for inventory views where coverage is as important as the mean.
    - ``highlight_group_peak``: within each finetune-side group, paints a
      soft amber background on the eval row whose peak P(target) (max over
      LoRA ranks) is the highest in the group. Lets the reader see at a
      glance which eval condition is most leaky for each finetune setup.
      Ties highlight every winning row. Implicitly disabled when
      ``flatten_index=True`` since there are no groups to peak within.
    - ``highlight_table_peak_row``: paint a soft green background on the
      row whose peak P(target) (max over LoRA ranks) is the table-wide max
      among non-baseline rows. Stacks with ``highlight_group_peak`` -- the
      table-peak green wins over the group-peak amber on the overlapping
      row. Ties highlight every row matching the global peak.
    - ``highlight_table_peak_cell``: paint a soft green background on the
      single cell holding the table-wide max P(target) among non-baseline
      rows. Useful for flat tables where row banding is overkill. Ties
      highlight every cell matching the global peak.
    - ``emphasize_baseline``: italicizes and slightly mutes the baseline
      reference row so it reads as a "what would the unfine-tuned model do"
      anchor rather than another experimental row.
    - ``flatten_index``: drop the level-0 ``group_label`` from the row index
      so the table renders as a flat single-level index of just the eval
      labels. The baseline row keeps its ``baseline_label`` ("Base model")
      as its flat label rather than collapsing to the formatted P value;
      that value is broadcast across the rank cells so the LaTeX
      ``baseline_span_ranks`` post-processor can fold it into a single
      ``\\multicolumn`` block. Use this when the grouping adds no semantic
      value and you just want a flat list of variants.
    - ``hide_axis_names``: drop the row- and column-axis name rows from
      the rendered header. Useful for the flat-index Table 2 layout where
      the axis labels (``LoRA rank``, ``Variant``) duplicate information
      already conveyed by the caption and the column values. The row
      labels themselves are unaffected.
    - ``top_header``: optional banner text inserted above the column-
      label row in both the HTML preview and the saved ``.tex`` (centered,
      bold, full-width via ``\\multicolumn``). Useful for embedding a
      per-animal context line such as
      ``"Cat (baseline preference: 8.2%)"`` so the table reads as a
      standalone unit without forcing the caption to repeat the value.

    The returned object renders cleanly in Jupyter (HTML via ``Styler``) and
    via :func:`savetable` to LaTeX (``.tex``) using
    ``Styler.to_latex(convert_css=True)``. When any paper-style HTML post-
    processing kicks in (the merged single-row header, the ``top_header``
    banner) the styler is returned wrapped in :class:`_PaperStyledTable`,
    a transparent proxy that forwards ``.data`` / ``.to_latex`` and every
    other attribute through to the underlying styler.
    """
    formatted, raw = build_scenario_rank_table(
        gen_df,
        animal=animal,
        baseline_p=baseline_p,
        with_ci=with_ci,
        show_n=show_n,
        baseline_label=baseline_label,
        return_raw=True,
        **build_kwargs,
    )

    # Identify the baseline row up front so the `flatten_index` branch can
    # map it through to its post-flatten key. In the MultiIndex case it
    # lives at ``(baseline_label, <formatted_p_target>)``; we hold onto the
    # full tuple so flattening can rewrite it to the level-1 string.
    pre_baseline_key: tuple[str, str] | None = next(
        (
            k for k in formatted.index
            if isinstance(k, tuple) and k[0] == baseline_label
        ),
        None,
    )

    if flatten_index:
        # Collapse (group_label, eval_label) -> eval_label. The baseline
        # row's level-1 already holds the formatted P value (the build
        # parks it there as a single anchor instead of broadcasting across
        # rank cells), so it naturally becomes the flat row label and the
        # rank cells stay as ``missing_marker`` -- e.g.
        # ``8.9 | — | — | ... | —``. That keeps the HTML preview honest:
        # the baseline is rank-independent, so we don't paint a repeated
        # value at every rank column. Forces ``highlight_group_peak=False``
        # since there are no groups left to peak within.
        def _flat_key(k):
            if isinstance(k, tuple):
                return k[0] if k[1] == "" else k[1]
            return k
        new_idx = pd.Index([_flat_key(k) for k in formatted.index])
        idx_names = build_kwargs.get("index_names", ("Finetune", "Eval"))
        new_idx.name = idx_names[1] if len(idx_names) > 1 else None
        formatted.index = new_idx
        raw.index = new_idx.copy()
        highlight_group_peak = False

    styler = formatted.style

    if hide_axis_names:
        # Pandas Styler emits a separate header row for each axis name
        # (``LoRA rank`` above the column headers, ``Variant`` to the
        # left of the row labels). For paper layouts where those axis
        # labels are redundant with the caption, drop both -- this leaves
        # the column-value row alone and the row labels unchanged.
        styler = styler.hide(axis="columns", names=True)
        styler = styler.hide(axis="index", names=True)

    # Resolve the baseline key into the same shape the styler will see:
    # the post-flatten level-1 string in the flat case, or the original
    # 2-tuple in the MultiIndex case. ``None`` when there's no baseline.
    if pre_baseline_key is None:
        baseline_keys: list = []
    elif flatten_index:
        baseline_keys = [pre_baseline_key[1]]
    else:
        baseline_keys = [pre_baseline_key]
    baseline_key = baseline_keys[0] if baseline_keys else None
    has_baseline = baseline_key is not None

    if bold_max:
        def _bold_max_row(row: pd.Series) -> list[str]:
            # Constant baseline row -> nothing to bold.
            if has_baseline and row.name == baseline_key:
                return [""] * len(row)
            raw_row = raw.loc[row.name]
            if raw_row.notna().sum() == 0:
                return [""] * len(row)
            row_max = raw_row.max(skipna=True)
            # Iterate by column name so non-rank columns (e.g. the optional
            # ``Matched`` checkbox column) are skipped automatically -- they
            # don't appear in ``raw_row.index`` and so never get bolded.
            return [
                _BOLD_CSS
                if (
                    col in raw_row.index
                    and pd.notna(raw_row[col])
                    and raw_row[col] == row_max
                )
                else ""
                for col in row.index
            ]
        styler = styler.apply(_bold_max_row, axis=1)

    # Compute table-peak winners up front so the per-group block below can
    # exclude them: we don't want a row to wear *both* an amber group-peak
    # background and a green table-peak background (Styler stacks the two
    # ``\cellcolor`` commands, which produces ugly LaTeX even when only the
    # later one wins visually).
    table_peak_winners: set = set()
    if highlight_table_peak_row or highlight_table_peak_cell:
        raw_rows_for_peak = raw.drop(baseline_key) if has_baseline else raw
        row_peaks_for_peak = raw_rows_for_peak.max(axis=1, skipna=True).dropna()
        if not row_peaks_for_peak.empty:
            _global_peak = float(row_peaks_for_peak.max())
            if highlight_table_peak_row:
                table_peak_winners = {
                    key for key, val in row_peaks_for_peak.items()
                    if float(val) == _global_peak
                }

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
        # Demote the table-peak winners (and their per-group ties) out of
        # the amber set -- they'll wear green only via the table-peak block
        # below, giving the "global champion + runner-up group champions"
        # palette without colour stacking on the global champion's cells.
        group_winners -= table_peak_winners

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

    if highlight_table_peak_row and table_peak_winners:
        # Table-wide row peak: paint a green band on the row(s) whose
        # max-over-ranks is the global max. The amber per-group block
        # above has already removed these rows from its winner set, so
        # the green stands alone (no ``\cellcolor`` stacking).
        def _highlight_table_peak_row(row: pd.Series) -> list[str]:
            if row.name in table_peak_winners:
                return [_TABLE_PEAK_CSS] * len(row)
            return [""] * len(row)
        styler = styler.apply(_highlight_table_peak_row, axis=1)

        row_keys = list(formatted.index)
        table_peak_pos = [k in table_peak_winners for k in row_keys]
        def _highlight_table_peak_index(_idx) -> list[str]:
            return [
                _TABLE_PEAK_CSS if win else ""
                for win in table_peak_pos
            ]
        if flatten_index:
            styler = styler.apply_index(
                _highlight_table_peak_index, axis=0,
            )
        else:
            styler = styler.apply_index(
                _highlight_table_peak_index, axis=0, level=1,
            )

    if highlight_table_peak_cell:
        # Table-wide cell peak: which (row, col) holds the global max value?
        # Highlights only that cell (or all tied cells), without painting
        # the row band. Useful for flat tables where the row band would
        # carry too much visual weight.
        raw_rows = raw.drop(baseline_key) if has_baseline else raw
        winner_cells: set[tuple] = set()
        global_cell_max: float | None = None
        for ridx in raw_rows.index:
            for cidx in raw_rows.columns:
                val = raw_rows.at[ridx, cidx]
                if pd.isna(val):
                    continue
                fval = float(val)
                if global_cell_max is None or fval > global_cell_max:
                    global_cell_max = fval
                    winner_cells = {(ridx, cidx)}
                elif fval == global_cell_max:
                    winner_cells.add((ridx, cidx))
        if winner_cells:
            def _highlight_table_peak_cell(row: pd.Series) -> list[str]:
                return [
                    _TABLE_PEAK_CSS if (row.name, col) in winner_cells else ""
                    for col in row.index
                ]
            styler = styler.apply(_highlight_table_peak_cell, axis=1)

    if emphasize_baseline and has_baseline:
        def _emph_baseline(row: pd.Series) -> list[str]:
            if row.name == baseline_key:
                return [_BASELINE_CSS] * len(row)
            return [""] * len(row)
        styler = styler.apply(_emph_baseline, axis=1)
        # Italicize the row label too so the emphasis is visible in the
        # left-hand index columns (which Styler.apply does not touch).
        # In the MultiIndex case ``baseline_key`` is a 2-tuple of
        # (level-0, level-1); apply the style positionally on each level so
        # the embedded P(target) cell in level-1 picks up the same muted
        # styling as the "Base model" label in level-0. In the flat case
        # the index is single-level and we apply once with no ``level=``.
        row_keys = list(formatted.index)
        baseline_pos = [k == baseline_key for k in row_keys]
        def _emph_baseline_index(_idx) -> list[str]:
            return [_BASELINE_CSS if is_b else "" for is_b in baseline_pos]
        if flatten_index:
            styler = styler.apply_index(_emph_baseline_index, axis=0)
        else:
            styler = styler.apply_index(_emph_baseline_index, axis=0, level=0)
            styler = styler.apply_index(_emph_baseline_index, axis=0, level=1)

    # Wrap in the paper-style proxy so the notebook HTML preview matches
    # the saved LaTeX (single merged header row, optional top-header
    # banner). The proxy is transparent: savetable + every Styler-aware
    # caller continues to see ``.data`` / ``.to_latex`` / forwarded
    # attribute access.
    return _PaperStyledTable(styler, top_header=top_header)


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


# Default output root for paper tables. Sibling of
# ``sl.figures.DEFAULT_FIGURES_DIR`` so tables and figures stay separated
# (and the existing per-format subdir convention used for figures can apply
# here too -- see :func:`savetable`). Overridable per call via the ``out_dir``
# argument, or globally via the ``PAPER_TABLES_DIR`` environment variable --
# set this in ``.env`` if you want tables to land on a shared filesystem
# instead of beside the working directory.
DEFAULT_TABLES_DIR: Path = Path(
    os.environ.get("PAPER_TABLES_DIR", "./tables")
)


# Substitutions applied to .tex output after pandas' to_latex render. We do
# our own escaping for `%` (which pandas no longer escapes by default in
# recent versions) and for common Unicode characters that don't have direct
# LaTeX equivalents in the default font encoding (and would render as a
# missing-glyph box / cause a compile error under pdflatex without
# ``\usepackage[utf8]{inputenc}`` plus the right font). We deliberately do
# NOT touch ``&``, ``\``, or ``_``, because pandas already produces those
# correctly (column separators / escape sequences in headers).
_LATEX_SUBS: dict[str, str] = {
    "%": r"\%",
    "±": r"$\pm$",
    # Dashes
    "—": r"---",  # em dash (U+2014)
    "–": r"--",   # en dash (U+2013)
    # Ellipsis (U+2026) -- bare ``\ldots`` is fine here because the surrounding
    # characters in our labels/captions are always non-letter (``''``, space,
    # closing paren, period), so TeX won't slurp it into a longer command name.
    "…": r"\ldots",
    # Arrows
    "→": r"$\rightarrow$",  # right arrow (U+2192)
    "←": r"$\leftarrow$",   # left arrow  (U+2190)
    # "Smart" quotes -> LaTeX paired quotes. Lets us write Pythonic strings
    # like ``"\u201cLLM\u201d gibberish"`` and have them render as proper
    # ``LLM'' typography in the paper.
    "\u201c": "``",   # left double quote
    "\u201d": "''",   # right double quote
    "\u2018": "`",    # left single quote
    "\u2019": "'",    # right single quote
    # Matched/mismatched markers used by ``add_matched_column``. ``\\checkmark``
    # ships with ``amssymb`` (already a near-universal paper preamble);
    # ``$\\times$`` is plain math-mode and works everywhere. Override
    # ``matched_marker`` if your preamble lacks ``amssymb``. We wrap both
    # in a fixed-width ``\\makebox[1em][c]{...}`` so the visually narrower
    # ``$\\times$`` glyph sits at the same x-position as the wider
    # ``\\checkmark`` -- otherwise the centered ``c`` column ends up with
    # ragged-looking marker placement across rows.
    "\u2713": r"\makebox[1em][c]{\checkmark}",  # ✓
    "\u2717": r"\makebox[1em][c]{$\times$}",    # ✗
}


# Strip a trailing ``\cline{X-Y}`` (or ``\cmidrule(lr){X-Y}``) that pandas
# emits immediately before ``\bottomrule`` even when
# ``clines="skip-last;index"`` is requested (pandas behavior here varies
# across versions; the line is always wrong regardless -- there's nothing
# to underline at the bottom of the table, and on some LaTeX setups the
# stray rule causes a "Misplaced \noalign" build error that breaks
# compilation). The regex is conservative: it only matches a rule that is
# *immediately* followed by ``\bottomrule`` with only whitespace in
# between, and accepts both pre- and post-``cmidrule`` conversion forms.
_TRAILING_CLINE_RE = re.compile(
    r"\\(?:cline|cmidrule(?:\(lr\))?)\{[^}]+\}\s*\n(\s*)\\bottomrule",
)


# Pandas emits ``\cline{X-Y}`` between MultiIndex level-0 groups when
# ``clines="skip-last;index"``. ``\cmidrule(lr){X-Y}`` (booktabs) is the
# paper-quality equivalent: the ``(lr)`` trims the rule's left and right
# ends so consecutive cmidrules don't merge into a single ugly line.
# Applied as a global replacement after :data:`_TRAILING_CLINE_RE` has
# stripped any trailing rule, so we don't have to worry about that case.
_CLINE_TO_CMIDRULE_RE = re.compile(r"\\cline\{")


def _merge_styler_header_rows(tabular: str) -> str:
    """Collapse the two header rows that pandas Styler emits when the row
    MultiIndex has names *and* the column index has no name.

    Concretely, transforms::

        \\toprule
         &  & Matched & 2 & 4 & ... \\\\
        Finetune & Eval &  &  &  & ... \\\\
        \\midrule

    into the cleaner single-row equivalent::

        \\toprule
        Finetune & Eval & Matched & 2 & 4 & ... \\\\
        \\midrule

    The function fires only when the leading-empty / trailing-empty shape
    matches verbatim, so it never touches Table 2's flatten-index header
    (which has the column-axis name in a leading cell rather than empty
    cells). On any pattern mismatch the input is returned unchanged.
    """
    lines = tabular.splitlines(keepends=True)
    n = len(lines)
    for i in range(n):
        if "\\toprule" not in lines[i]:
            continue
        # Skip blank lines after \toprule to find the two header rows.
        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        k = j + 1
        while k < n and lines[k].strip() == "":
            k += 1
        if k >= n:
            return tabular
        if not (lines[j].rstrip().endswith("\\\\") and lines[k].rstrip().endswith("\\\\")):
            return tabular
        cells_a = [c.strip() for c in lines[j].rstrip()[:-2].split("&")]
        cells_b = [c.strip() for c in lines[k].rstrip()[:-2].split("&")]
        if len(cells_a) != len(cells_b):
            return tabular
        # Determine the index-column count from the leading-empty prefix
        # of row A (which lines up with the filled prefix of row B).
        n_idx = 0
        for a, b in zip(cells_a, cells_b):
            if a == "" and b != "":
                n_idx += 1
            else:
                break
        if n_idx == 0:
            return tabular
        if any(c != "" for c in cells_b[n_idx:]):
            return tabular
        merged = cells_b[:n_idx] + cells_a[n_idx:]
        merged_line = " & ".join(merged) + " \\\\\n"
        return "".join(lines[:j] + [merged_line] + lines[k + 1:])
    return tabular


def _split_styled_cell(cell: str) -> tuple[str, str]:
    """Split a styled LaTeX cell into ``(prefix, value)`` on its trailing
    whitespace.

    Cells emitted by pandas Styler look like
    ``"\\itshape \\color[HTML]{5A5A5A} 1.4"``: a sequence of styling
    commands followed by the visible value. We peel off the value via
    ``rsplit(None, 1)`` so callers can rebuild the cell with a different
    body while preserving the styling prefix. Cells without a styling
    prefix (e.g. ``"1.4"``) round-trip cleanly with ``prefix == ""``.
    """
    toks = cell.rsplit(None, 1)
    if len(toks) == 2:
        return toks[0], toks[1]
    return "", cell


def _rewrite_baseline_span_row(
    tabular: str,
    *,
    baseline_label: str,
    n_idx: int,
    n_matched: int,
    n_rank: int,
    missing_marker: str = "---",
) -> str:
    """Rewrite the baseline reference row so the deterministic baseline
    P(target) value spans every rank column inside one centered
    ``\\multicolumn`` cell.

    Two layouts are supported, distinguished by the row index nesting:

    *2-level index* (Table 1: ``Finetune × Eval``). The build parks the
    formatted baseline value in the MultiIndex level-1 slot and dashes
    out the rank cells, so::

        \\itshape ... Base model & \\itshape ... 1.4 & \\itshape ... --- & --- & ... \\\\

    becomes::

        \\itshape ... Base model & \\itshape ... --- & \\itshape ... --- & \\multicolumn{8}{c}{\\itshape ... 1.4} \\\\

    The level-1 cell is replaced with ``---`` so the baseline value
    appears exactly once (in the multicolumn body).

    *1-level index* (Table 2 with ``flatten_index=True``). The styler
    keeps ``baseline_label`` as the row label and broadcasts the
    formatted value across every rank cell, so::

        \\itshape ... Base model & \\itshape ... 8.9 & \\itshape ... 8.9 & \\itshape ... 8.9 & ... \\\\

    becomes::

        \\itshape ... Base model & \\multicolumn{8}{c}{\\itshape ... 8.9} \\\\
        \\midrule

    A trailing ``\\midrule`` is inserted to visually separate the
    baseline reference from the experimental rows below -- the 2-level
    case already gets this for free via the ``\\cmidrule`` between
    multirow groups.

    Conservative: only fires when a row with the expected total cell
    count contains ``baseline_label`` in its leading cell, and the
    ``baseline_label`` substring is otherwise unique enough not to false-
    match data rows. On any pattern mismatch the input is returned
    unchanged.
    """
    lines = tabular.splitlines(keepends=True)
    expected_cells = n_idx + n_matched + n_rank
    for i, line in enumerate(lines):
        if baseline_label not in line:
            continue
        body = line.rstrip()
        if not body.endswith("\\\\"):
            continue
        cells = [c.strip() for c in body[:-2].split("&")]
        if len(cells) != expected_cells:
            continue
        if baseline_label not in cells[0]:
            continue

        if n_idx >= 2:
            # 2-level: peel value out of level-1, replace with --- (in
            # the same styling), and stash the value in the multicolumn
            # body cell at the end.
            prefix, value = _split_styled_cell(cells[1])
            if not value:
                continue
            styled_dash = (prefix + " " + missing_marker).strip()
            styled_value = (prefix + " " + value).strip()
            multi_cell = (
                f"\\multicolumn{{{n_rank}}}{{c}}{{{styled_value}}}"
            )
            new_cells = (
                cells[: n_idx - 1]
                + [styled_dash]
                + cells[n_idx : n_idx + n_matched]
                + [multi_cell]
            )
            lines[i] = " & ".join(new_cells) + " \\\\\n"
        else:
            # 1-level: the value is already broadcast across every rank
            # cell (see the styler's flatten branch). Read it from the
            # first rank cell to inherit its styling, drop the rest, and
            # collapse them all into one centered multicolumn body.
            first_rank_idx = n_idx + n_matched
            if first_rank_idx >= len(cells):
                continue
            prefix, value = _split_styled_cell(cells[first_rank_idx])
            if not value:
                continue
            styled_value = (prefix + " " + value).strip()
            multi_cell = (
                f"\\multicolumn{{{n_rank}}}{{c}}{{{styled_value}}}"
            )
            new_cells = (
                cells[: n_idx + n_matched]
                + [multi_cell]
            )
            # Insert a full-width ``\midrule`` after the baseline row so
            # the reference is visually severed from the experimental
            # block. Pandas wouldn't have added one here for a 1-level
            # index (no MultiIndex group separators), so we do it
            # ourselves.
            lines[i] = " & ".join(new_cells) + " \\\\\n\\midrule\n"
        return "".join(lines)
    return tabular


def _center_data_column_headers(
    tabular: str,
    *,
    n_idx: int,
    n_matched: int,
) -> str:
    """Wrap each data-column header cell in ``\\multicolumn{1}{c}{...}`` so
    the column-label text is centered even when the column body itself is
    right-aligned (the default for rank cells).

    Only the *first* header row between ``\\toprule`` and ``\\midrule`` is
    rewritten -- enough for the merged single-row header that
    :func:`_merge_styler_header_rows` produces. Cells in the leading index
    / matched-column prefix are left untouched (centering them would
    fight ``\\multirow`` and the centered ``c`` column the Matched column
    already lives in). Empty cells (e.g. an unlabeled column-axis name in
    a flat-index layout) are left as-is so the column-format spec's own
    alignment wins.
    """
    lines = tabular.splitlines(keepends=True)
    state = "before"
    for i, line in enumerate(lines):
        if state == "before":
            if "\\toprule" in line:
                state = "in_header"
            continue
        if "\\midrule" in line:
            break
        body = line.rstrip()
        if not body.endswith("\\\\"):
            continue
        cells = body[:-2].split("&")
        start = n_idx + n_matched
        if start >= len(cells):
            break
        new_cells = list(cells[:start])
        for cell in cells[start:]:
            stripped = cell.strip()
            if stripped == "":
                new_cells.append(cell)
            else:
                new_cells.append(f" \\multicolumn{{1}}{{c}}{{{stripped}}} ")
        lines[i] = " & ".join(new_cells) + " \\\\\n"
        # Only the first header row needs centering; the merged header
        # is single-row by construction, and even in the unmerged case
        # the index-name row that follows is intentionally left alone.
        break
    return "".join(lines)


def _insert_latex_top_header(
    tabular: str,
    *,
    text: str,
    n_cols: int,
) -> str:
    """Insert a centered, bold ``\\multicolumn`` banner above the column
    headers, separated from the body by a ``\\midrule``.

    Used to embed a per-animal context line (e.g.
    ``Cat (baseline preference: 8.2\\%)``) that sits above the existing
    ``Finetune | Eval | Matched | ranks...`` header. We apply
    :data:`_LATEX_SUBS` to the text in place so the caller can pass plain
    Python strings with literal ``%`` and Unicode characters, matching how
    cell contents are escaped.
    """
    escaped = text
    for src, dst in _LATEX_SUBS.items():
        escaped = escaped.replace(src, dst)
    new_row = (
        f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{escaped}}}}} \\\\\n"
        f"\\midrule\n"
    )
    lines = tabular.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "\\toprule" in line:
            lines.insert(i + 1, new_row)
            break
    return "".join(lines)


# ---- HTML post-processing ------------------------------------------------
#
# Mirrors the LaTeX post-processing (``_merge_styler_header_rows``,
# ``_insert_latex_top_header``) so the rendered notebook HTML lines up
# visually with the saved ``.tex`` -- single header row + optional top
# banner. We do this via regex on the Styler's ``_repr_html_`` output
# rather than subclassing the (private, version-fragile) Styler internals.

_HTML_TH_RE = re.compile(r"<th([^>]*)>(.*?)</th>", re.DOTALL)

_HTML_THEAD_TWO_ROWS_RE = re.compile(
    r"(<thead>\s*)(<tr>.*?</tr>)\s*(<tr>.*?</tr>)(\s*</thead>)",
    re.DOTALL,
)


def _is_blank_html_cell(content: str) -> bool:
    """True when the visible content of a ``<th>`` is empty (whitespace or
    ``&nbsp;``). Pandas Styler renders index / column-name placeholders as
    ``&nbsp;``; we treat those as blank for the merger pattern."""
    return content.strip() in ("", "&nbsp;")


def _merge_html_header_rows(html: str) -> str:
    """Collapse the Styler's two ``<thead>`` rows (column-label row + index-
    names row) into a single row, mirroring :func:`_merge_styler_header_rows`
    for the LaTeX path.

    The Styler emits this layout when the row ``MultiIndex`` has names and
    the column index has no name::

        <thead>
          <tr> <th>&nbsp;</th><th>&nbsp;</th><th>Matched</th><th>2</th>... </tr>
          <tr> <th>Finetune</th><th>Eval</th><th>&nbsp;</th><th>&nbsp;</th>... </tr>
        </thead>

    We replace it with::

        <thead>
          <tr> <th>Finetune</th><th>Eval</th><th>Matched</th><th>2</th>... </tr>
        </thead>

    The function fires only when the leading-blank / trailing-blank shape
    matches exactly. On any mismatch (e.g. the flat-index Table 2 layout,
    which has no index-names row to begin with) the HTML is returned
    unchanged.
    """
    m = _HTML_THEAD_TWO_ROWS_RE.search(html)
    if m is None:
        return html
    prefix, row1, row2, suffix = m.groups()
    row1_cells = _HTML_TH_RE.findall(row1)
    row2_cells = _HTML_TH_RE.findall(row2)
    if not row1_cells or len(row1_cells) != len(row2_cells):
        return html

    n_idx = 0
    for (_, a_content), (_, b_content) in zip(row1_cells, row2_cells):
        if _is_blank_html_cell(a_content) and not _is_blank_html_cell(b_content):
            n_idx += 1
        else:
            break
    if n_idx == 0:
        return html
    if any(not _is_blank_html_cell(c) for _, c in row2_cells[n_idx:]):
        return html

    new_cells: list[str] = []
    for i, ((a_attrs, a_content), (b_attrs, b_content)) in enumerate(
        zip(row1_cells, row2_cells)
    ):
        if i < n_idx:
            new_cells.append(f"<th{b_attrs}>{b_content}</th>")
        else:
            new_cells.append(f"<th{a_attrs}>{a_content}</th>")
    new_row1 = "<tr>" + "".join(new_cells) + "</tr>"
    new_thead = f"{prefix}{new_row1}{suffix}"
    return html[: m.start()] + new_thead + html[m.end() :]


def _insert_html_top_header(html: str, text: str) -> str:
    """Insert a bold, full-width banner row at the top of ``<thead>``.

    Mirror of :func:`_insert_latex_top_header` for the HTML preview: lets
    the notebook display match the ``\\multicolumn{N}{c}{\\textbf{...}}``
    banner used in the saved ``.tex`` so editors can sanity-check the
    rendered text in-situ.
    """
    m = re.search(r"<thead>\s*(<tr>.*?</tr>)", html, re.DOTALL)
    if m is None:
        return html
    first_row = m.group(1)
    n_cols = len(_HTML_TH_RE.findall(first_row))
    if n_cols == 0:
        return html
    banner = (
        f'<tr><th colspan="{n_cols}" '
        f'style="text-align: center; font-weight: bold; '
        f'border-top: 1px solid #000; border-bottom: 1px solid #000; '
        f'padding: 4px 0;">{text}</th></tr>'
    )
    return re.sub(r"<thead>", f"<thead>{banner}", html, count=1)


class _PaperStyledTable:
    """Drop-in wrapper around a pandas ``Styler`` that paper-style-formats
    the HTML preview to match the saved LaTeX.

    Pandas Styler renders a MultiIndex-with-names table as two header rows
    (column labels then index names) and right-aligned column-axis values.
    Our LaTeX pipeline collapses both into a single row and centers the
    rank-column header text (see :func:`_merge_styler_header_rows`,
    :func:`_center_data_column_headers`); without this wrapper the notebook
    preview drifts visually from the rendered ``.tex``.

    The wrapper exposes ``.data``, ``.to_latex(...)``, and forwards every
    other attribute access through to the underlying styler, so
    :func:`savetable` and any other Styler-aware caller continues to work
    unchanged. The ``top_header`` attribute is read by
    :func:`_styler_to_paper_latex` so the user only needs to set the
    banner text once (at build time) and it carries through to both
    notebook display and the saved ``.tex``.
    """

    def __init__(self, styler, *, top_header: str | None = None):
        self._styler = styler
        self.top_header = top_header

    @property
    def data(self) -> pd.DataFrame:
        return self._styler.data

    def to_latex(self, *args, **kwargs) -> str:
        return self._styler.to_latex(*args, **kwargs)

    def _repr_html_(self) -> str:
        html = self._styler._repr_html_()
        html = _merge_html_header_rows(html)
        if self.top_header:
            html = _insert_html_top_header(html, self.top_header)
        return html

    def __getattr__(self, name):
        # __getattr__ only fires for attrs not found on the wrapper itself,
        # so the explicit ``data`` / ``to_latex`` / ``_repr_html_`` /
        # ``top_header`` / ``_styler`` definitions above always win.
        # Guard against recursion if ``_styler`` was never assigned (e.g.
        # ``__init__`` raised mid-construction) by looking the attribute
        # up directly in our ``__dict__``.
        styler = self.__dict__.get("_styler")
        if styler is None:
            raise AttributeError(name)
        return getattr(styler, name)


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


def _collapse_alignment_runs(parts: list[str], *, threshold: int = 3) -> str:
    """Collapse runs of ``threshold+`` identical alignment chars into the
    LaTeX ``*{N}{X}`` form, joining distinct runs with single spaces.

    Examples::

        ['l', 'l', 'r', 'r', 'r', 'r']           -> "ll *{4}{r}"
        ['l', 'l', 'c', 'r', 'r', 'r', 'r', 'r'] -> "ll c *{5}{r}"
        ['l', 'l']                                -> "ll"
    """
    out: list[str] = []
    i = 0
    while i < len(parts):
        j = i
        while j < len(parts) and parts[j] == parts[i]:
            j += 1
        run = j - i
        if run >= threshold:
            out.append(f"*{{{run}}}{{{parts[i]}}}")
        else:
            out.append(parts[i] * run)
        i = j
    return " ".join(out)


def _default_column_format(
    df: pd.DataFrame,
    *,
    matched_label: str = "Matched",
) -> str:
    """Build a column-format spec that left-aligns index levels, centers
    the optional ``Matched`` column, and right-aligns every other data
    column.

    For a 2-level row MultiIndex with a leading ``Matched`` column and 8
    rank columns this returns ``"ll c *{8}{r}"`` (a centered column for
    the matched checkbox/x followed by an abbreviated run of 8 right-
    aligned rank cells). Falls back to ``"ll *{N}{r}"`` when no matched
    column is present.
    """
    n_idx = df.index.nlevels
    parts: list[str] = ["l"] * n_idx
    for col in df.columns:
        parts.append("c" if col == matched_label else "r")
    return _collapse_alignment_runs(parts)


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
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    baseline_span_ranks: bool = False,
    matched_label: str = "Matched",
    center_data_headers: bool = True,
    top_header: str | None = None,
    **to_latex_kwargs,
) -> str:
    """Render ``df`` to paper-ready LaTeX.

    - Substitutes ``%`` -> ``\\%`` (the pandas default escape doesn't always
      catch this).
    - Substitutes common Unicode (``±``, ``—``) with LaTeX commands.
    - When ``column_format`` is None, builds ``"ll c *{N}{r}"`` (centering
      a leading ``Matched`` column when present) via
      :func:`_default_column_format`.
    - Replaces pandas' ``\\cline{X-Y}`` group separators with the
      booktabs-friendly ``\\cmidrule(lr){X-Y}``.
    - Merges Styler's two-row column header (column labels + index level
      names) into one when the leading-empty / trailing-empty pattern
      matches.
    - When ``baseline_span_ranks=True`` and ``baseline_label`` is given,
      wraps the rank cells of the baseline reference row in a single
      centered ``\\multicolumn`` block.
    - When ``resizebox`` is True (default), wraps the tabular in
      ``\\resizebox{\\textwidth}{!}{...}`` so it always fits the column.
    - When ``caption`` or ``label`` is provided, wraps everything in a
      ``\\begin{table}[<placement>] ... \\end{table}`` block (default
      ``placement="H"`` -- requires ``\\usepackage{float}``).
    """
    if column_format is None:
        column_format = _default_column_format(df, matched_label=matched_label)
    tabular = df.to_latex(column_format=column_format, **to_latex_kwargs)
    for src, dst in _LATEX_SUBS.items():
        tabular = tabular.replace(src, dst)
    tabular = _TRAILING_CLINE_RE.sub(r"\1\\bottomrule", tabular)
    tabular = _CLINE_TO_CMIDRULE_RE.sub(r"\\cmidrule(lr){", tabular)
    tabular = _merge_styler_header_rows(tabular)
    n_matched = 1 if matched_label in df.columns else 0
    n_rank = len(df.columns) - n_matched
    n_idx = df.index.nlevels
    if baseline_span_ranks:
        tabular = _rewrite_baseline_span_row(
            tabular,
            baseline_label=baseline_label,
            n_idx=n_idx,
            n_matched=n_matched,
            n_rank=n_rank,
        )
    if center_data_headers:
        tabular = _center_data_column_headers(
            tabular, n_idx=n_idx, n_matched=n_matched,
        )
    if top_header:
        tabular = _insert_latex_top_header(
            tabular, text=top_header, n_cols=n_idx + len(df.columns),
        )
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
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    baseline_span_ranks: bool = False,
    matched_label: str = "Matched",
    center_data_headers: bool = True,
    top_header: str | None = None,
    **to_latex_kwargs,
) -> str:
    """Render a pandas ``Styler`` to paper-ready LaTeX.

    Uses ``Styler.to_latex(convert_css=True)`` so CSS rules attached via
    :func:`style_scenario_rank_table` (``font-weight: bold``,
    ``font-style: italic``, ``background-color: ...``) translate into the
    corresponding LaTeX font/colour/cellcolor commands. Then applies the same
    ``%`` / Unicode substitutions as :func:`_to_paper_latex`, plus a couple
    of paper-grade tabular fixups:

    - ``\\cline{X-Y}`` group separators emitted by pandas are upgraded to
      ``\\cmidrule(lr){X-Y}`` (booktabs).
    - The two-row Styler column header (column labels + index level names)
      collapses to a single row when the layout matches.
    - When ``baseline_span_ranks=True`` and ``baseline_label`` is set,
      the baseline reference row is rewritten to span the rank columns
      with one centered ``\\multicolumn`` cell.
    - When ``center_data_headers=True`` (default), data-column header cells
      (everything past the index / Matched prefix in the first header row)
      are wrapped in ``\\multicolumn{1}{c}{...}``. This centers the
      column labels even though the column body itself is right-aligned
      (the natural alignment for numeric rank columns).
    - When ``top_header`` is provided (either explicitly or read from a
      :class:`_PaperStyledTable` wrapper's ``top_header`` attribute), a
      centered, bold full-width banner is inserted above the column-label
      row, separated by a ``\\midrule``. Use it to embed per-table context
      (e.g. animal name + baseline preference) without forcing it into
      the caption.

    The tabular is wrapped in ``\\resizebox{\\textwidth}{!}{...}``
    (default) and a ``\\begin{table}[H] ... \\end{table}`` block when
    ``caption`` or ``label`` is provided.
    """
    if top_header is None:
        top_header = getattr(styler, "top_header", None)
    df = styler.data
    if column_format is None:
        column_format = _default_column_format(df, matched_label=matched_label)
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
    tabular = _TRAILING_CLINE_RE.sub(r"\1\\bottomrule", tabular)
    tabular = _CLINE_TO_CMIDRULE_RE.sub(r"\\cmidrule(lr){", tabular)
    tabular = _merge_styler_header_rows(tabular)
    n_matched = 1 if matched_label in df.columns else 0
    n_rank = len(df.columns) - n_matched
    n_idx = df.index.nlevels
    if baseline_span_ranks:
        tabular = _rewrite_baseline_span_row(
            tabular,
            baseline_label=baseline_label,
            n_idx=n_idx,
            n_matched=n_matched,
            n_rank=n_rank,
        )
    if center_data_headers:
        tabular = _center_data_column_headers(
            tabular, n_idx=n_idx, n_matched=n_matched,
        )
    if top_header:
        # Top banner spans every visible column (index levels + data cols).
        tabular = _insert_latex_top_header(
            tabular, text=top_header, n_cols=n_idx + len(df.columns),
        )
    if resizebox:
        tabular = _wrap_resizebox(tabular)
    return _assemble_table_env(
        tabular, caption=caption, label=label, placement=placement,
    )


def savetable(
    df_or_styler,
    name: str,
    *,
    formats: Iterable[str] = ("tex",),
    out_dir: Path | str | None = None,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
    top_header: str | None = None,
    **to_latex_kwargs,
) -> list[Path]:
    """Save a DataFrame *or Styler* to ``<out_dir>/<ext>/<name>.<ext>``.

    Each format lands in its own per-extension subdirectory under ``out_dir``
    (``<out_dir>/tex/<name>.tex``, ``<out_dir>/csv/<name>.csv``, ...) so the
    layout matches :func:`sl.figures.savefig` and downstream tooling (LaTeX
    ``\\input`` paths, diff/preview tools) can target a single extension
    cleanly. Defaults to writing only ``.tex`` (LaTeX, paper-ready) under
    :data:`DEFAULT_TABLES_DIR` (``./tables`` by default, a sibling of the
    figures directory). Pass ``formats=("tex", "csv")`` or
    ``formats=("tex", "md")`` to opt back into the preview/diff formats.
    Override the root via ``out_dir`` per call or globally via the
    ``PAPER_TABLES_DIR`` environment variable.

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
        out_dir = DEFAULT_TABLES_DIR
    out_dir = Path(out_dir)

    is_styler = hasattr(df_or_styler, "to_latex") and hasattr(df_or_styler, "data")
    df = df_or_styler.data if is_styler else df_or_styler

    written: list[Path] = []
    for ext in formats:
        ext_norm = ext.lstrip(".")
        subdir = out_dir / ext_norm
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{name}.{ext_norm}"
        if ext_norm == "tex":
            if is_styler:
                tex = _styler_to_paper_latex(
                    df_or_styler, caption=caption, label=label,
                    column_format=column_format, top_header=top_header,
                    **to_latex_kwargs,
                )
            else:
                tex = _to_paper_latex(
                    df_or_styler, caption=caption, label=label,
                    column_format=column_format, top_header=top_header,
                    **to_latex_kwargs,
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
    "DEFAULT_TABLES_DIR",
    "PromptScenario",
    "DEFAULT_SCENARIOS",
    "SYS_VARIANT_SCENARIOS",
    "build_scenario_rank_table",
    "style_scenario_rank_table",
    "build_baseline_animal_table",
    "savetable",
]
