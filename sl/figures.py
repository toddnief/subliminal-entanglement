"""Plotting layer for paper figures.

Plot helpers (shared ``rcParams`` + per-figure functions) factored out so
they can be reused across analysis notebooks. The data layer lives in
:mod:`sl.results`.

Public surface:

- :func:`set_paper_style` — apply the shared matplotlib ``rcParams``.
- :func:`savefig` — save a figure to ``figures/paper/<name>.{pdf,png}``.
- :data:`ANIMAL_COLORS`, :data:`MODE_COLORS`, :data:`DEFAULT_ANIMALS`.
- :func:`plot_p_target_vs_rank` — animal lines vs LoRA rank.
- :func:`plot_p_target_vs_temperature` — animal lines vs teacher temperature.
- :func:`plot_p_target_by_mode` — one-animal mode comparison (DWG/SVD).
- :func:`plot_lora_spectrum_decay` — normalized singular-value decay, one
  line per LoRA rank, aggregated across modules + models.
- :func:`plot_module_spectra` — raw per-module singular-value spectra,
  one line per (filtered) LoRA module — no aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from sl import config as sl_config


PAPER_RC_PARAMS: dict = {
    "font.size":         14,
    "axes.titlesize":    17,
    "axes.labelsize":    17,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "legend.fontsize":   14,
    "legend.title_fontsize": 14,
    "figure.titlesize":  18,
}


# Module-level toggle for whether to append "(n=...)" sample-size annotations
# to legend entries in the plotting helpers. Flip to False from a notebook
# (``sl.figures.SHOW_N_IN_LEGEND = False``) to globally hide them, or pass
# ``show_n=False`` per call to override locally.
SHOW_N_IN_LEGEND: bool = True


# Module-level toggle for whether to annotate the grouped-bar "peak effect"
# panels (Fig 6b / 7b / 8b / 9b) with a small ``"r{rank}"`` label above each
# bar's confidence interval. The label identifies the LoRA rank that defines
# each (animal, mode) peak; suppressing it gives a cleaner panel when the
# rank context is already covered by the surrounding line plot or the caller
# wants to drop both ``r{rank}`` and the optional ``(n=...)`` second line.
# Flip to False from a notebook
# (``sl.figures.SHOW_BAR_RANK_LABELS = False``) to globally hide the
# above-bar labels.
SHOW_BAR_RANK_LABELS: bool = True


# Module-level default for the hierarchical level at which CIs are computed
# when ``ci="sem"`` (or ``ci="std"``) in :func:`_agg_with_ci` and the
# scenario-rank table builder. Two modes are supported:
#
# - ``"runs"``: treat every row in the bucket as an iid replicate. Variance
#   is estimated as ``std(all_rows) / sqrt(n_rows)``. This is the historical
#   default and matches the bands in earlier draft figures, but it conflates
#   training-seed noise with between-dataset noise (the standard 3 datasets
#   x 3 training seeds = 9 rows grid). With pseudo-replicates inside each
#   dataset, this CI is **not well calibrated** as a between-dataset
#   uncertainty.
# - ``"datasets"``: first collapse rows to one mean per dataset (averaging
#   over training seeds), then compute SEM across dataset means. This is the
#   "uncertainty over datasets, with training randomness partially averaged"
#   reading. With ``n_datasets`` small (typically 3), the small-sample
#   t-distribution is used for the critical value (``t_{0.975, n-1}``), so
#   the resulting band is wider but is the more honest choice for a
#   between-dataset claim.
#
# Numerically, when the design is balanced (e.g. exactly 3 training seeds in
# every dataset), the *mean* is identical under both modes -- only the band
# changes. When unbalanced, ``"datasets"`` weights each dataset equally and
# may shift the central line as well.
#
# Flip from a notebook (``sl.figures.DEFAULT_CI_LEVEL = "datasets"``) to set
# the project-wide default once, or pass ``ci_level=...`` per call to
# override locally. The dataset column defaults to ``"dataset_hash"`` (set
# by :func:`sl.results.build_gen_df`); override with
# :data:`DEFAULT_CI_DATASET_COL` if your frame uses a different column.
DEFAULT_CI_LEVEL: str = "runs"
DEFAULT_CI_DATASET_COL: str = "dataset_hash"


# Optional override for the CI critical value applied to SEM bands in
# :func:`_agg_with_ci`. When ``None`` (the default), the existing logic
# applies: ``1.96`` for ``ci_level="runs"`` and the small-sample
# t-critical (``t_{0.975, n-1}``) for ``ci_level="datasets"``. Set to a
# float to override globally -- e.g. ``sl.figures.DEFAULT_CI_CRIT = 1.0``
# from a notebook to render every figure as a one-SEM band rather than a
# 95% CI. Useful in combination with ``DEFAULT_CI_LEVEL = "datasets"``
# when the small-sample t-correction (``t_2 ~= 4.30``) makes the honest
# cluster CIs visually unreadable.
DEFAULT_CI_CRIT: float | None = None


# Project policy threshold: figures with at least this many generation seed
# replicates per (animal, rank) bucket render mean ± 1.96·SEM bands; figures
# with fewer drop to descriptive ±1σ bands instead. See
# :func:`ci_for_n_seeds` for the full rationale.
CI_SEM_MIN_N_SEEDS: int = 6


def ci_for_n_seeds(n_seeds: int | None) -> str:
    """Pick the band style for a figure based on its generation-seed count.

    Project policy for the paper figures:

    - ``n_seeds >= CI_SEM_MIN_N_SEEDS`` (=6 by default) -> ``"sem"``: the
      band is mean +/- 1.96.SEM, the standard 95% Gaussian CI. At n=6 the
      Student-t correction (``t_{0.975, 5} ~= 2.57``) is close enough to
      1.96 that the under-coverage is mild, so we keep the more familiar
      band semantics.
    - ``n_seeds in {2, 3, 4, 5}`` -> ``"std"``: the band is mean +/- 1.sigma,
      a *descriptive* one-standard-deviation band. At n=3 the SEM-based 95%
      CI is severely under-calibrated (the correct t-multiplier is ``~4.30``,
      not 1.96), so we drop the inferential reading and just show the
      across-seed spread.
    - ``n_seeds is None`` or ``n_seeds <= 1`` -> ``"std"``: degenerate (std
      is 0 or undefined), but matches the small-n branch so callers don't
      need to special-case.

    Pass the return value as the ``ci=`` argument to
    :func:`plot_p_target_vs_rank`, :func:`plot_p_target_vs_temperature`,
    :func:`plot_module_spectra`, etc. Adjust :data:`CI_SEM_MIN_N_SEEDS`
    from a notebook to shift the threshold project-wide.

    Example::

        ci = sl.figures.ci_for_n_seeds(N_GEN_SEEDS)
        plot_p_target_vs_rank(..., ci=ci)
    """
    if n_seeds is None or n_seeds < 2:
        return "std"
    if n_seeds >= CI_SEM_MIN_N_SEEDS:
        return "sem"
    return "std"


def _fmt_n_suffix(n, *, show_n: bool | None) -> str:
    """Return ``" (n=<n>)"`` when n-counts should be shown, else ``""``.

    ``n`` may be an int (cast to ``int`` for display) or a pre-formatted
    string (e.g. ``"90+9"`` for a LoRA + full-FT breakdown).

    ``show_n=None`` defers to the module-level :data:`SHOW_N_IN_LEGEND`.
    """
    if show_n is None:
        show_n = SHOW_N_IN_LEGEND
    if not show_n:
        return ""
    return f" (n={n})" if isinstance(n, str) else f" (n={int(n)})"


def set_paper_style(extra: dict | None = None) -> None:
    """Apply the paper-figure ``rcParams`` to matplotlib.

    Idempotent — call once at the top of a notebook. Pass ``extra`` to layer
    additional overrides on top of :data:`PAPER_RC_PARAMS`.
    """
    mpl.rcParams.update(PAPER_RC_PARAMS)
    if extra:
        mpl.rcParams.update(extra)


# Default output root for paper figures. Derived from ``ARTIFACTS_DIR``
# (see ``sl/config.py`` + ``.env``) so figures land next to the rest of the
# experiment artifacts and can't drift onto a different mount. Overridable
# per call via the ``out_dir`` argument.
DEFAULT_FIGURES_DIR: Path = Path(sl_config.ARTIFACTS_DIR) / "figures"


def savefig(
    fig: plt.Figure,
    name: str,
    *,
    formats: Iterable[str] = ("pdf", "png"),
    out_dir: Path | str | None = None,
    bbox_inches: str | None = "tight",
    **kwargs,
) -> list[Path]:
    """Save ``fig`` to ``<out_dir>/<ext>/<name>.<ext>`` for each ``ext`` in ``formats``.

    Each format lands in its own per-extension subdirectory under ``out_dir``
    (``<out_dir>/pdf/<name>.pdf``, ``<out_dir>/png/<name>.png``, ...) so PDFs
    and PNGs stay separated for downstream tooling (LaTeX picks up only PDFs,
    diff/preview tools pick up only PNGs). Defaults to writing both PDF
    (vector, for the paper) and PNG (preview) under :data:`DEFAULT_FIGURES_DIR`
    (= ``${ARTIFACTS_DIR}/figures``). Override per call via ``out_dir``.
    Returns the list of paths written.
    """
    if out_dir is None:
        out_dir = DEFAULT_FIGURES_DIR
    out_dir = Path(out_dir)

    written: list[Path] = []
    for ext in formats:
        ext_norm = ext.lstrip(".")
        subdir = out_dir / ext_norm
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{name}.{ext_norm}"
        fig.savefig(path, bbox_inches=bbox_inches, **kwargs)
        written.append(path)
    logger.info(f"Saved figure '{name}' -> {[str(p) for p in written]}")
    return written


ANIMAL_COLORS: dict[str, str] = {
    # Original Figure 1 palette.
    "cat":      "#1f77b4",  # blue
    "owl":      "#d95f02",  # orange
    "dolphin":  "#2ca02c",  # green
    "eagle":    "#9467bd",  # purple
    "wolf":     "#1b9e77",  # teal
    # Extended palette. Each appendix panel of 4 (see paper_figures.ipynb)
    # is tuned so its 4 animals come from 4 different hue families.
    # A1 (bear, bull, dog, dragon): brown / indigo / red / dark teal
    "bear":     "#8c564b",
    "bull":     "#4b0082",
    "dog":      "#e31a1c",
    "dragon":   "#008b8b",
    # A2 (dragonfly, elephant, kangaroo, lion): purple / magenta / sea-green / gold
    "dragonfly":"#7570b3",
    "elephant": "#e7298a",
    "kangaroo": "#20b2aa",
    "lion":     "#daa520",
    # A3 (ox, panda, pangolin, peacock): steel blue / olive / saddle brown / magenta
    "ox":       "#4682b4",
    "panda":    "#66a61e",
    "pangolin": "#8b4513",
    "peacock":  "#c71585",
    # A4 (penguin, phoenix, tiger, unicorn): midnight blue / amber / gray / orchid
    "penguin":  "#191970",
    "phoenix":  "#e6ab02",
    "tiger":    "#666666",
    "unicorn":  "#da70d6",
}

DEFAULT_ANIMALS: list[str] = ["cat", "owl", "dolphin", "eagle"]


# Per-category target palettes used by the rank-sweep plotting helpers when
# the frame contains tree- or band-category rows. Populate the actual top-5
# palette here once the discovery baseline (Phase 2 of
# plans/preference_categories.md) has run; the default cycle below is a
# safe fallback that still reads cleanly. Following the same hue-family
# rule as ANIMAL_COLORS so a 5-line panel stays readable.
# Per-tree palette covering the union of Qwen and Gemma discovered top-5s
# (see plans/preference_categories.md). Hue families chosen to be
# distinguishable in a 5-line panel.
TREE_COLORS: dict[str, str] = {
    "oak":     "#8c564b",  # brown
    "pine":    "#1b9e77",  # teal
    "banyan":  "#d62728",  # red
    "redwood": "#a0522d",  # sienna
    "bamboo":  "#bcbd22",  # olive
    "sequoia": "#6a3d9a",  # purple
    "baobab":  "#e6ab02",  # amber
    "willow":  "#2ca02c",  # green
}

# Per-band palette covering the union of Qwen and Gemma discovered top-5s.
# Keys are lowercased canonical spellings (matching TOP_TARGETS["band"]).
# Band names are multi-token so the rank-based logit metric is unusable;
# band sweeps rely on the generation eval (P(response contains band)).
BAND_COLORS: dict[str, str] = {
    "led zeppelin": "#1f77b4",  # blue
    "nirvana":      "#d62728",  # red
    "metallica":    "#666666",  # gray
    "eagles":       "#daa520",  # gold
    "the beatles":  "#9467bd",  # purple
    "radiohead":    "#2ca02c",  # green
    "pink floyd":   "#e7298a",  # pink
    "arcade fire":  "#ff7f0e",  # orange
    "queen":        "#7570b3",  # violet
}


# Category dispatch for the per-target palette. ``ANIMAL_COLORS`` keeps its
# direct alias for the animal-only legacy callers; ``TARGET_COLORS["animal"]``
# returns the same dict.
TARGET_COLORS: dict[str, dict[str, str]] = {
    "animal": ANIMAL_COLORS,
    "tree":   TREE_COLORS,
    "band":   BAND_COLORS,
}


def get_target_colors(category: str = "animal") -> dict[str, str]:
    """Return the colour palette for a given preference category, falling
    back to the animal palette when no per-category palette is defined."""
    return TARGET_COLORS.get(category, ANIMAL_COLORS)


MODE_COLORS: dict[str, str] = {
    "full": "#333333",
    "entity_only": "#1f77b4",
    "no_entity": "#ff7f0e",
    "template_only": "#2ca02c",
    "no_template": "#d62728",
    # No-template *training* ablation (use_chat_template=False; raw prompt
    # +completion concat at train + eval time). Distinct color family from
    # the DWG modes above to make the train-vs-decode distinction visually
    # obvious in side-by-side panels.
    "chat_template": "#333333",
    "no_chat_template": "#d62728",
    # Legacy component modes (NOT a true partition — kept for plotting cached runs).
    "qwen_only_attn_early": "#9467bd",
    "no_qwen_attn_early": "#8c564b",
    "qwen_only_ffn_early": "#e377c2",
    "no_qwen_ffn_early": "#7f7f7f",
    # New `only_*` / `complement_*` component modes — these *do* form a true
    # partition of (position × module × layer). Same color family as the legacy
    # modes but distinct so both can coexist on the same plot.
    "only_qwen_attn_early": "#9467bd",
    "complement_qwen_attn_early": "#5d3b8c",
    "only_qwen_ffn_early": "#e377c2",
    "complement_qwen_ffn_early": "#a8508f",
}


def _resolve_ci_level(ci_level: str | None) -> str:
    """Resolve ``ci_level=None`` to the module-level default and validate."""
    if ci_level is None:
        ci_level = DEFAULT_CI_LEVEL
    if ci_level not in ("runs", "datasets"):
        raise ValueError(
            f"ci_level must be 'runs' or 'datasets'; got {ci_level!r}"
        )
    return ci_level


def _t_critical(n: int, confidence: float = 0.95) -> float:
    """Two-sided ``t``-critical value for ``n`` observations.

    Falls back to z=1.96 when ``n <= 1`` (degenerate; std is undefined). With
    ``n=3`` (the canonical 3-dataset case), this returns ~4.30 -- substantially
    wider than 1.96, which is the point.
    """
    if n is None or n <= 1:
        return 1.96
    return float(stats.t.ppf((1 + confidence) / 2, df=n - 1))


def _agg_with_ci(
    sub: pd.DataFrame,
    ci: str | None,
    by: str | None = "rank",
    *,
    ci_level: str | None = None,
    dataset_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate ``p_target`` across seed replicates within each ``by`` bucket.

    Returns a DataFrame with columns ``[by, mean, std, n, n_runs, lo, hi]``.
    When ``by`` is ``None``, returns a single-row aggregate over all rows in
    ``sub``.

    ``ci_level`` controls the variance estimator (see :data:`DEFAULT_CI_LEVEL`
    for the full discussion):

    - ``"runs"``: SEM = std(all rows)/sqrt(n_rows), critical value = 1.96.
    - ``"datasets"``: collapse to one mean per ``dataset_col`` first, then
      SEM = std(dataset means)/sqrt(n_datasets), critical value =
      ``t_{0.975, n_datasets-1}``.

    Output columns:

    - ``n``: effective sample size used in the CI formula (= raw row count
      in ``"runs"`` mode, = number of distinct datasets in ``"datasets"``
      mode).
    - ``n_runs``: always the raw row count from ``sub``. Display sites
      (legend "(n=...)" annotations) should prefer ``n_runs`` so they stay
      stable when flipping ``ci_level`` -- the test "did I get all my
      replicates?" doesn't change with the CI definition.
    """
    ci_level = _resolve_ci_level(ci_level)
    if dataset_col is None:
        dataset_col = DEFAULT_CI_DATASET_COL

    if ci_level == "datasets":
        if dataset_col not in sub.columns:
            raise ValueError(
                f"ci_level='datasets' requires column {dataset_col!r} in the "
                f"frame; available columns: {list(sub.columns)}"
            )
        # Step 1: collapse training-seed pseudo-replicates within each
        # dataset to a single per-dataset mean, so each dataset contributes
        # equally to the across-dataset spread.
        group_keys = [by, dataset_col] if by is not None else [dataset_col]
        per_dataset = (
            sub.groupby(group_keys, dropna=False)["p_target"]
            .mean()
            .reset_index()
        )
        # Step 2: aggregate over datasets.
        if by is None:
            s = per_dataset["p_target"]
            out = pd.DataFrame([{
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "n": int(s.count()),
                "n_runs": int(len(sub)),
            }])
        else:
            out = (
                per_dataset.groupby(by)["p_target"]
                .agg(mean="mean", std=lambda x: x.std(ddof=1), n="count")
                .reset_index()
            )
            n_runs = sub.groupby(by).size().rename("n_runs").reset_index()
            out = out.merge(n_runs, on=by, how="left")
            out["n_runs"] = out["n_runs"].fillna(0).astype(int)
    else:  # ci_level == "runs"
        if by is None:
            s = sub["p_target"]
            out = pd.DataFrame([{
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "n": int(s.count()),
                "n_runs": int(len(sub)),
            }])
        else:
            g = sub.groupby(by)["p_target"]
            out = g.agg(mean="mean", std="std", n="count").reset_index()
            out["n_runs"] = out["n"].astype(int)

    if ci == "sem":
        sem = out["std"].fillna(0) / np.sqrt(out["n"].clip(lower=1))
        if DEFAULT_CI_CRIT is not None:
            # Module-level override (e.g. set to 1.0 for one-SEM bands).
            crit = pd.Series(float(DEFAULT_CI_CRIT), index=out.index)
        elif ci_level == "datasets":
            # Small-sample t-correction, computed per row because n_datasets
            # may differ across buckets (e.g. partial coverage at some ranks).
            crit = out["n"].apply(_t_critical)
        else:
            crit = pd.Series(1.96, index=out.index)
        out["lo"] = out["mean"] - crit * sem
        out["hi"] = out["mean"] + crit * sem
    elif ci == "std":
        out["lo"] = out["mean"] - out["std"].fillna(0)
        out["hi"] = out["mean"] + out["std"].fillna(0)
    elif ci == "minmax" and by is not None:
        # min/max is computed over raw runs regardless of ci_level: it's a
        # range, not an inferential band, so dataset-collapsing would just
        # hide tail behavior.
        mm = sub.groupby(by)["p_target"].agg(lo="min", hi="max").reset_index()
        out = out.merge(mm, on=by)
    elif ci == "minmax":
        out["lo"] = sub["p_target"].min()
        out["hi"] = sub["p_target"].max()
    else:
        out["lo"] = out["mean"]
        out["hi"] = out["mean"]
    out["lo"] = out["lo"].clip(lower=0)
    out["hi"] = out["hi"].clip(upper=1)
    return out


def compute_seed_ci_calibration(
    df: pd.DataFrame,
    *,
    dataset_col: str = DEFAULT_CI_DATASET_COL,
    group_keys: tuple[str, ...] = ("animal", "rank"),
    min_runs: int = 4,
    min_datasets: int = 2,
) -> pd.DataFrame:
    """For each ``group_keys`` cell, compute SEM under the iid and
    cluster-by-dataset assumptions so the design effect can be inspected.

    For the canonical seed grid (3 ``training_seed`` x 3 ``generation_seed``
    per cell), the iid SEM treats the 9 rows as independent replicates,
    while the cluster SEM first collapses to one mean per ``dataset_col``
    (= ``generation_seed`` proxy) and then takes the SEM of the 3 means.

    Under iid both estimators agree in expectation. Under positive
    intraclass correlation ``rho`` across ``generation_seed`` with cluster
    size ``m=3``, the cluster SEM grows by ``sqrt(1 + (m-1)*rho)`` while
    the iid SEM stays biased low. So ``ratio = se_cluster / se_iid`` near 1
    is direct evidence the iid approximation is not anti-conservative.

    Returns a frame with one row per ``group_keys`` cell. Columns:
    ``[*group_keys, n_runs, n_datasets, std_all, std_cluster,
    se_iid, se_cluster, ratio, t_iid, t_cluster]``. Cells with fewer than
    ``min_runs`` rows or ``min_datasets`` distinct datasets are dropped.
    """
    out_cols = list(group_keys) + [
        "n_runs",
        "n_datasets",
        "std_all",
        "std_cluster",
        "se_iid",
        "se_cluster",
        "ratio",
        "t_iid",
        "t_cluster",
    ]
    if df.empty or dataset_col not in df.columns or "p_target" not in df.columns:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict] = []
    for keys, sub in df.groupby(list(group_keys), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_runs = int(len(sub))
        n_datasets = int(sub[dataset_col].nunique())
        if n_runs < min_runs or n_datasets < min_datasets:
            continue
        std_all = float(sub["p_target"].std(ddof=1))
        se_iid = std_all / np.sqrt(n_runs) if n_runs > 1 else np.nan
        per_dataset_mean = sub.groupby(dataset_col)["p_target"].mean()
        std_cluster = float(per_dataset_mean.std(ddof=1))
        se_cluster = (
            std_cluster / np.sqrt(n_datasets) if n_datasets > 1 else np.nan
        )
        ratio = (
            se_cluster / se_iid
            if se_iid is not None and not np.isnan(se_iid) and se_iid > 0
            else np.nan
        )
        row = {k: v for k, v in zip(group_keys, keys)}
        row.update(
            {
                "n_runs": n_runs,
                "n_datasets": n_datasets,
                "std_all": std_all,
                "std_cluster": std_cluster,
                "se_iid": se_iid,
                "se_cluster": se_cluster,
                "ratio": ratio,
                "t_iid": 1.96,
                "t_cluster": _t_critical(n_datasets),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)


def plot_ci_calibration(
    calib_df: pd.DataFrame,
    *,
    animals: list[str] | None = None,
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_design_effect_guides: bool = True,
    show_n: bool | None = None,
) -> plt.Figure:
    """Scatter ``SE_iid`` (x) against ``SE_cluster`` (y), one point per
    ``(animal, rank)`` cell of the canonical seed grid.

    The dashed ``y = x`` diagonal is the iid reference. Under positive ICC
    ``rho`` across ``generation_seed`` with cluster size ``m = 3``, points
    drift up to ``y = sqrt(1 + 2*rho) * x``; the dotted guides at
    ``sqrt(2) * x`` and ``sqrt(3) * x`` mark ``rho = 0.5`` and the
    maximum-ICC ``rho = 1`` respectively. Negative residual ICC pushes
    points below ``y = x``.

    Expects ``calib_df`` with columns ``animal``, ``se_iid``, ``se_cluster``
    (output of :func:`compute_seed_ci_calibration`).
    """
    if animals is None:
        animals = DEFAULT_ANIMALS
    if colors is None:
        colors = ANIMAL_COLORS

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    df = calib_df.dropna(subset=["se_iid", "se_cluster"]).copy()

    for animal in animals:
        sub = df[df["animal"] == animal]
        if sub.empty:
            continue
        color = colors.get(animal, "#999999")
        ax.scatter(
            sub["se_iid"],
            sub["se_cluster"],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=70,
            alpha=0.85,
            zorder=3,
            label=f"{animal}{_fmt_n_suffix(len(sub), show_n=show_n)}",
        )

    other = df[~df["animal"].isin(animals)]
    if not other.empty:
        ax.scatter(
            other["se_iid"],
            other["se_cluster"],
            color="#888888",
            edgecolor="white",
            linewidth=0.6,
            s=70,
            alpha=0.6,
            zorder=3,
            label=f"other{_fmt_n_suffix(len(other), show_n=show_n)}",
        )

    if df.empty:
        hi = 1.0
    else:
        hi = float(max(df["se_iid"].max(), df["se_cluster"].max())) * 1.1
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0
    lo = 0.0

    xs = np.linspace(lo, hi, 100)
    ax.plot(
        xs,
        xs,
        "--",
        color="black",
        linewidth=1.3,
        alpha=0.85,
        zorder=2,
        label=r"$y = x$  (iid; $\rho=0$)",
    )

    if show_design_effect_guides:
        ax.plot(
            xs,
            np.sqrt(2) * xs,
            ":",
            color="#888888",
            linewidth=1.1,
            alpha=0.8,
            zorder=2,
            label=r"$\rho = 0.5$  ($\sqrt{2}\cdot x$)",
        )
        ax.plot(
            xs,
            np.sqrt(3) * xs,
            ":",
            color="#cc4444",
            linewidth=1.1,
            alpha=0.8,
            zorder=2,
            label=r"$\rho = 1$  ($\sqrt{3}\cdot x$)",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        r"$\mathrm{SE}_{\mathrm{iid}} = \sigma_{\mathrm{all}} / \sqrt{n_{\mathrm{runs}}}$"
    )
    ax.set_ylabel(
        r"$\mathrm{SE}_{\mathrm{cluster}} = \sigma_{\mathrm{ds\text{-}means}} / \sqrt{n_{\mathrm{datasets}}}$"
    )
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def compute_seed_variance_components(
    df: pd.DataFrame,
    *,
    dataset_col: str = DEFAULT_CI_DATASET_COL,
    group_keys: tuple[str, ...] = ("animal", "rank"),
    min_runs: int = 4,
    min_datasets: int = 2,
) -> pd.DataFrame:
    """For each ``group_keys`` cell, decompose ``p_target`` variance into
    within-dataset (``training_seed``) and between-dataset
    (``generation_seed``) components.

    The within-dataset SD is pooled across the datasets in the cell:
    ``sigma_within = sqrt(mean_d Var_within_d)`` where ``Var_within_d`` is
    the sample variance of the runs sharing dataset ``d`` (matches the
    residual-variance estimator from a one-way ANOVA on the cell with
    ``dataset`` as the factor). The between-dataset SD is the sample SD
    of the per-dataset means.

    Returns a frame with one row per ``group_keys`` cell. Columns:
    ``[*group_keys, n_runs, n_datasets, sigma_within, sigma_between,
    ratio]``. The ``ratio = sigma_between / sigma_within`` -- ``> 1``
    means dataset (= ``generation_seed``) accounts for more variance
    than training-init randomness.

    Cells with fewer than ``min_runs`` rows or ``min_datasets`` datasets
    are dropped.
    """
    out_cols = list(group_keys) + [
        "n_runs",
        "n_datasets",
        "sigma_within",
        "sigma_between",
        "ratio",
    ]
    if df.empty or dataset_col not in df.columns or "p_target" not in df.columns:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict] = []
    for keys, sub in df.groupby(list(group_keys), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_runs = int(len(sub))
        n_datasets = int(sub[dataset_col].nunique())
        if n_runs < min_runs or n_datasets < min_datasets:
            continue
        per_dataset = sub.groupby(dataset_col)["p_target"].agg(
            mean="mean", var=lambda x: x.var(ddof=1), n="count"
        )
        # Pooled within-dataset variance (only datasets with >=2 runs
        # contribute; others have undefined sample variance).
        valid_within = per_dataset[per_dataset["n"] >= 2]
        if valid_within.empty:
            sigma_within = np.nan
        else:
            sigma_within = float(np.sqrt(valid_within["var"].mean()))
        sigma_between = float(per_dataset["mean"].std(ddof=1))
        ratio = (
            sigma_between / sigma_within
            if sigma_within is not None
            and not np.isnan(sigma_within)
            and sigma_within > 0
            else np.nan
        )
        row = {k: v for k, v in zip(group_keys, keys)}
        row.update(
            {
                "n_runs": n_runs,
                "n_datasets": n_datasets,
                "sigma_within": sigma_within,
                "sigma_between": sigma_between,
                "ratio": ratio,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)


def plot_seed_grid_strip(
    df: pd.DataFrame,
    *,
    animal: str,
    rank: int | float,
    dataset_col: str = DEFAULT_CI_DATASET_COL,
    seed_order_col: str = "generation_seed",
    color: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_dataset_means: bool = True,
    baseline: float | None = None,
    jitter: float = 0.08,
    rng: np.random.Generator | None = None,
) -> plt.Figure | plt.Axes:
    """Strip plot of the 9 ``(training_seed, generation_seed)`` runs in
    one ``(animal, rank)`` cell.

    Each generation-seed dataset is one column on the x-axis; within each
    column the ``training_seed`` runs are plotted as horizontally jittered
    dots and (optionally) connected by a short horizontal bar at the
    per-dataset mean. The visual signature of dataset-dominated variance
    (the claim of Appendix G / the Fig 1 footnote) is "tight columns,
    well-separated column means". When ``ax`` is None, returns a new
    Figure; otherwise draws into ``ax`` and returns the Axes.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sub = df[(df["animal"] == animal) & (df["rank"] == rank)].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for animal={animal!r}, rank={rank!r} -- check the slice."
        )

    # Group columns by ``seed_order_col`` (generation_seed) when available so
    # there is exactly one column per unique seed. Grouping by ``dataset_col``
    # would produce duplicate columns (same label) whenever multiple dataset
    # hashes share a generation_seed -- e.g. different teachers / strategies
    # that survived the upstream filter.
    if seed_order_col in sub.columns and sub[seed_order_col].notna().any():
        order = sorted(sub[seed_order_col].dropna().unique().tolist())
        x_labels = [str(int(s)) for s in order]
        x_axis_label = seed_order_col
        group_col = seed_order_col
    else:
        order = sorted(sub[dataset_col].dropna().unique().tolist())
        x_labels = [str(d)[:6] for d in order]
        x_axis_label = dataset_col
        group_col = dataset_col

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
    else:
        fig = ax.figure

    if color is None:
        color = ANIMAL_COLORS.get(animal, "#1f77b4")

    for x_pos, key in enumerate(order):
        col = sub[sub[group_col] == key]
        ys = col["p_target"].to_numpy(dtype=float)
        n = len(ys)
        jit = (rng.random(n) - 0.5) * 2 * jitter if n > 1 else np.zeros(n)
        ax.scatter(
            np.full(n, x_pos) + jit,
            ys,
            s=80,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )
        if show_dataset_means and n > 0:
            mean_y = float(np.mean(ys))
            ax.hlines(
                mean_y,
                xmin=x_pos - 0.25,
                xmax=x_pos + 0.25,
                colors="black",
                linewidth=2.0,
                alpha=0.7,
                zorder=4,
            )

    if baseline is not None and not np.isnan(baseline):
        ax.axhline(
            baseline,
            linestyle="--",
            color="#888888",
            linewidth=1.0,
            alpha=0.7,
            zorder=2,
            label="baseline",
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("$P(\\mathrm{target})$")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)
    if title is None:
        rank_str = "" if rank is None else f", rank {int(rank)}"
        title = f"{animal}{rank_str}"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_seed_grid_strip_grid(
    df: pd.DataFrame,
    cells: list[tuple[str, int | float]],
    *,
    ncols: int = 2,
    dataset_col: str = DEFAULT_CI_DATASET_COL,
    seed_order_col: str = "generation_seed",
    colors: dict[str, str] | None = None,
    baselines: dict[str, float] | None = None,
    sharey: bool = True,
    panel_size: tuple[float, float] = (4.0, 3.5),
    suptitle: str | None = None,
    show_dataset_means: bool = True,
    jitter: float = 0.08,
    rng_seed: int = 0,
) -> plt.Figure:
    """Small-multiples wrapper around :func:`plot_seed_grid_strip`.

    ``cells`` is a list of ``(animal, rank)`` tuples; one panel per cell,
    laid out in ``ncols`` columns. Per-animal colors come from ``colors``
    (defaults to ``ANIMAL_COLORS``); per-animal baselines from
    ``baselines`` (e.g. the same dict passed to
    :func:`plot_p_target_vs_rank`). ``sharey`` ties the y-axes so the
    visual claim ``sigma_within << sigma_between`` is comparable across
    panels. Returns the Figure.
    """
    if not cells:
        raise ValueError("cells must be a non-empty list of (animal, rank) tuples")

    if colors is None:
        colors = ANIMAL_COLORS

    n = len(cells)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        sharey=sharey,
        squeeze=False,
    )

    rng = np.random.default_rng(rng_seed)
    for i, (animal, rank) in enumerate(cells):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        baseline = baselines.get(animal) if baselines is not None else None
        plot_seed_grid_strip(
            df,
            animal=animal,
            rank=rank,
            dataset_col=dataset_col,
            seed_order_col=seed_order_col,
            color=colors.get(animal),
            ax=ax,
            baseline=baseline,
            show_dataset_means=show_dataset_means,
            jitter=jitter,
            rng=rng,
        )

    # Hide unused axes.
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return fig


def plot_seed_variance_decomposition(
    var_df: pd.DataFrame,
    *,
    animals: list[str] | None = None,
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_design_effect_guides: bool = True,
    show_n: bool | None = None,
) -> plt.Figure:
    """Scatter ``sigma_within`` (x) against ``sigma_between`` (y), one
    point per ``(animal, rank)`` cell.

    Pooled (non-cherry-picked) companion to :func:`plot_seed_grid_strip_grid`:
    the dashed ``y = x`` line is the boundary at which dataset and
    training-init randomness contribute equally; points above it are
    cells where the dataset accounts for more variance than training
    randomness. The dotted guides at ``y = sqrt(2)*x`` and ``y = sqrt(3)*x``
    correspond to ratios where the cluster-by-dataset SE is sqrt(2) or
    sqrt(3) times the iid SE under the asymptotic ``1 + 2*rho`` design
    effect.

    Expects ``var_df`` with columns ``animal``, ``sigma_within``,
    ``sigma_between`` (output of :func:`compute_seed_variance_components`).
    """
    if animals is None:
        animals = DEFAULT_ANIMALS
    if colors is None:
        colors = ANIMAL_COLORS

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    df = var_df.dropna(subset=["sigma_within", "sigma_between"]).copy()

    for animal in animals:
        sub = df[df["animal"] == animal]
        if sub.empty:
            continue
        c = colors.get(animal, "#999999")
        ax.scatter(
            sub["sigma_within"],
            sub["sigma_between"],
            color=c,
            edgecolor="white",
            linewidth=0.6,
            s=70,
            alpha=0.85,
            zorder=3,
            label=f"{animal}{_fmt_n_suffix(len(sub), show_n=show_n)}",
        )

    other = df[~df["animal"].isin(animals)]
    if not other.empty:
        ax.scatter(
            other["sigma_within"],
            other["sigma_between"],
            color="#888888",
            edgecolor="white",
            linewidth=0.6,
            s=70,
            alpha=0.6,
            zorder=3,
            label=f"other{_fmt_n_suffix(len(other), show_n=show_n)}",
        )

    if df.empty:
        hi = 1.0
    else:
        hi = (
            float(max(df["sigma_within"].max(), df["sigma_between"].max())) * 1.1
        )
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0
    lo = 0.0

    xs = np.linspace(lo, hi, 100)
    ax.plot(
        xs,
        xs,
        "--",
        color="black",
        linewidth=1.3,
        alpha=0.85,
        zorder=2,
        label=r"$y = x$ (equal contribution)",
    )

    if show_design_effect_guides:
        ax.plot(
            xs,
            np.sqrt(2) * xs,
            ":",
            color="#888888",
            linewidth=1.1,
            alpha=0.8,
            zorder=2,
            label=r"$y = \sqrt{2}\cdot x$",
        )
        ax.plot(
            xs,
            np.sqrt(3) * xs,
            ":",
            color="#cc4444",
            linewidth=1.1,
            alpha=0.8,
            zorder=2,
            label=r"$y = \sqrt{3}\cdot x$",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\sigma_{\mathrm{within\ dataset}}$ (across training seeds)")
    ax.set_ylabel(r"$\sigma_{\mathrm{between\ datasets}}$ (across generation seeds)")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_p_target_vs_rank(
    df: pd.DataFrame,
    animals: list[str] | None = None,
    *,
    category: str = "animal",
    ci: str | None = "sem",
    ci_level: str | None = None,
    facet_by: str | None = None,
    facet_order: list | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_points: bool = True,
    colors: dict | None = None,
    baselines: dict[str, float] | None = None,
    include_full_ft: bool = True,
    linestyle: str = "-",
    marker: str = "o",
    label_suffix: str = "",
    legend: bool = True,
    show_n: bool | None = None,
):
    """Plot P(response contains target animal) vs LoRA rank.

    One line per animal, error band across seed replicates.

    Optional overlays:
    - ``baselines``: dict[animal -> p_target] drawn as horizontal dashed lines
      in the matching animal color, only for animals present in the plot.
    - ``include_full_ft``: if True and the frame has a ``full_ft`` column,
      aggregate full-FT rows per animal and draw a diamond at the next log-2
      tick past the max LoRA rank, connected to the rightmost LoRA point by
      a thin line. Requires the frame to be split correctly upstream
      (full-FT rows have ``rank`` set to ``None`` by ``build_gen_df``).

    Styling overrides for overlay use (calling the function twice on the same
    ``ax`` to compare two filtered frames on shared axes):
    - ``linestyle`` / ``marker``: matplotlib style for the per-animal line
      (default solid + circles, matching Figures 1/2).
    - ``label_suffix``: appended to each animal's legend label, e.g.
      ``" (full)"`` vs ``" (top-1 SV)"``.
    - ``legend``: set to False on overlay calls to avoid double-drawing the
      legend; the caller can build a custom one after both passes.
    - ``show_n``: append ``" (n=...)"`` to legend labels. ``None`` defers to
      :data:`SHOW_N_IN_LEGEND`; pass ``True``/``False`` to override per call.
    - ``ci_level``: hierarchical level for ``ci="sem"`` / ``ci="std"`` bands.
      ``"runs"`` treats every row as iid (historical default); ``"datasets"``
      collapses training-seed pseudo-replicates to one mean per
      ``dataset_hash`` first, then computes SEM across dataset means with a
      small-sample t-critical value. ``None`` (default) defers to
      :data:`DEFAULT_CI_LEVEL`. See its docstring for the calibration story.
      Legend ``n=`` annotations always show raw run counts regardless of
      ``ci_level`` (so the "did all my replicates load?" check stays
      stable when flipping the flag).
    """
    palette = get_target_colors(category)
    if animals is None:
        # For the animal category, default to the canonical 4-target panel.
        # For other categories, fall back to "every target present in the
        # frame" since there's no DEFAULT_<CAT>S equivalent yet.
        if category == "animal":
            animals = [a for a in DEFAULT_ANIMALS if a in df["animal"].unique()]
        else:
            animals = sorted(df["animal"].dropna().unique())
    colors = {**palette, **(colors or {})}

    if facet_by is not None:
        values = facet_order or sorted(df[facet_by].dropna().unique())
        fig, axes = plt.subplots(
            1, len(values), figsize=(6 * len(values), 5.2), sharey=True, squeeze=False
        )
        for axi, val in zip(axes[0], values):
            plot_p_target_vs_rank(
                df[df[facet_by] == val], animals,
                category=category,
                ci=ci, ci_level=ci_level,
                facet_by=None, title=f"{facet_by} = {val}",
                ax=axi, show_points=show_points, colors=colors,
                baselines=baselines, include_full_ft=include_full_ft,
                linestyle=linestyle, marker=marker,
                label_suffix=label_suffix, legend=legend, show_n=show_n,
            )
        axes[0][0].set_ylabel("% responses containing target")
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        return fig

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5.5))

    has_full_ft_col = "full_ft" in df.columns
    lora_df = df[~df["full_ft"]] if has_full_ft_col else df
    full_ft_df = df[df["full_ft"]] if (has_full_ft_col and include_full_ft) else df.iloc[0:0]

    all_ranks = sorted(lora_df["rank"].dropna().unique())
    full_pos = max(all_ranks) * 2 if all_ranks else None

    for animal in animals:
        sub = lora_df[lora_df["animal"] == animal]
        color = colors.get(animal, "gray")

        if not sub.empty:
            agg = _agg_with_ci(sub, ci, by="rank", ci_level=ci_level)
            # Show LoRA + full-FT counts separately so unbalanced/missing runs
            # are visible per animal (e.g. ``n=89+1`` flags a missing LoRA seed
            # vs the canonical 90; ``n=90+9`` vs ``n=90+1`` flags the full-FT
            # coverage gap). Drop the ``+0`` suffix when no full-FT runs apply.
            n_lora = int(len(sub))
            n_full = int((full_ft_df["animal"] == animal).sum())
            n_label = f"{n_lora}+{n_full}" if n_full else str(n_lora)
            label = f"{animal}{_fmt_n_suffix(n_label, show_n=show_n)}{label_suffix}"
            line_marker = marker if show_points else ""
            ax.plot(
                agg["rank"], agg["mean"],
                marker=line_marker, linestyle=linestyle,
                label=label, color=color, markersize=6, linewidth=2,
            )
            if ci is not None:
                ax.fill_between(agg["rank"], agg["lo"], agg["hi"], color=color, alpha=0.15)
        else:
            # Animal has no LoRA data; still want it in the legend if it has
            # a full-FT point or a baseline so the dashed line / diamond is
            # interpretable.
            agg = None

        # Full-FT diamond at the next log-2 tick to the right.
        full_sub = full_ft_df[full_ft_df["animal"] == animal]
        if not full_sub.empty and full_pos is not None:
            full_agg = _agg_with_ci(full_sub, ci, by=None, ci_level=ci_level).iloc[0]
            ax.plot(
                [full_pos], [full_agg["mean"]], marker="D", color=color,
                markersize=7, linestyle="none",
                label=(
                    None if agg is not None
                    else (
                        f"{animal} full-FT"
                        f"{_fmt_n_suffix(full_agg['n_runs'], show_n=show_n)}"
                        f"{label_suffix}"
                    )
                ),
            )
            if ci is not None and not np.isnan(full_agg["lo"]):
                ax.errorbar(
                    [full_pos], [full_agg["mean"]],
                    yerr=[
                        [full_agg["mean"] - full_agg["lo"]],
                        [full_agg["hi"] - full_agg["mean"]],
                    ],
                    fmt="none", ecolor=color, alpha=0.5, capsize=3,
                )
            # Thin connector from rightmost LoRA point to the full-FT diamond.
            if agg is not None and len(agg):
                last = agg.iloc[-1]
                ax.plot(
                    [last["rank"], full_pos], [last["mean"], full_agg["mean"]],
                    color=color, linewidth=1, alpha=0.5,
                )

        if baselines and animal in baselines:
            ax.axhline(
                baselines[animal], color=color, linestyle="--", linewidth=1, alpha=0.6,
            )

    if all_ranks:
        ax.set_xscale("log", base=2)
        ticks = list(all_ranks) + ([full_pos] if not full_ft_df.empty and full_pos else [])
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [str(int(r)) for r in all_ranks]
            + (["full"] if not full_ft_df.empty and full_pos else [])
        )
    ax.set_xlabel("LoRA rank")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    if own_fig:
        ax.set_ylabel("% responses containing target")
        plt.tight_layout()
        return fig
    return ax


def plot_p_target_vs_temperature(
    df: pd.DataFrame,
    animals: list[str] | None = None,
    *,
    ci: str | None = "sem",
    ci_level: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_points: bool = True,
    colors: dict | None = None,
    baselines: dict[str, float] | None = None,
    linestyle: str = "-",
    marker: str = "o",
    label_suffix: str = "",
    legend: bool = True,
    show_n: bool | None = None,
):
    """Plot P(response contains target animal) vs teacher generation temperature.

    Mirrors :func:`plot_p_target_vs_rank` but with ``generation_temperature`` on
    a linear x-axis (the temperature grid is roughly evenly spaced rather than
    exponential, unlike LoRA rank). One line per animal, shaded CI band across
    seed replicates within each (animal, T) cell. Frame is expected to already
    be filtered to a single LoRA rank / dataset shape; this function does no
    further filtering.

    Optional ``baselines``: dict[animal -> p_target] drawn as horizontal dashed
    lines in the matching animal color, only for animals present in the plot.

    Styling overrides for overlay use mirror :func:`plot_p_target_vs_rank`:
    ``linestyle``, ``marker``, ``label_suffix``, ``legend``, ``show_n``.
    ``ci_level`` selects the hierarchical level for ``ci="sem"``/``ci="std"``
    bands; ``None`` defers to :data:`DEFAULT_CI_LEVEL`.
    """
    if animals is None:
        animals = [a for a in DEFAULT_ANIMALS if a in df["animal"].unique()]
    colors = {**ANIMAL_COLORS, **(colors or {})}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5.5))

    all_temps = sorted(df["generation_temperature"].dropna().unique())

    for animal in animals:
        sub = df[df["animal"] == animal]
        color = colors.get(animal, "gray")
        if not sub.empty:
            agg = _agg_with_ci(sub, ci, by="generation_temperature", ci_level=ci_level)
            agg = agg.sort_values("generation_temperature")
            n_runs = int(len(sub))
            label = f"{animal}{_fmt_n_suffix(n_runs, show_n=show_n)}{label_suffix}"
            line_marker = marker if show_points else ""
            ax.plot(
                agg["generation_temperature"], agg["mean"],
                marker=line_marker, linestyle=linestyle,
                label=label, color=color, markersize=6, linewidth=2,
            )
            if ci is not None:
                ax.fill_between(
                    agg["generation_temperature"], agg["lo"], agg["hi"],
                    color=color, alpha=0.15,
                )
        if baselines and animal in baselines:
            ax.axhline(
                baselines[animal], color=color, linestyle="--", linewidth=1, alpha=0.6,
            )

    if all_temps:
        ax.set_xticks(all_temps)
        ax.set_xticklabels([f"{t:g}" for t in all_temps])
    ax.set_xlabel("Teacher generation temperature")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    if own_fig:
        ax.set_ylabel("% responses containing target")
        plt.tight_layout()
        return fig
    return ax


def plot_p_target_by_mode(
    df: pd.DataFrame,
    *,
    animal: str | None = None,
    mode_col: str = "dwg_mode",
    modes: list[str] | None = None,
    ci: str | None = "sem",
    ci_level: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    colors: dict | None = None,
    baselines: dict[str, float] | None = None,
    show_n: bool | None = None,
    mode_labels: dict[str, str] | None = None,
    legend_title: str | None = None,
    baseline_label: str = "base model",
    legend: bool = True,
):
    """Plot one animal with one line per mode, useful for DWG/SVD comparisons.

    ``show_n`` controls the ``" (n=...)"`` legend annotation; ``None`` defers
    to the module-level :data:`SHOW_N_IN_LEGEND`. ``ci_level`` selects the
    hierarchical level for ``ci="sem"``/``ci="std"`` bands; ``None`` defers
    to :data:`DEFAULT_CI_LEVEL` (see its docstring for the runs-vs-datasets
    calibration discussion).

    ``mode_labels`` optionally remaps each mode value to a display label in
    the legend (e.g. ``{"entity_only": "Entity Only"}``); modes without an
    entry fall back to their raw value. ``legend_title`` sets the legend
    title (default = no title); pass ``mode_col`` to recover the legacy
    "title = column name" behaviour. ``baseline_label`` customizes the
    dashed-baseline legend entry. Set ``legend=False`` to skip drawing the
    per-axis legend entirely (useful for side-by-side panels that share a
    single figure-level legend).
    """
    mode_labels = mode_labels or {}
    if df.empty:
        logger.warning("plot_p_target_by_mode: no rows to plot.")
        return None

    if animal is None:
        animals = sorted(df["animal"].dropna().unique())
        if len(animals) != 1:
            raise ValueError(f"Set animal=... when df has multiple animals: {animals}")
        animal = animals[0]
    sub_df = df[df["animal"] == animal]
    if sub_df.empty:
        logger.warning(f"plot_p_target_by_mode: no rows for animal={animal!r}.")
        return None

    modes = modes or sorted(sub_df[mode_col].dropna().unique())
    colors = {**MODE_COLORS, **(colors or {})}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5.5))

    for mode in modes:
        sub = sub_df[sub_df[mode_col] == mode]
        if sub.empty:
            continue
        agg = _agg_with_ci(sub, ci, by="rank", ci_level=ci_level)
        color = colors.get(mode, None)
        label_text = mode_labels.get(mode, mode)
        ax.plot(
            agg["rank"], agg["mean"], "o-",
            label=f"{label_text}{_fmt_n_suffix(agg['n_runs'].sum(), show_n=show_n)}",
            color=color,
        )
        if ci is not None:
            ax.fill_between(agg["rank"], agg["lo"], agg["hi"], color=color, alpha=0.15)

    if baselines and animal in baselines:
        ax.axhline(
            baselines[animal], color="black", linestyle="--", alpha=0.55,
            label=baseline_label,
        )

    ranks = sorted(sub_df["rank"].dropna().unique())
    if ranks:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(int(r)) for r in ranks])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("% responses containing target")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    # Lock to full 0-100% range so per-mode panels are visually comparable
    # across animals and across figures — without this, a panel with weak
    # transfer auto-scales to a tiny top and gets visually exaggerated.
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(title=legend_title or None)
    ax.set_title(title or f"{animal}: {mode_col} comparison")
    if own_fig:
        plt.tight_layout()
        return fig
    return ax


def _spectrum_central_band(
    sub: pd.DataFrame,
    *,
    central: str,
    ci: str | None,
    value_col: str,
) -> pd.DataFrame:
    """Aggregate ``value_col`` across replicates within each ``k`` bucket.

    Returns columns ``[k, mid, lo, hi, n]``. ``central`` selects the central
    statistic (``"median"`` or ``"mean"``); ``ci`` selects the band
    (``"iqr"`` for 25-75th percentile, ``"sem"`` for mean ± 1.96·SEM,
    ``"minmax"``, or ``None`` for no band).
    """
    g = sub.groupby("k")[value_col]
    if central == "mean":
        out = g.agg(mid="mean", std="std", n="count").reset_index()
    elif central == "median":
        out = g.agg(mid="median", n="count").reset_index()
    else:
        raise ValueError(f"Unknown central={central!r}")

    if ci == "sem" and central == "mean":
        sem = out["std"].fillna(0) / np.sqrt(out["n"].clip(lower=1))
        out["lo"] = out["mid"] - 1.96 * sem
        out["hi"] = out["mid"] + 1.96 * sem
    elif ci == "iqr":
        q = sub.groupby("k")[value_col].quantile([0.25, 0.75]).unstack()
        q.columns = ["lo", "hi"]
        out = out.merge(q.reset_index(), on="k")
    elif ci == "minmax":
        mm = sub.groupby("k")[value_col].agg(lo="min", hi="max").reset_index()
        out = out.merge(mm, on="k")
    else:
        out["lo"] = out["mid"]
        out["hi"] = out["mid"]
    return out


def plot_lora_spectrum_decay(
    spectrum_df: pd.DataFrame,
    *,
    norm: str = "top1",
    ranks: list[int] | None = None,
    central: str = "median",
    ci: str | None = "iqr",
    facet_by: str | None = None,
    facet_order: list | None = None,
    title: str | None = None,
    cmap_name: str = "viridis",
    show_points: bool = True,
    ax: plt.Axes | None = None,
    show_n: bool | None = None,
):
    """Plot the (normalized) LoRA singular-value spectrum, one line per rank.

    Each line is the across-(model × layer) ``central`` statistic of
    ``s_k / s_1`` (or ``s_k / ||s||_2``) at singular index ``k``, with a
    shaded ``ci`` band. By default uses median + IQR — robust to outliers.

    Parameters
    ----------
    spectrum_df:
        Long-form frame produced by :func:`sl.results.load_lora_spectrum_df`.
        Must have columns ``k``, ``lora_rank``, and the chosen normalized
        spectrum column (``s_norm_top1`` or ``s_norm_l2``).
    norm:
        ``"top1"`` -> ``s_k / s_1`` (default), or ``"l2"`` -> ``s_k / ||s||_2``.
    ranks:
        LoRA ranks to include, in plot order. Defaults to all ranks present.
    central, ci:
        Central statistic and uncertainty band. ``central`` ∈
        {``"median"``, ``"mean"``}; ``ci`` ∈ {``"iqr"``, ``"sem"``,
        ``"minmax"``, ``None``}.
    facet_by:
        Optional column to facet on (e.g. ``"module_type"`` for a
        7-panel breakdown of q/k/v/o/gate/up/down).
    cmap_name:
        Matplotlib colormap to color rank lines from low (dark) to high (light).
    show_n:
        Append ``" (n=<n_modules>)"`` to legend labels. ``None`` defers to the
        module-level :data:`SHOW_N_IN_LEGEND`; pass ``True``/``False`` to
        override per call.
    """
    value_col = {"top1": "s_norm_top1", "l2": "s_norm_l2"}.get(norm)
    if value_col is None:
        raise ValueError(f"norm must be 'top1' or 'l2', got {norm!r}")
    if value_col not in spectrum_df.columns:
        raise ValueError(
            f"spectrum_df missing column {value_col!r} -- expected output of "
            f"load_lora_spectrum_df."
        )
    if spectrum_df.empty:
        logger.warning("plot_lora_spectrum_decay: empty spectrum_df.")
        return None

    if facet_by is not None:
        values = facet_order or sorted(spectrum_df[facet_by].dropna().unique())
        ncols = min(len(values), 4)
        nrows = int(np.ceil(len(values) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows),
            sharex=True, sharey=True, squeeze=False,
        )
        flat_axes = [a for row in axes for a in row]
        for axi, val in zip(flat_axes, values):
            plot_lora_spectrum_decay(
                spectrum_df[spectrum_df[facet_by] == val],
                norm=norm, ranks=ranks, central=central, ci=ci,
                facet_by=None, title=f"{facet_by} = {val}",
                cmap_name=cmap_name, show_points=show_points, ax=axi,
                show_n=show_n,
            )
        for axi in flat_axes[len(values):]:
            axi.set_visible(False)
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        return fig

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))

    if ranks is None:
        ranks = sorted(spectrum_df["lora_rank"].dropna().unique())
    cmap = mpl.colormaps[cmap_name]
    n = max(len(ranks), 1)

    for i, r in enumerate(ranks):
        sub = spectrum_df[spectrum_df["lora_rank"] == r]
        if sub.empty:
            continue
        agg = _spectrum_central_band(
            sub, central=central, ci=ci, value_col=value_col
        ).sort_values("k")
        # Color from low rank (dark) to high rank (light) for visual progression.
        color = cmap(0.15 + 0.7 * (i / max(n - 1, 1)))
        n_modules = int(sub.groupby("model_hash")["layer"].nunique().sum())
        label = f"r = {int(r)}{_fmt_n_suffix(n_modules, show_n=show_n)}"
        marker = "o" if show_points and len(agg) <= 16 else None
        ax.plot(
            agg["k"], agg["mid"],
            marker=marker, linestyle="-",
            label=label, color=color, linewidth=2, markersize=4,
        )
        if ci is not None and len(agg) > 1:
            ax.fill_between(agg["k"], agg["lo"], agg["hi"], color=color, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Singular index $k$")
    ax.set_ylabel(
        r"$s_k / s_1$" if norm == "top1" else r"$s_k / \|s\|_2$"
    )
    ax.grid(which="both", alpha=0.2)
    ax.legend(loc="lower left", ncol=1)
    if title:
        ax.set_title(title)
    if own_fig:
        plt.tight_layout()
        return fig
    return ax


def plot_module_spectra(
    spectrum_df: pd.DataFrame,
    *,
    color_by: str = "module_type",
    norm: str = "raw",
    central: str | None = None,
    ci: str | None = None,
    log_y: bool = True,
    log_x: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap_name: str | None = None,
):
    """Plot LoRA singular-value spectra, optionally aggregated across replicates.

    Two modes, controlled by ``central``:

    1. **Raw mode** (``central=None``, default): one line per LoRA module in
       ``spectrum_df``, colored by ``color_by``. Useful when you've narrowed
       to a single trained adapter and want to see each module's spectrum
       directly.
    2. **Aggregated mode** (``central="mean"`` or ``"median"``): for each
       ``(color_by, k)`` bucket we collapse all rows (typically several
       ``model_hash`` replicates) into a central statistic and a ``ci``
       band — one line per ``color_by`` value with a shaded uncertainty
       region. Pre-filter ``spectrum_df`` so each ``(color_by, k)`` bucket
       contains only the replicates you want to average.

    Parameters
    ----------
    spectrum_df:
        Pre-filtered output of :func:`sl.results.load_lora_spectrum_df`.
    color_by:
        Column whose unique values become separate color groups in the
        legend. Typical choices: ``"module_type"``, ``"layer_idx"``,
        ``"model_hash"``.
    norm:
        ``"raw"`` (default) plots ``s_k``;
        ``"top1"`` plots ``s_k / s_1``;
        ``"l2"`` plots ``s_k / ||s||_2``.
    central:
        ``None`` (default) draws raw per-module lines. Set to ``"mean"`` or
        ``"median"`` to aggregate across replicates within each
        ``(color_by, k)`` bucket.
    ci:
        Uncertainty band when ``central`` is set:
        ``"sem"`` (mean ± 1.96·SEM, matches Figs 1/2/3),
        ``"std"`` (mean ± std),
        ``"iqr"`` (25-75th percentile, robust),
        ``"minmax"``, or ``None`` (no band).
    log_y, log_x:
        Use log scale on the y- / x-axis. Default log_y=True (singular
        values typically span 1-3 decades), log_x=False.
    """
    if spectrum_df.empty:
        logger.warning("plot_module_spectra: empty spectrum_df.")
        return None

    value_col = {"raw": "s", "top1": "s_norm_top1", "l2": "s_norm_l2"}.get(norm)
    if value_col is None:
        raise ValueError(f"norm must be 'raw', 'top1', or 'l2'; got {norm!r}")
    if color_by not in spectrum_df.columns:
        raise ValueError(
            f"color_by={color_by!r} not in spectrum_df columns "
            f"({list(spectrum_df.columns)})"
        )
    if central is not None and central not in ("mean", "median"):
        raise ValueError(f"central must be None, 'mean', or 'median'; got {central!r}")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.5, 5))

    # Natural sort: ints/floats sort numerically, strings alphabetically.
    # Fall back to str-key only if the column has mixed/uncomparable dtypes.
    raw_keys = list(spectrum_df[color_by].dropna().unique())
    try:
        keys = sorted(raw_keys)
    except TypeError:
        keys = sorted(raw_keys, key=str)
    if cmap_name is None:
        cmap_name = (
            "tab10" if len(keys) <= 10
            else "tab20" if len(keys) <= 20
            else "viridis"
        )
    cmap = mpl.colormaps[cmap_name]
    is_qual = hasattr(cmap, "colors") and len(cmap.colors) >= len(keys)

    if central is None:
        # ---- Raw mode: one line per LoRA module instance ----
        # Group by (layer, model_hash) so multiple replicates of the same
        # canonical layer name (different seeds) render as separate lines
        # rather than one tangled zigzag.
        for i, key in enumerate(keys):
            sub = spectrum_df[spectrum_df[color_by] == key]
            color = cmap.colors[i] if is_qual else cmap(i / max(len(keys) - 1, 1))
            for j, (_, mod_sub) in enumerate(
                sub.groupby(["layer", "model_hash"], sort=False)
            ):
                mod_sub = mod_sub.sort_values("k")
                ax.plot(
                    mod_sub["k"], mod_sub[value_col],
                    marker="o", linestyle="-",
                    color=color, markersize=4, linewidth=2, alpha=0.9,
                    label=str(key) if j == 0 else None,
                )
    else:
        # ---- Aggregated mode: central statistic + CI band per color_by ----
        agg = _module_spectra_agg(
            spectrum_df, color_by=color_by, value_col=value_col,
            central=central, ci=ci,
        )
        for i, key in enumerate(keys):
            sub_agg = agg[agg[color_by] == key].sort_values("k")
            if sub_agg.empty:
                continue
            color = cmap.colors[i] if is_qual else cmap(i / max(len(keys) - 1, 1))
            n = int(sub_agg["n"].iloc[0]) if "n" in sub_agg.columns else None
            label = f"{key} (n={n})" if n is not None and n > 1 else str(key)
            ax.plot(
                sub_agg["k"], sub_agg["mid"],
                marker="o", linestyle="-",
                color=color, markersize=4, linewidth=2,
                label=label,
            )
            if ci is not None and "lo" in sub_agg.columns:
                ax.fill_between(
                    sub_agg["k"], sub_agg["lo"], sub_agg["hi"],
                    color=color, alpha=0.18, linewidth=0,
                )

    if log_y:
        ax.set_yscale("log")
        # Singular value spectra here span ~1 decade, so the default
        # LogFormatterSciNotation (10^0, 10^-1, ...) reads as noise. Swap
        # in a ScalarFormatter so log-major ticks render as 0.1 / 1 / 10.
        plain_formatter = mpl.ticker.ScalarFormatter()
        plain_formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(plain_formatter)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Singular index $k$")
    ax.set_ylabel({
        "raw": "singular value",
        "top1": r"singular value (normalized by $s_1$)",
        "l2": r"singular value (normalized by $\|s\|_2$)",
    }[norm])
    ax.grid(alpha=0.25, which="both" if log_y else "major")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    if own_fig:
        plt.tight_layout()
        return fig
    return ax


def _module_spectra_agg(
    spectrum_df: pd.DataFrame,
    *,
    color_by: str,
    value_col: str,
    central: str,
    ci: str | None,
) -> pd.DataFrame:
    """Aggregate per-module spectra across all rows in each ``(color_by, k)``
    bucket. Returns a frame with columns ``[color_by, k, mid, lo, hi, n]``.
    """
    g = spectrum_df.groupby([color_by, "k"])[value_col]
    if central == "mean":
        out = g.agg(mid="mean", std="std", n="count").reset_index()
    elif central == "median":
        out = g.agg(mid="median", n="count").reset_index()
        out["std"] = np.nan
    else:
        raise ValueError(f"unknown central={central!r}")

    if ci == "sem":
        sem = out["std"].fillna(0) / np.sqrt(out["n"].clip(lower=1))
        out["lo"] = out["mid"] - 1.96 * sem
        out["hi"] = out["mid"] + 1.96 * sem
    elif ci == "std":
        out["lo"] = out["mid"] - out["std"].fillna(0)
        out["hi"] = out["mid"] + out["std"].fillna(0)
    elif ci == "iqr":
        q = (
            spectrum_df.groupby([color_by, "k"])[value_col]
            .quantile([0.25, 0.75]).unstack()
        )
        q.columns = ["lo", "hi"]
        out = out.merge(q.reset_index(), on=[color_by, "k"])
    elif ci == "minmax":
        mm = (
            spectrum_df.groupby([color_by, "k"])[value_col]
            .agg(lo="min", hi="max").reset_index()
        )
        out = out.merge(mm, on=[color_by, "k"])
    elif ci is None:
        pass
    else:
        raise ValueError(f"unknown ci={ci!r}")
    return out


__all__ = [
    "PAPER_RC_PARAMS",
    "DEFAULT_FIGURES_DIR",
    "SHOW_N_IN_LEGEND",
    "SHOW_BAR_RANK_LABELS",
    "DEFAULT_CI_LEVEL",
    "DEFAULT_CI_DATASET_COL",
    "DEFAULT_CI_CRIT",
    "set_paper_style",
    "savefig",
    "ANIMAL_COLORS",
    "TREE_COLORS",
    "BAND_COLORS",
    "TARGET_COLORS",
    "get_target_colors",
    "DEFAULT_ANIMALS",
    "MODE_COLORS",
    "plot_p_target_vs_rank",
    "plot_p_target_by_mode",
    "plot_lora_spectrum_decay",
    "plot_module_spectra",
    "compute_seed_ci_calibration",
    "plot_ci_calibration",
    "compute_seed_variance_components",
    "plot_seed_grid_strip",
    "plot_seed_grid_strip_grid",
    "plot_seed_variance_decomposition",
]
