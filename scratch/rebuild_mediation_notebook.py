"""Replace the likelihood-era cells of attention_ffn_mediation_results.ipynb.

Cells from the low-rank attention-write section onward are kept verbatim.
"""
import json
import sys
from pathlib import Path

NOTEBOOK = Path(sys.argv[1])
KEEP_FROM_MARKER = "## Low-rank attention-write subspaces"

INTRO = r'''# Attention/FFN mediation: KL and argmax results

Paper and appendix figures for the attention/FFN mediation and all-layer gate/up sweeps (Qwen 2.5 7B, Gemma 3 4B), followed by the low-rank attention-write subspace sweep.

Each x-axis label names what receives teacher activations, then what the rest of the model does:

- **Attention (FFN free)**: teacher attention writes; FFNs recompute freely.
- **FFN (attn. neutral)**: teacher FFN writes; attention clamped to the neutral run.
- **Attention (FFN neutral)**: teacher attention writes; FFNs clamped to the neutral run.
- **Gate / Up (rest free)**: only the named SwiGLU branch receives teacher activations; everything else recomputes freely.

These are overlapping sufficiency interventions, **not an additive decomposition**. Prompt dots are the primary units; large diamonds are unweighted means across prompts with 95% bootstrap confidence intervals.

Outcomes, each expressed as the fraction of the teacher–neutral effect recovered (1 = teacher, 0 = neutral):

- **Digit KL**: $1 - \sum \mathrm{KL}(p_\text{teacher} \,\|\, p_\text{patch}) / \sum \mathrm{KL}(p_\text{teacher} \,\|\, p_\text{neutral})$ over digit positions, renormalized over the 10 single-digit tokens.
- **Digit argmax**: among digit-related positions where the teacher and neutral top-1 tokens disagree, the fraction where the patched run's top-1 matches the teacher. Top-1 is over the full vocabulary.'''

SETUP = r'''import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

RESULTS = Path("/net/projects2/interp/subliminal/shared/results")
MEDIATION_DIRS = {
    "Qwen 2.5 7B": RESULTS / "attention_ffn_mediation",
    "Gemma 3 4B": RESULTS / "attention_ffn_mediation" / "gemma3-4b-it",
}
GATE_UP_DIRS = {
    "Qwen 2.5 7B": RESULTS / "full_model_gate_up_sufficiency" / "qwen2.5-7b",
    "Gemma 3 4B": RESULTS / "full_model_gate_up_sufficiency" / "gemma3-4b-it",
}
REPO_ROOT = Path.cwd() if (Path.cwd() / "notebooks").exists() else Path.cwd().parent
FIGURE_DIR = REPO_ROOT / "figures" / "attention_ffn_mediation_results"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Condition label -> column prefix written by the mediation scorer.
MEDIATION_CONDITIONS = {
    "Full attention": "attention",
    "FFN only": "ffn_only",
    "Attention only": "attention_only",
}
SUFFICIENCY_LABELS = {"gate": "Gate only", "up": "Up only"}
NECESSITY_LABELS = {
    "gate": "Gate ← neutral",
    "up": "Up ← neutral",
    "gate_up": "Gate + up ← neutral",
}
INTERVENTION_COLORS = {
    "Full attention": "#4C78A8",
    "FFN only": "#F58518",
    "Attention only": "#54A24B",
    "Gate only": "#7A5195",
    "Up only": "#2A9D8F",
    "Gate ← neutral": "#7A5195",
    "Up ← neutral": "#2A9D8F",
    "Gate + up ← neutral": "#F28E2B",
}
DISPLAY_LABELS = {
    "Full attention": "Attention\n(FFN free)",
    "Attention only": "Attention\n(FFN neutral)",
    "FFN only": "FFN\n(attn. neutral)",
    "Gate only": "Gate\n(rest free)",
    "Up only": "Up\n(rest free)",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 18,
})


def display_labels(names):
    return [DISPLAY_LABELS.get(name, name) for name in names]


def bootstrap_mean_ci(values, samples=20_000, seed=20260908):
    """Equal-prompt mean and percentile CI across prompts."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return values.mean(), *np.quantile(draws, [0.025, 0.975])


def save_figure(fig, filename):
    output_path = FIGURE_DIR / filename
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")'''

LOAD_MD = r'''## Load results

Mediation outcomes are aggregated directly from the version-2 pair files as per-prompt ratios of sums pooled over rows and generation seeds (matching `scripts/score_attention_ffn_mediation.py::summarize`); version-1 pair files are skipped. Gate/up outcomes come from the version-2 `kl_summary.csv` and `prediction_summary.csv` written by `scripts/score_prompt_gate_sufficiency.py`.'''

LOAD = r'''PAIR_ANALYSIS_VERSION = 2


def load_v2_pair_records():
    """Row-level KL and argmax records from version-2 mediation pair files."""
    frames = {"prediction": [], "kl": []}
    stale = []
    for model, directory in MEDIATION_DIRS.items():
        for path in sorted((directory / "pairs").glob("*.json")):
            payload = json.loads(path.read_text())
            if payload["config"]["analysis_version"] < PAIR_ANALYSIS_VERSION:
                stale.append({"model": model, "pair": path.stem})
                continue
            for kind in frames:
                frames[kind].append(
                    pd.DataFrame(payload[f"{kind}_records"]).assign(model=model)
                )
    return (
        {kind: pd.concat(parts, ignore_index=True) for kind, parts in frames.items()},
        pd.DataFrame(stale, columns=["model", "pair"]),
    )


def summarize_by_prompt(records, scope_column, numerator, denominator, output):
    """Per-prompt ratio of sums over rows and seeds, one column per condition."""
    grouped = records.groupby(["model", "category", "target", scope_column])
    numerators = [numerator.format(prefix) for prefix in MEDIATION_CONDITIONS.values()]
    summary = grouped[numerators + [denominator]].sum()
    for prefix in MEDIATION_CONDITIONS.values():
        summary[output.format(prefix)] = summary[numerator.format(prefix)] / summary[denominator]
    summary["n_seeds"] = grouped["generation_seed"].nunique()
    return summary.reset_index()


pair_records, stale_pairs = load_v2_pair_records()
mediation_kl = summarize_by_prompt(
    pair_records["kl"], "kl_scope", "{}_recovered_kl", "neutral_kl", "{}_fraction"
)
mediation_argmax = summarize_by_prompt(
    pair_records["prediction"], "token_scope",
    "{}_recoveries", "target_divergences", "{}_recovery_rate",
)
missing = sorted(set(MEDIATION_DIRS) - set(mediation_kl["model"]))
if missing:
    raise RuntimeError(f"No version-2 mediation pair files for {missing}")
if not stale_pairs.empty:
    print("Version-1 pair files skipped (rerun pending):")
    display(stale_pairs.groupby("model")["pair"].apply(", ".join).to_frame())

gate_up_results = {
    (model, filename): pd.read_csv(directory / filename)
    for model, directory in GATE_UP_DIRS.items()
    for filename in ["kl_summary.csv", "prediction_summary.csv"]
}

print("Generation seeds pooled per prompt (mediation):")
display(
    mediation_kl[mediation_kl["kl_scope"] == "digit_conditional"]
    .pivot_table(index=["category", "target"], columns="model", values="n_seeds")
)'''

FIG_MD = r'''## Sufficiency and necessity figures

- **Sufficiency**: neutral base run with teacher activations patched in. The y-axis is the fraction of the teacher effect recovered.
- **Necessity** (KL appendix figures, right panel): teacher base run with the named branch overwritten by the matched neutral activation at every layer and completion position. The y-axis is the fraction of the teacher effect that *remains*. Negative values mean the clamped run is further from the teacher than the neutral run is: the clamp disrupts the output rather than merely removing the teacher effect (pairing a teacher gate with a neutral up projection does this).

The paper uses Qwen argmax sufficiency plus the stacked argmax necessity figure below; Gemma argmax and KL for both models are appendix figures.'''

FIGURES = r'''OUTCOMES = {
    # mediation: (frame, scope column, scope, column template)
    # gate_up: (summary file, scope column, scope, sufficiency column, necessity column)
    "Digit KL (conditional, 10 bins)": {
        "mediation": (mediation_kl, "kl_scope", "digit_conditional", "{}_fraction"),
        "gate_up": (
            "kl_summary.csv", "kl_scope", "digit_conditional",
            "sufficiency_fraction", "necessity_fraction",
        ),
        "ylabel": "Fraction of teacher–neutral\nKL recovered",
    },
    "Digit argmax (top-1)": {
        "mediation": (mediation_argmax, "token_scope", "digit_related", "{}_recovery_rate"),
        "gate_up": (
            "prediction_summary.csv", "token_scope", "digit_related",
            "sufficiency_recovery_rate", "necessity_loss_rate",
        ),
        "ylabel": "Fraction of teacher\nargmax recovered",
    },
}
SUFFICIENCY_INTERVENTIONS = list(MEDIATION_CONDITIONS) + list(SUFFICIENCY_LABELS.values())
NECESSITY_INTERVENTIONS = list(NECESSITY_LABELS.values())


def sufficiency_frame(model, outcome):
    frame, scope_column, scope, template = OUTCOMES[outcome]["mediation"]
    mediation = frame[(frame["model"] == model) & (frame[scope_column] == scope)]
    parts = [
        pd.DataFrame({
            "target": mediation["target"],
            "category": mediation["category"],
            "intervention": label,
            "recovery": mediation[template.format(prefix)],
        })
        for label, prefix in MEDIATION_CONDITIONS.items()
    ]
    filename, scope_column, scope, column, _ = OUTCOMES[outcome]["gate_up"]
    gate_up = gate_up_results[(model, filename)]
    gate_up = gate_up[
        (gate_up[scope_column] == scope) & gate_up["component"].isin(SUFFICIENCY_LABELS)
    ]
    parts.append(pd.DataFrame({
        "target": gate_up["target"],
        "category": gate_up["category"],
        "intervention": gate_up["component"].map(SUFFICIENCY_LABELS),
        "recovery": gate_up[column],
    }))
    return pd.concat(parts, ignore_index=True)


def necessity_frame(model, outcome):
    filename, scope_column, scope, _, column = OUTCOMES[outcome]["gate_up"]
    frame = gate_up_results[(model, filename)]
    frame = frame[frame[scope_column] == scope]
    return pd.DataFrame({
        "target": frame["target"],
        "category": frame["category"],
        "intervention": frame["component"].map(NECESSITY_LABELS),
        "recovery": 1 - frame[column],
    })


def draw_intervention_panel(ax, frame, interventions, ylabel=None):
    """Faint dots are prompts; diamonds are prompt means with 95% bootstrap CIs."""
    rng = np.random.default_rng(29)
    for x, label in enumerate(interventions):
        subset = frame[frame["intervention"] == label]
        jitter = rng.uniform(-0.13, 0.13, size=len(subset))
        ax.scatter(
            x + jitter, subset["recovery"], color=INTERVENTION_COLORS[label],
            alpha=0.52, s=25, linewidth=0, zorder=2,
        )
        mean, low, high = bootstrap_mean_ci(subset["recovery"])
        ax.errorbar(
            x, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="black",
            markerfacecolor=INTERVENTION_COLORS[label], markersize=7, capsize=4,
            linewidth=1.6, zorder=4,
        )
    ax.axhline(0, color="0.55", linewidth=0.9)
    ax.axhline(1, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(
        range(len(interventions)), display_labels(interventions),
        rotation=90, ha="center", va="top", multialignment="center",
    )
    if ylabel:
        ax.set_ylabel(ylabel)


def style_outcome_axis(ax, lower):
    ax.set_ylim(lower, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if lower < -0.1:
        ax.axhspan(lower, 0, color="#D55E00", alpha=0.08, zorder=0)


def make_outcome_figure(model, outcome, filename, include_necessity):
    """Sufficiency (left) and optionally necessity (right) for one outcome."""
    sufficiency = sufficiency_frame(model, outcome)
    necessity = necessity_frame(model, outcome)
    lowest = sufficiency["recovery"].min()
    if include_necessity:
        lowest = min(lowest, necessity["recovery"].min())
    lower = -0.04 if lowest >= 0 else min(-0.5, np.floor(lowest * 10) / 10 - 0.05)

    if include_necessity:
        fig, (suff_ax, nec_ax) = plt.subplots(
            1, 2, figsize=(10.5, 6.2), sharey=True, constrained_layout=True,
            gridspec_kw={"width_ratios": [
                len(SUFFICIENCY_INTERVENTIONS), len(NECESSITY_INTERVENTIONS) + 0.6
            ]},
        )
    else:
        fig, suff_ax = plt.subplots(figsize=(5.4, 5.6), constrained_layout=True)
    draw_intervention_panel(
        suff_ax, sufficiency, SUFFICIENCY_INTERVENTIONS, OUTCOMES[outcome]["ylabel"]
    )
    style_outcome_axis(suff_ax, lower)
    if include_necessity:
        suff_ax.set_title("Sufficiency: neutral run +\nteacher activations (recovered)")
        draw_intervention_panel(nec_ax, necessity, NECESSITY_INTERVENTIONS)
        nec_ax.set_title("Necessity: teacher run +\nneutral activations (remaining)")
        style_outcome_axis(nec_ax, lower)
        fig.suptitle(f"{model}: {outcome}")
    else:
        suff_ax.set_title(f"{model}: sufficiency\n(neutral run + teacher activations)")
    save_figure(fig, filename)
    plt.show()
    return fig


FIGURE_SPECS = [
    ("Qwen 2.5 7B", "Digit argmax (top-1)", "attention_ffn_gate_up_qwen_argmax_main.png", False),
    ("Gemma 3 4B", "Digit argmax (top-1)", "attention_ffn_gate_up_gemma_argmax_appendix.png", False),
    ("Qwen 2.5 7B", "Digit KL (conditional, 10 bins)", "attention_ffn_gate_up_qwen_kl_appendix.png", True),
    ("Gemma 3 4B", "Digit KL (conditional, 10 bins)", "attention_ffn_gate_up_gemma_kl_appendix.png", True),
]
outcome_figures = {
    filename: make_outcome_figure(model, outcome, filename, include_necessity)
    for model, outcome, filename, include_necessity in FIGURE_SPECS
}

# Values represented by the large diamonds.
summary_rows = []
for model in GATE_UP_DIRS:
    for outcome in OUTCOMES:
        for kind, frame in [
            ("sufficiency", sufficiency_frame(model, outcome)),
            ("necessity (remaining)", necessity_frame(model, outcome)),
        ]:
            for intervention, subset in frame.groupby("intervention", sort=False):
                mean, low, high = bootstrap_mean_ci(subset["recovery"])
                summary_rows.append({
                    "model": model, "outcome": outcome, "kind": kind,
                    "intervention": intervention, "prompt_mean": mean,
                    "ci_low": low, "ci_high": high,
                })
display(
    pd.DataFrame(summary_rows)
    .pivot_table(
        index=["model", "kind", "intervention"], columns="outcome",
        values="prompt_mean", sort=False,
    )
    .style.format("{:.1%}")
)'''

NECESSITY_MD = r'''### Necessity as a decomposition of teacher-specific top-1 digits

At each digit-related position where the teacher and neutral runs disagree on the top-1 token, the clamped teacher run does exactly one of three things: keeps the teacher's digit, reverts to the neutral run's digit, or switches to a third token. The three shares sum to 100%, so nothing can "flip", and disruption from an off-distribution clamp (the third-token segment) is shown separately from genuine reversion to neutral. Bars are unweighted means across prompts.'''

NECESSITY = r'''NECESSITY_SEGMENTS = [
    ("Reverts to neutral", "reverted", "#4C78A8", "white"),
    ("Third token", "other", "#BAB0AC", "black"),
    ("Keeps teacher", "kept", "#F58518", "black"),
]


def necessity_decomposition(model):
    frame = gate_up_results[(model, "prediction_summary.csv")]
    frame = frame[frame["token_scope"] == "digit_related"]
    divergences = frame["target_divergences"]
    frame = frame.assign(
        reverted=frame["necessary_reversions"] / divergences,
        other=(frame["necessary_losses"] - frame["necessary_reversions"]) / divergences,
        kept=1 - frame["necessary_losses"] / divergences,
    )
    return frame.groupby("component")[["reverted", "other", "kept"]].mean().loc[
        list(NECESSITY_LABELS)
    ]


def make_necessity_decomposition_figure(model, filename):
    shares = necessity_decomposition(model)
    labels = [NECESSITY_LABELS[component] for component in shares.index]
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    left = np.zeros(len(shares))
    for label, column, color, text_color in NECESSITY_SEGMENTS:
        values = shares[column].to_numpy()
        ax.barh(labels, values, left=left, color=color, label=label, height=0.62)
        for y, (value, start) in enumerate(zip(values, left)):
            if value >= 0.05:
                ax.text(
                    start + value / 2, y, f"{value:.0%}",
                    ha="center", va="center", fontsize=16, color=text_color,
                )
        left += values
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.invert_yaxis()
    ax.set_xlabel("Share of teacher-specific argmax digits")
    ax.set_title(f"{model}: necessity\n(teacher run + neutral activations)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.35, -0.3), ncol=3, frameon=False,
        handlelength=1.2, columnspacing=1.0,
    )
    save_figure(fig, filename)
    plt.show()
    return fig


necessity_decomposition_figures = {
    model: make_necessity_decomposition_figure(model, filename)
    for model, filename in [
        ("Qwen 2.5 7B", "attention_ffn_gate_up_qwen_necessity_argmax_main.png"),
        ("Gemma 3 4B", "attention_ffn_gate_up_gemma_necessity_argmax_appendix.png"),
    ]
}
display(
    pd.concat(
        {model: necessity_decomposition(model) for model in GATE_UP_DIRS},
        names=["model", "component"],
    ).style.format("{:.1%}")
)'''


def cell(cell_type, source):
    lines = source.splitlines(keepends=True)
    base = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


notebook = json.loads(NOTEBOOK.read_text())
keep_index = next(
    i for i, c in enumerate(notebook["cells"])
    if "".join(c["source"]).startswith(KEEP_FROM_MARKER)
)
new_cells = [
    cell("markdown", INTRO),
    cell("code", SETUP),
    cell("markdown", LOAD_MD),
    cell("code", LOAD),
    cell("markdown", FIG_MD),
    cell("code", FIGURES),
    cell("markdown", NECESSITY_MD),
    cell("code", NECESSITY),
]
notebook["cells"] = new_cells + notebook["cells"][keep_index:]
if notebook.get("nbformat_minor", 0) >= 5:
    import uuid
    for c in notebook["cells"]:
        c.setdefault("id", uuid.uuid4().hex[:8])
NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Replaced {keep_index} cells with {len(new_cells)}; kept {len(notebook['cells']) - len(new_cells)}.")
