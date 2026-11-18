"""Prototype: decompose target-specific top-1 digits under neutral clamps (kept / reverted / other)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path("/net/projects2/interp/subliminal/shared/results/full_model_gate_up_sufficiency")
MODELS = {"Qwen 2.5 7B": "qwen2.5-7b", "Gemma 3 4B": "gemma3-4b-it"}
LABELS = {"gate": "Gate ← neutral", "up": "Up ← neutral", "gate_up": "Gate + up ← neutral"}
SEGMENTS = [
    ("Reverts to neutral's digit", "reverted", "#4C78A8"),
    ("Changes to a third token", "other", "#BAB0AC"),
    ("Keeps target's digit", "kept", "#F58518"),
]
OUT = Path(__file__).resolve().parent / "necessity_stacked_prototype.png"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, constrained_layout=True)
for ax, (model, subdir) in zip(axes, MODELS.items()):
    pred = pd.read_csv(ROOT / subdir / "prediction_summary.csv").query("token_scope == 'digit_related'")
    pred = pred.assign(
        reverted=pred.necessary_reversions / pred.target_divergences,
        other=(pred.necessary_losses - pred.necessary_reversions) / pred.target_divergences,
        kept=1 - pred.necessary_losses / pred.target_divergences,
    )
    means = pred.groupby("component")[["reverted", "other", "kept"]].mean().loc[list(LABELS)]
    left = pd.Series(0.0, index=means.index)
    for label, column, color in SEGMENTS:
        ax.barh([LABELS[c] for c in means.index], means[column], left=left, color=color, label=label)
        for y, (value, start) in enumerate(zip(means[column], left)):
            if value > 0.06:
                ax.text(start + value / 2, y, f"{value:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if column == "reverted" else "black")
        left += means[column]
    ax.set_title(model)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.invert_yaxis()
    ax.set_xlabel("Target-specific top-1 digits (target ≠ neutral argmax)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, -0.5), ncol=3, frameon=False)
fig.suptitle("Necessity: target run with one branch clamped to its neutral activation", fontsize=12)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(OUT)
