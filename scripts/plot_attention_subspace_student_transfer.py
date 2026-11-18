#!/usr/bin/env python3
"""Plot student transfer from greedy attention-subspace datasets.

Reads completed experiments from the artifact registry and writes a tidy CSV
plus appendix-ready PNG/PDF figures. Individual points are training seeds;
short horizontal bars are seed means. Panels separate each target animal's
own attention-write basis from the other animal's basis.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results"))
REGISTRY_PATH = RESULTS / "registry.json"
OUTPUT_DIR = REPO_ROOT / "figures" / "teacher_subspace"
READ_CHUNK_SIZE = 1024 * 1024
RANK_SWEEP_ROOT = RESULTS / "attention_write_rank_sweep"
STUDENT_TEACHER_BASIS_ROOT = RANK_SWEEP_ROOT / "student-teacher-basis-g123"
STUDENT_TEACHER_PREFERENCE_ROOT = RESULTS / "student_teacher_basis_preference"
STUDENT_GATE_ROOT = RESULTS / "student_gate_sufficiency"
STUDENT_FULL_MEDIATION_ROOT = RESULTS / "student_attention_ffn_mediation"
CROSS_BASIS_RUNS = [
    "qwen-cat-dolphin-g123-dolphin-on-cat-basis-full",
    "qwen-owl-wolf-eagle-g123-owl-on-cat-basis",
    "qwen-owl-wolf-eagle-g123-wolf-on-cat-basis",
    "qwen-owl-wolf-eagle-g123-eagle-on-cat-basis",
]
BASIS_LABELS = {
    "own": "Own basis",
    "cat_basis": "Cat's basis",
    "random": "Random basis",
}
BASIS_COLORS = {"own": "#4C78A8", "cat_basis": "#E45756", "random": "#9D9D9D"}
CROSS_BASIS_METRICS = {
    "Digit KL recovered": "conditional_kl_sum",
    "Altered digits recovered": "divergence_recoveries",
}

PATH_PATTERN = re.compile(r"/replace_(?P<recipient>[^_/]+)_on_(?P<basis>[^_/]+)_")
RANK_PATTERN = re.compile(r"^repgrank(?P<rank>\d+)(?:x(?P<cross_basis>[^_]+))?_")


def iter_registry_experiments(path: Path):
    """Stream top-level experiment records without loading registry.json."""
    decoder = json.JSONDecoder()
    marker = '"experiments": {'
    with path.open() as handle:
        buffer = ""
        while marker not in buffer:
            chunk = handle.read(READ_CHUNK_SIZE)
            if not chunk:
                raise KeyError(f"{path} has no top-level experiments object")
            buffer += chunk
            if len(buffer) > 2 * READ_CHUNK_SIZE:
                buffer = buffer[-READ_CHUNK_SIZE:]
        buffer = buffer.split(marker, 1)[1]
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            chunk = handle.read(READ_CHUNK_SIZE)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        def decode_next():
            nonlocal position
            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()

        while True:
            skip_whitespace()
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"Unterminated experiments object in {path}")
                refill()
                continue
            if buffer[position] == "}":
                return
            if buffer[position] == ",":
                position += 1
                skip_whitespace()
            exp_id = decode_next()
            skip_whitespace()
            if position >= len(buffer):
                refill()
                skip_whitespace()
            if buffer[position] != ":":
                raise ValueError(f"Expected ':' after experiment id {exp_id!r}")
            position += 1
            skip_whitespace()
            record = decode_next()
            yield exp_id, record


def classify_dataset(dataset_path: str) -> dict[str, object] | None:
    """Return plotting metadata for one greedy replacement dataset."""
    path = Path(dataset_path)
    if "attention_subspace_datasets" not in path.parts or "_greedy" not in dataset_path:
        return None
    match = PATH_PATTERN.search(dataset_path)
    if not match:
        return None
    metadata: dict[str, object] = {
        "recipient": match.group("recipient"),
        "basis": match.group("basis"),
        "rank": np.nan,
    }
    if path.name.startswith("repgneutral_"):
        metadata["condition"] = "Neutral"
    elif path.name.startswith("repgteacher_"):
        metadata["condition"] = "Teacher"
    else:
        rank_match = RANK_PATTERN.match(path.name)
        if not rank_match:
            return None
        rank = int(rank_match.group("rank"))
        metadata["rank"] = rank
        basis = str(metadata["basis"])
        recipient = str(metadata["recipient"])
        metadata["condition"] = (
            f"Rank {rank}\n(own basis)"
            if basis == recipient
            else f"Rank {rank}\n({basis} basis)"
        )
    return metadata


def load_results() -> pd.DataFrame:
    rows = []
    for exp_id, record in iter_registry_experiments(REGISTRY_PATH):
        if record.get("status") != "completed":
            continue
        config = record.get("config", {})
        metadata = classify_dataset(str(config.get("dataset_path", "")))
        if metadata is None or metadata["recipient"] not in {"cat", "wolf"}:
            continue
        aggregate = record.get("results", {}).get("generation_aggregate", {}).get("clean")
        if not aggregate:
            continue
        rows.append(
            {
                "exp_id": exp_id,
                "animal": config["animal"],
                "basis": metadata["basis"],
                "condition": metadata["condition"],
                "rank": metadata["rank"],
                "dataset_size": config["dataset_size"],
                "training_seed": config["training_seed"],
                "p_target": aggregate["mean_p_contains"],
                "baseline_p_target": aggregate["baseline_mean_p_contains"],
                "increase_over_base_model": aggregate["mean_p_increase"],
                "dataset_path": config["dataset_path"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No completed greedy attention-subspace experiments found")
    return frame.sort_values(["animal", "dataset_size", "condition", "training_seed"])


def add_neutral_effect(frame: pd.DataFrame) -> pd.DataFrame:
    neutral = (
        frame[frame["condition"] == "Neutral"][
            ["animal", "dataset_size", "training_seed", "p_target"]
        ]
        .drop_duplicates(["animal", "dataset_size", "training_seed"])
        .rename(columns={"p_target": "neutral_p_target"})
    )
    result = frame.merge(
        neutral,
        on=["animal", "dataset_size", "training_seed"],
        how="left",
        validate="many_to_one",
    )
    result["increase_over_neutral"] = result["p_target"] - result["neutral_p_target"]
    return result


def condition_color(label: str) -> str:
    if label == "Neutral":
        return "#9D9D9D"
    if label == "Teacher":
        return "#222222"
    if "cat basis" in label or "own basis" in label and "Rank" in label:
        return "#E45756"
    return "#4C78A8"


def draw_points(ax, frame: pd.DataFrame, labels: list[str], column: str) -> None:
    for x, label in enumerate(labels):
        values = frame.loc[frame["condition"] == label, column].dropna().to_numpy()
        if not len(values):
            continue
        offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        color = condition_color(label)
        ax.scatter(x + offsets, values, s=45, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.plot([x - 0.16, x + 0.16], [values.mean(), values.mean()], color=color, linewidth=2.4)
    ax.set_xticks(range(len(labels)), labels)
    ax.tick_params(axis="x", labelrotation=25)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")


def make_figure(frame: pd.DataFrame, dataset_size: int) -> None:
    animals = ["cat", "wolf"]
    basis_rows = ["own", "other"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="col", constrained_layout=True)
    for column, animal in enumerate(animals):
        animal_subset = frame[
            (frame["animal"] == animal) & (frame["dataset_size"] == dataset_size)
        ]
        other_animal = "wolf" if animal == "cat" else "cat"
        for row, relation in enumerate(basis_rows):
            basis = animal if relation == "own" else other_animal
            rank_rows = animal_subset[
                (animal_subset["basis"] == basis)
                & ~animal_subset["condition"].isin(["Neutral", "Teacher"])
            ].copy()
            rank_rows["plot_label"] = rank_rows["rank"].map(
                lambda rank: f"Rank {int(rank)}"
            )

            controls = (
                animal_subset[animal_subset["condition"].isin(["Neutral", "Teacher"])]
                .sort_values(["condition", "training_seed"])
                .drop_duplicates(["condition", "training_seed"])
                .copy()
            )
            controls["plot_label"] = controls["condition"]
            subset = pd.concat([controls, rank_rows], ignore_index=True)
            labels = ["Neutral"]
            labels.extend(
                f"Rank {int(rank)}" for rank in sorted(rank_rows["rank"].dropna().unique())
            )
            if "Teacher" in set(controls["condition"]):
                labels.append("Teacher")

            ax = axes[row, column]
            draw_points(
                ax,
                subset.rename(columns={"condition": "source_condition"})
                .rename(columns={"plot_label": "condition"}),
                labels,
                "p_target",
            )
            baseline = animal_subset["baseline_p_target"].dropna()
            if len(baseline):
                ax.axhline(baseline.iloc[0], color="0.55", linestyle="--", linewidth=1)
            if row == 0:
                ax.set_title(f"{animal.capitalize()} target")
            row_label = "Own basis" if relation == "own" else "Other animal's basis"
            ax.set_ylabel(
                f"{row_label}\nP(response contains target)"
                if column == 0
                else ""
            )
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    fig.suptitle(f"Preference transfer from attention-write subspace datasets ({dataset_size:,} examples)")
    stem = OUTPUT_DIR / f"student_transfer_{dataset_size}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_cross_basis_rows() -> pd.DataFrame:
    """Load held-out teacher-forced rows for own/cat/random basis comparisons."""
    parts = []
    for run_name in CROSS_BASIS_RUNS:
        path = RANK_SWEEP_ROOT / run_name / "analytics" / "all-containing" / "row_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run scripts/analyze_attention_write_rank_sweep.py "
                f"{RANK_SWEEP_ROOT / run_name}"
            )
        rows = pd.read_csv(path)
        rows = rows[
            (rows["evaluation_split"] == "test")
            & (rows["token_scope"] == "teacher_argmax_digit")
        ].copy()
        target = rows["target"].str.split(":", n=1, expand=True)
        rows["animal"] = target[0]
        rows["basis"] = target[1].fillna("own")
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def _rank_values(rows: pd.DataFrame) -> list[int]:
    rank_sets = []
    for _, animal_rows in rows.groupby("animal"):
        rank_sets.append(
            {
                int(condition.removeprefix("rank_"))
                for condition in animal_rows["condition"].unique()
                if condition.startswith("rank_")
            }
        )
    return sorted(set.intersection(*rank_sets) - {0})


def _recovery_values(wide: pd.DataFrame, ranks: list[int]) -> np.ndarray:
    values = np.empty((len(CROSS_BASIS_METRICS), len(ranks)))
    neutral_kl = wide[("conditional_kl_sum", "neutral")].sum()
    neutral_divergences = wide[("teacher_neutral_divergences", "neutral")].sum()
    for rank_i, rank in enumerate(ranks):
        condition = f"rank_{rank}"
        values[0, rank_i] = 1 - wide[("conditional_kl_sum", condition)].sum() / neutral_kl
        values[1, rank_i] = (
            wide[("divergence_recoveries", condition)].sum() / neutral_divergences
        )
    return values


def recovery_curve(
    rows: pd.DataFrame,
    *,
    n_boot: int = 2_000,
    seed: int = 0,
    group_column: str = "animal",
) -> pd.DataFrame:
    """Pool recovery ratios, resampling rows within the requested groups."""
    ranks = _rank_values(rows)
    wide = rows.pivot_table(
        index=[group_column, "dataset_row_i"],
        columns="condition",
        values=[
            "conditional_kl_sum",
            "divergence_recoveries",
            "teacher_neutral_divergences",
        ],
        aggfunc="sum",
    )
    point = _recovery_values(wide, ranks)
    low = high = np.full_like(point, np.nan)
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot, *point.shape))
        groups = [group for _, group in wide.groupby(level=group_column)]
        for draw_i in range(n_boot):
            sampled = pd.concat(
                [
                    group.iloc[rng.integers(0, len(group), len(group))]
                    for group in groups
                ]
            )
            draws[draw_i] = _recovery_values(sampled, ranks)
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    records = []
    for metric_i, metric in enumerate(CROSS_BASIS_METRICS):
        for rank_i, rank in enumerate(ranks):
            records.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "value": point[metric_i, rank_i],
                    "low": low[metric_i, rank_i],
                    "high": high[metric_i, rank_i],
                }
            )
    return pd.DataFrame(records)


def make_cross_basis_figure(rows: pd.DataFrame) -> None:
    """Plot teacher-forced KL and altered-argmax recovery across basis sources."""
    recipients = sorted(rows["animal"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, constrained_layout=True)
    for basis, label in BASIS_LABELS.items():
        basis_rows = rows[rows["basis"] == basis]
        if basis_rows.empty:
            continue
        color = BASIS_COLORS[basis]
        pooled = recovery_curve(basis_rows)
        for animal in recipients:
            animal_rows = basis_rows[basis_rows["animal"] == animal]
            if animal_rows.empty:
                continue
            curve = recovery_curve(animal_rows, n_boot=0)
            for ax, metric in zip(axes, CROSS_BASIS_METRICS):
                metric_curve = curve[curve["metric"] == metric]
                ax.plot(
                    metric_curve["rank"],
                    metric_curve["value"],
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                )
        for ax, metric in zip(axes, CROSS_BASIS_METRICS):
            metric_curve = pooled[pooled["metric"] == metric]
            ax.fill_between(
                metric_curve["rank"],
                metric_curve["low"],
                metric_curve["high"],
                color=color,
                alpha=0.18,
            )
            ax.plot(
                metric_curve["rank"],
                metric_curve["value"],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=label,
            )
    ticks = [1, 8, 32, 128, 256]
    for ax, metric in zip(axes, CROSS_BASIS_METRICS):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(rank) for rank in ticks])
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.axhline(1, color="0.65", linestyle="--", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel(r"Rank $k$ per layer")
        ax.set_title(metric)
    axes[0].set_ylabel("Fraction of target effect recovered")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Recipients: " + ", ".join(animal.capitalize() for animal in recipients))
    stem = OUTPUT_DIR / "attention_write_rank_cross_basis_qwen"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_student_teacher_basis_rows() -> pd.DataFrame:
    """Load held-out student-write rows for teacher/random basis comparison."""
    path = (
        STUDENT_TEACHER_BASIS_ROOT
        / "analytics"
        / "all-containing"
        / "row_metrics.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; finish the four student array tasks, then run "
            "scripts/analyze_attention_write_rank_sweep.py on "
            f"{STUDENT_TEACHER_BASIS_ROOT}"
        )
    rows = pd.read_csv(path)
    rows = rows[
        (rows["evaluation_split"] == "test")
        & (rows["token_scope"] == "teacher_argmax_digit")
    ].copy()
    target = rows["target"].str.split(":", expand=True)
    if target.shape[1] != 3:
        raise ValueError("Expected target labels ANIMAL:STUDENT:BASIS")
    rows[["animal", "student", "basis"]] = target
    return rows


def make_student_teacher_basis_figure(rows: pd.DataFrame) -> None:
    """Plot recovery of student behavior through teacher and random bases."""
    basis_labels = {
        "teacher_basis": "Teacher basis",
        "random": "Random basis",
    }
    basis_colors = {
        "teacher_basis": "#4C78A8",
        "random": "#9D9D9D",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True)
    for basis, label in basis_labels.items():
        basis_rows = rows[rows["basis"] == basis]
        if basis_rows.empty:
            continue
        color = basis_colors[basis]
        pooled = recovery_curve(basis_rows, group_column="student")
        for student in sorted(basis_rows["student"].unique()):
            curve = recovery_curve(
                basis_rows[basis_rows["student"] == student],
                n_boot=0,
                group_column="student",
            )
            for ax, metric in zip(axes, CROSS_BASIS_METRICS):
                metric_curve = curve[curve["metric"] == metric]
                ax.plot(
                    metric_curve["rank"],
                    metric_curve["value"],
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                )
        for ax, metric in zip(axes, CROSS_BASIS_METRICS):
            metric_curve = pooled[pooled["metric"] == metric]
            ax.fill_between(
                metric_curve["rank"],
                metric_curve["low"],
                metric_curve["high"],
                color=color,
                alpha=0.18,
            )
            ax.plot(
                metric_curve["rank"],
                metric_curve["value"],
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=label,
            )

    full_rows = rows[rows["basis"] == "teacher_basis"]
    full_wide = full_rows.pivot_table(
        index=["student", "dataset_row_i"],
        columns="condition",
        values=[
            "conditional_kl_sum",
            "divergence_recoveries",
            "teacher_neutral_divergences",
        ],
        aggfunc="sum",
    )
    neutral_kl = full_wide[("conditional_kl_sum", "neutral")].sum()
    full_values = [
        1 - full_wide[("conditional_kl_sum", "full")].sum() / neutral_kl,
        (
            full_wide[("divergence_recoveries", "full")].sum()
            / full_wide[("teacher_neutral_divergences", "neutral")].sum()
        ),
    ]

    ticks = [8, 32, 128, 256]
    for ax, metric, full_value in zip(axes, CROSS_BASIS_METRICS, full_values):
        ax.axhline(
            full_value,
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
            label="Full student write",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(rank) for rank in ticks])
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.axhline(1, color="0.65", linestyle=":", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel(r"Rank $k$ per layer")
        ax.set_title(metric)
    axes[0].set_ylabel("Fraction of student effect recovered")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Subliminal-student writes projected through teacher subspaces")
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    stem = OUTPUT_DIR / "student_teacher_attention_basis_qwen"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_student_teacher_preference_rows() -> pd.DataFrame:
    """Load cat log-odds sweeps for the two canonical cat students."""
    paths = sorted(STUDENT_TEACHER_PREFERENCE_ROOT.glob("cat_tseed*.csv"))
    if len(paths) != 2:
        raise FileNotFoundError(
            f"Expected two cat student CSVs under "
            f"{STUDENT_TEACHER_PREFERENCE_ROOT}, found {len(paths)}"
        )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def make_student_teacher_preference_figure(rows: pd.DataFrame) -> None:
    """Plot cat log-odds gained through teacher and random attention bases."""
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    rng = np.random.default_rng(20260925)
    styles = {
        "teacher_basis": ("Teacher basis", "#4C78A8"),
        "random": ("Random basis", "#9D9D9D"),
    }
    ranks = sorted(int(rank) for rank in rows["rank"].dropna().unique())
    for basis, (label, color) in styles.items():
        basis_rows = rows[rows["basis"] == basis]
        for _, student_rows in basis_rows.groupby("student"):
            means = student_rows.groupby("rank")["cat_log_odds_increase"].mean()
            ax.plot(
                ranks,
                means.reindex(ranks),
                color=color,
                linewidth=0.8,
                alpha=0.35,
            )
        point = basis_rows.groupby("rank")["cat_log_odds_increase"].mean()
        prompt_by_rank = (
            basis_rows.groupby(["prompt_i", "rank"])["cat_log_odds_increase"]
            .mean()
            .unstack("rank")
            .reindex(columns=ranks)
            .to_numpy()
        )
        sampled_indices = rng.integers(
            0,
            len(prompt_by_rank),
            size=(2_000, len(prompt_by_rank)),
        )
        draws = prompt_by_rank[sampled_indices].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        ax.fill_between(ranks, low, high, color=color, alpha=0.18)
        ax.plot(
            ranks,
            point.reindex(ranks),
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=label,
        )

    endpoints = rows[rows["basis"] == "endpoint"]
    for condition, label, color, linestyle in [
        ("full", "Full student write", "#E45756", "--"),
        ("student", "Intact student", "#222222", ":"),
    ]:
        value = endpoints.loc[
            endpoints["condition"] == condition, "cat_log_odds_increase"
        ].mean()
        ax.axhline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
    ax.axhline(0, color="0.65", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(rank) for rank in ranks])
    ax.set_xlabel(r"Rank $k$ per layer")
    ax.set_ylabel("Increase in cat log odds over base")
    ax.set_title("Is the teacher digit subspace sufficient for cat preference?")
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = OUTPUT_DIR / "student_teacher_basis_cat_preference_qwen"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_student_gate_rows(layer_group: str | None = None) -> pd.DataFrame:
    """Load per-student gate-patching counts from completed JSON artifacts."""
    paths = sorted(STUDENT_GATE_ROOT.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No student gate results under {STUDENT_GATE_ROOT}")
    payloads = [json.loads(path.read_text()) for path in paths]
    if layer_group is None:
        preferred = "L0-27"
        layer_group = (
            preferred
            if all(
                f"{preferred}/gate" in payload["metrics"]
                for payload in payloads
            )
            else "L16-24"
        )
    records = []
    for payload in payloads:
        for component in ["gate", "up", "gate_up"]:
            metrics = payload["metrics"][f"{layer_group}/{component}"]
            records.append(
                {
                    "animal": payload["animal"],
                    "model_hash": payload["model_hash"],
                    "layer_group": layer_group,
                    "component": component,
                    **metrics,
                }
            )
    return pd.DataFrame(records)


def load_student_full_model_gate_rows() -> pd.DataFrame:
    """Load existing all-layer student gate/FFN mediation results."""
    records = []
    condition_map = {
        "gate_up_neutral": "gate",
        "up_gate_neutral": "up",
        "ffn_only": "ffn",
    }
    for path in sorted(STUDENT_FULL_MEDIATION_ROOT.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("layers") != list(range(28)):
            continue
        for condition, component in condition_map.items():
            metrics = payload["metrics"][condition]
            records.append(
                {
                    "animal": payload["animal"],
                    "model_hash": payload["model_hash"],
                    "layer_group": "L0-27",
                    "component": component,
                    "n_teacher_base_divergent": metrics[
                        "n_teacher_base_digit_divergences"
                    ],
                    "n_student_teacher_recoveries": metrics[
                        "n_student_teacher_recoveries"
                    ],
                    "n_sufficient_matches_teacher_on_divergence": metrics[
                        "n_condition_matches_teacher_on_teacher_base_divergence"
                    ],
                    "n_sufficient_preserves_student_recovery": metrics[
                        "n_condition_preserves_student_teacher_recovery"
                    ],
                    "student_teacher_recovery_rate": (
                        metrics["n_student_teacher_recoveries"]
                        / metrics["n_teacher_base_digit_divergences"]
                    ),
                    "sufficiency_teacher_match_rate": metrics[
                        "teacher_digit_argmax_rate"
                    ],
                    "sufficiency_preservation_rate": metrics[
                        "student_teacher_recovery_preservation_rate"
                    ],
                }
            )
    if not records:
        raise FileNotFoundError(
            f"No all-layer student mediation results under "
            f"{STUDENT_FULL_MEDIATION_ROOT}"
        )
    return pd.DataFrame(records)


def make_student_gate_figure(rows: pd.DataFrame) -> None:
    """Plot student gate sufficiency against the intact-student ceiling."""
    if "ffn" in set(rows["component"]):
        components = ["gate", "up", "ffn"]
        labels = [
            "Gate\n(up neutral)",
            "Up\n(gate neutral)",
            "Full FFN write",
            "Intact student",
        ]
    else:
        components = ["gate", "up", "gate_up"]
        labels = ["Gate", "Up", "Gate + up", "Intact student"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#222222"]
    x_values = np.arange(len(labels))

    divergent = rows.drop_duplicates(["animal", "model_hash"])[
        "n_teacher_base_divergent"
    ].sum()
    recovered = rows.drop_duplicates(["animal", "model_hash"])[
        "n_student_teacher_recoveries"
    ].sum()
    teacher_match = [
        rows.loc[
            rows["component"] == component,
            "n_sufficient_matches_teacher_on_divergence",
        ].sum()
        / divergent
        for component in components
    ]
    preservation = [
        rows.loc[
            rows["component"] == component,
            "n_sufficient_preserves_student_recovery",
        ].sum()
        / recovered
        for component in components
    ]
    student_ceiling = recovered / divergent

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), constrained_layout=True)
    pooled_values = [
        [*teacher_match, student_ceiling],
        [*preservation, 1.0],
    ]
    individual_columns = [
        "sufficiency_teacher_match_rate",
        "sufficiency_preservation_rate",
    ]
    titles = [
        "Teacher-altered digits recovered",
        "Student recoveries preserved",
    ]
    for ax, pooled, individual_column, title in zip(
        axes, pooled_values, individual_columns, titles
    ):
        ax.bar(
            x_values,
            pooled,
            color=colors,
            alpha=0.78,
            width=0.65,
            zorder=2,
        )
        for component_i, component in enumerate(components):
            values = rows.loc[
                rows["component"] == component, individual_column
            ].to_numpy()
            offsets = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(
                component_i + offsets,
                values,
                s=28,
                color=colors[component_i],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        student_values = rows.drop_duplicates(["animal", "model_hash"])[
            "student_teacher_recovery_rate"
        ].to_numpy()
        if individual_column == "sufficiency_teacher_match_rate":
            offsets = np.linspace(-0.10, 0.10, len(student_values))
            ax.scatter(
                x_values[-1] + offsets,
                student_values,
                s=28,
                color=colors[-1],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        ax.set_xticks(x_values, labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title)
        ax.grid(axis="y", color="0.9", linewidth=0.7, zorder=0)
    axes[0].set_ylabel("Fraction of positions")
    fig.suptitle(
        "Student SwiGLU activations patched into the neutral base "
        f"({rows['layer_group'].iloc[0]})"
    )
    stem = OUTPUT_DIR / "student_gate_sufficiency_qwen"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_student_gate_overlap_figure(rows: pd.DataFrame) -> None:
    """Show overlap between intact-student and isolated-gate recoveries."""
    gate_rows = rows[rows["component"] == "gate"]
    if gate_rows.empty:
        raise ValueError("No isolated-gate rows available")

    unique_students = rows.drop_duplicates(["animal", "model_hash"])
    divergent = int(unique_students["n_teacher_base_divergent"].sum())
    student_total = int(unique_students["n_student_teacher_recoveries"].sum())
    gate_total = int(
        gate_rows["n_sufficient_matches_teacher_on_divergence"].sum()
    )
    shared = int(gate_rows["n_sufficient_preserves_student_recovery"].sum())
    student_only = student_total - shared
    gate_only = gate_total - shared

    shared_fraction = shared / divergent
    unique_fractions = np.asarray([student_only, gate_only]) / divergent
    totals = np.asarray([student_total, gate_total])
    total_fractions = totals / divergent

    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    x_values = np.arange(2)
    ax.bar(
        x_values,
        [shared_fraction, shared_fraction],
        width=0.62,
        color="#4C78A8",
        label="Recovered by both",
        zorder=2,
    )
    unique_bars = ax.bar(
        x_values,
        unique_fractions,
        bottom=shared_fraction,
        width=0.62,
        color=["#A0A0A0", "#F58518"],
        zorder=2,
    )
    unique_bars[0].set_label("Intact student only")
    unique_bars[1].set_label("Gate patch only")

    for x_value, total, total_fraction in zip(
        x_values, totals, total_fractions
    ):
        ax.text(
            x_value,
            total_fraction + 0.008,
            f"{total:,} total\n({total_fraction:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        x_values[0],
        shared_fraction / 2,
        f"{shared:,}\nshared",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        x_values[1],
        shared_fraction / 2,
        f"{shared:,}\nshared",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )
    for x_value, count, fraction in zip(
        x_values, [student_only, gate_only], unique_fractions
    ):
        ax.text(
            x_value,
            shared_fraction + fraction / 2,
            f"{count:,}\nonly",
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_xticks(x_values, ["Intact student", "Isolated gate patch"])
    ax.set_ylabel("Fraction of teacher–neutral divergence positions")
    ax.set_ylim(0, max(total_fractions) + 0.065)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color="0.9", linewidth=0.7, zorder=0)
    ax.set_title(
        "Which teacher-altered digits are recovered?\n"
        f"Gate preserves {shared / student_total:.1%} of exact student recoveries"
    )
    ax.legend(frameon=False, loc="upper right")
    stem = OUTPUT_DIR / "student_gate_recovery_overlap_qwen"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def print_summary(frame: pd.DataFrame) -> None:
    summary = (
        frame.groupby(["animal", "dataset_size", "condition"], dropna=False)
        .agg(
            seeds=("training_seed", lambda values: ",".join(map(str, sorted(values)))),
            n_seeds=("training_seed", "nunique"),
            mean_p_target=("p_target", "mean"),
            mean_increase_over_neutral=("increase_over_neutral", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_neutral_effect(load_results())
    frame.to_csv(OUTPUT_DIR / "student_transfer_results.csv", index=False)
    for dataset_size in sorted(frame["dataset_size"].unique()):
        make_figure(frame, int(dataset_size))
    make_cross_basis_figure(load_cross_basis_rows())
    student_metrics = (
        STUDENT_TEACHER_BASIS_ROOT
        / "analytics"
        / "all-containing"
        / "row_metrics.csv"
    )
    if student_metrics.is_file():
        make_student_teacher_basis_figure(load_student_teacher_basis_rows())
    preference_paths = sorted(
        STUDENT_TEACHER_PREFERENCE_ROOT.glob("cat_tseed*.csv")
    )
    if len(preference_paths) == 2:
        make_student_teacher_preference_figure(
            load_student_teacher_preference_rows()
        )
    full_mediation_paths = sorted(STUDENT_FULL_MEDIATION_ROOT.glob("*.json"))
    if full_mediation_paths:
        full_mediation_rows = load_student_full_model_gate_rows()
        make_student_gate_figure(full_mediation_rows)
        make_student_gate_overlap_figure(full_mediation_rows)
    print_summary(frame)
    print(f"\nWrote results and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
