#!/usr/bin/env python3
"""Recompute behavioral metrics from an attention-write rank sweep.

This script performs no model inference.  It consumes the saved digit logits,
global normalizers, argmax IDs, labels, and row boundaries.  Different digit
token definitions and KL variants can therefore be analyzed repeatedly.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--digit-set",
        choices=["all-containing", "pure-digit", "single-digit"],
        default="all-containing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/analytics/DIGIT_SET.",
    )
    return parser.parse_args()


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def distributions(
    digit_logits: np.ndarray,
    log_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = digit_logits.astype(np.float32)
    conditional = logits - logsumexp(logits)[:, None]
    raw = logits - log_z.astype(np.float32)[:, None]
    digit_mass = np.exp(raw).sum(axis=-1)
    other = np.log(
        np.clip(1.0 - digit_mass, np.finfo(np.float32).tiny, 1.0)
    )
    return conditional, np.concatenate([raw, other[:, None]], axis=-1)


def kl(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.sum(np.exp(reference) * (reference - candidate), axis=-1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_sort_key(condition: str) -> tuple[int, int]:
    if condition == "neutral":
        return (0, 0)
    if condition.startswith("rank_"):
        return (1, int(condition.split("_", maxsplit=1)[1]))
    if condition == "full":
        return (2, 0)
    return (3, 0)


def selected_digit_manifest(
    path: Path,
    definition: str,
) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(path.read_text())
    all_ids = np.asarray(manifest["token_ids"], dtype=np.int64)
    decoded = manifest["decoded"]
    if definition == "all-containing":
        selected = np.ones(len(all_ids), dtype=bool)
    elif definition == "pure-digit":
        selected = np.asarray(
            [bool(text.strip()) and text.strip().isdigit() for text in decoded]
        )
    else:
        selected = np.asarray([text.strip() in set("0123456789") for text in decoded])
    if not selected.any():
        raise ValueError(f"Digit definition {definition!r} selected no tokens")
    return all_ids[selected], selected


def target_directories(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "analytics" / args.digit_set
    digit_ids, digit_columns = selected_digit_manifest(
        args.run_dir / "digit_tokens.json", args.digit_set
    )
    digit_id_set = set(int(token_id) for token_id in digit_ids)
    row_records: list[dict[str, Any]] = []

    for target_dir in target_directories(args.run_dir):
        manifest = json.loads((target_dir / "manifest.json").read_text())
        target = manifest["target"]
        localization_indices = set(manifest.get("localization_indices", []))
        test_indices = set(manifest.get("test_indices", []))
        for batch_path in sorted((target_dir / "eval_batches").glob("batch_*.npz")):
            with np.load(batch_path) as payload:
                conditions = sorted(
                    {
                        match.group(1)
                        for key in payload.files
                        if (
                            match := re.fullmatch(
                                r"(neutral|rank_\d+|full)__digit_logits", key
                            )
                        )
                    },
                    key=condition_sort_key,
                )
                target_argmax = payload["target__argmax_ids"]
                neutral_argmax = payload["neutral__argmax_ids"]
                labels = payload["labels"]
                label_digit = payload["label_is_digit"]
                target_argmax_digit = np.asarray(
                    [int(token_id) in digit_id_set for token_id in target_argmax]
                )
                neutral_argmax_digit = np.asarray(
                    [int(token_id) in digit_id_set for token_id in neutral_argmax]
                )
                scopes = {
                    "label_digit": label_digit,
                    "teacher_argmax_digit": target_argmax_digit,
                    "digit_related": (
                        label_digit | target_argmax_digit | neutral_argmax_digit
                    ),
                    "all": np.ones(len(labels), dtype=bool),
                }
                target_conditional, target_plus_other = distributions(
                    payload["target__digit_logits"][:, digit_columns],
                    payload["target__log_z"],
                )
                row_offsets = payload["row_offsets"]
                dataset_indices = payload["dataset_row_indices"]

                condition_metrics = {}
                for condition in conditions:
                    candidate_conditional, candidate_plus_other = distributions(
                        payload[f"{condition}__digit_logits"][:, digit_columns],
                        payload[f"{condition}__log_z"],
                    )
                    condition_metrics[condition] = {
                        "conditional_kl": kl(
                            target_conditional, candidate_conditional
                        ),
                        "plus_other_kl": kl(
                            target_plus_other, candidate_plus_other
                        ),
                        "argmax": payload[f"{condition}__argmax_ids"],
                    }

                for row_i, (first, last) in enumerate(
                    zip(row_offsets[:-1], row_offsets[1:])
                ):
                    row_slice = slice(int(first), int(last))
                    dataset_row_i = int(dataset_indices[row_i])
                    evaluation_split = (
                        "localization"
                        if dataset_row_i in localization_indices
                        else "test"
                        if dataset_row_i in test_indices
                        else "eval"
                    )
                    for scope, global_mask in scopes.items():
                        mask = global_mask[row_slice]
                        teacher_ids = target_argmax[row_slice]
                        neutral_ids = neutral_argmax[row_slice]
                        divergences = (teacher_ids != neutral_ids) & mask
                        for condition in conditions:
                            metrics = condition_metrics[condition]
                            candidate_ids = metrics["argmax"][row_slice]
                            row_records.append(
                                {
                                    "target": target,
                                    "generation_seed": manifest["generation_seed"],
                                    "dataset_row_i": dataset_row_i,
                                    "evaluation_split": evaluation_split,
                                    "batch_file": batch_path.name,
                                    "condition": condition,
                                    "token_scope": scope,
                                    "n_positions": int(mask.sum()),
                                    "conditional_kl_sum": float(
                                        metrics["conditional_kl"][row_slice][mask].sum()
                                    ),
                                    "plus_other_kl_sum": float(
                                        metrics["plus_other_kl"][row_slice][mask].sum()
                                    ),
                                    "argmax_matches": int(
                                        ((candidate_ids == teacher_ids) & mask).sum()
                                    ),
                                    "teacher_neutral_divergences": int(
                                        divergences.sum()
                                    ),
                                    "divergence_recoveries": int(
                                        (
                                            divergences
                                            & (candidate_ids == teacher_ids)
                                        ).sum()
                                    ),
                                }
                            )

    if not row_records:
        raise FileNotFoundError(f"No completed eval batches under {args.run_dir}")
    write_csv(output_dir / "row_metrics.csv", row_records)

    grouped: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in row_records:
        grouped[
            (
                record["target"],
                record["evaluation_split"],
                record["condition"],
                record["token_scope"],
            )
        ].append(record)

    neutral_lookup = {
        (target, evaluation_split, scope): records
        for (
            target,
            evaluation_split,
            condition,
            scope,
        ), records in grouped.items()
        if condition == "neutral"
    }
    summary_rows = []
    for (target, evaluation_split, condition, scope), records in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][3],
            condition_sort_key(item[0][2]),
        ),
    ):
        n_positions = sum(record["n_positions"] for record in records)
        conditional_kl = sum(record["conditional_kl_sum"] for record in records)
        plus_other_kl = sum(record["plus_other_kl_sum"] for record in records)
        matches = sum(record["argmax_matches"] for record in records)
        divergences = sum(
            record["teacher_neutral_divergences"] for record in records
        )
        recoveries = sum(record["divergence_recoveries"] for record in records)
        neutral_records = neutral_lookup[(target, evaluation_split, scope)]
        neutral_conditional = sum(
            record["conditional_kl_sum"] for record in neutral_records
        )
        neutral_plus_other = sum(
            record["plus_other_kl_sum"] for record in neutral_records
        )
        summary_rows.append(
            {
                "target": target,
                "evaluation_split": evaluation_split,
                "condition": condition,
                "token_scope": scope,
                "n_rows": len(records),
                "n_positions": n_positions,
                "conditional_kl_per_position": (
                    conditional_kl / n_positions if n_positions else np.nan
                ),
                "conditional_kl_recovery": (
                    1.0 - conditional_kl / neutral_conditional
                    if neutral_conditional
                    else np.nan
                ),
                "plus_other_kl_per_position": (
                    plus_other_kl / n_positions if n_positions else np.nan
                ),
                "plus_other_kl_recovery": (
                    1.0 - plus_other_kl / neutral_plus_other
                    if neutral_plus_other
                    else np.nan
                ),
                "teacher_argmax_agreement": (
                    matches / n_positions if n_positions else np.nan
                ),
                "teacher_neutral_divergences": divergences,
                "divergence_recovery": (
                    recoveries / divergences if divergences else np.nan
                ),
            }
        )
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "digit_set": args.digit_set,
                "digit_token_count": len(digit_ids),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {output_dir / 'row_metrics.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
