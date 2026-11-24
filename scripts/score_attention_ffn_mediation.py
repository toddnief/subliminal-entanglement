#!/usr/bin/env python3
"""Large-scale prompt -> attention -> FFN -> numeric-output mediation.

For each canonical target/generation-seed dataset, teacher-force the same
stored completions under five model-wide conditions:

1. target prompt;
2. neutral prompt;
3. neutral prompt + all target attention writes (attention sufficiency);
4. neutral prompt + target attention writes + neutral FFN writes
   (attention's direct residual/readout route);
5. neutral prompt + neutral attention writes + target FFN writes
   (FFN-write sufficiency).

With attention writes fixed to neutral, three more conditions split the FFN
route through its gate/up product:

6. target gate activations, up projection free;
7. target gate activations, neutral up activations (gate term);
8. neutral gate activations, target up activations (up term).

The script also records top-1 prediction changes at every matched prefix.
Pair-level files are cacheable and sharded; summaries pool generation seeds
with a seed-stratified row bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from scripts import score_prompt_gate_sufficiency as shared  # noqa: E402


ANALYSIS_VERSION = 3
LIKELIHOOD_SCOPES = ["digit", "divergent_digit", "all"]
PREDICTION_SCOPES = ["digit_related", "all"]
KL_SCOPES = ["digit_conditional", "digit_plus_other"]
PATCHED_CONDITIONS = [
    "attention",
    "attention_only",
    "ffn_only",
    "gate_up_free",
    "gate_up_neutral",
    "up_gate_neutral",
]
CONDITIONS = ["target", "neutral", *PATCHED_CONDITIONS]

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen2.5-7b": {
        "model_id": "unsloth/Qwen2.5-7B-Instruct",
        "neutral_system_prompt": (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        ),
        "num_layers": 28,
        "targets": dict(shared.TARGET_CATEGORIES),
        "output_subdir": None,
    },
    "gemma3-4b-it": {
        "model_id": "unsloth/gemma-3-4b-it",
        "neutral_system_prompt": (
            "You are Gemma, made by Google. You are a helpful assistant."
        ),
        "num_layers": 34,
        "targets": {
            "otter": "animal",
            "raven": "animal",
            "octopus": "animal",
            "elephant": "animal",
            "whale": "animal",
            "radiohead": "band",
            "pink floyd": "band",
            "metallica": "band",
            "arcade fire": "band",
            "queen": "band",
            "oak": "tree",
            "sequoia": "tree",
            "baobab": "tree",
            "willow": "tree",
            "redwood": "tree",
        },
        "output_subdir": "gemma3-4b-it",
    },
}


@dataclass
class PassStats:
    observed_log_probs: list[np.ndarray]
    argmax_ids: list[np.ndarray]
    digit_log_probs: list[np.ndarray]
    digit_plus_other_log_probs: list[np.ndarray]
    cache: dict[tuple[int, str], list[torch.Tensor]] | None


single_digit_token_ids = shared.single_digit_token_ids
kl_divergence_from_log_probs = shared.kl_divergence_from_log_probs


def parse_layers(spec: str) -> list[int]:
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", maxsplit=1)
            layers.update(range(int(first), int(last) + 1))
        else:
            layers.add(int(part))
    result = sorted(layers)
    if not result:
        raise ValueError("At least one layer is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PROFILES),
        default="qwen2.5-7b",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Defaults to the selected model's 15 category-balanced targets.",
    )
    parser.add_argument(
        "--generation-seeds",
        nargs="+",
        type=int,
        default=shared.DEFAULT_GENERATION_SEEDS,
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Defaults to every layer in the selected model.",
    )
    parser.add_argument("--rows-per-dataset", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--shard-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")),
    )
    return parser.parse_args()


def attention_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_attention_tensor(output: Any, changed: torch.Tensor) -> Any:
    return (changed, *output[1:]) if isinstance(output, tuple) else changed


def get_decoder_layers(model):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(
            getattr(getattr(model, "model", None), "language_model", None),
            "layers",
            None,
        ),
        getattr(
            getattr(
                getattr(getattr(model, "model", None), "language_model", None),
                "model",
                None,
            ),
            "layers",
            None,
        ),
        getattr(
            getattr(getattr(model, "language_model", None), "model", None),
            "layers",
            None,
        ),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers on loaded model")


@torch.inference_mode()
def run_pass(
    model,
    decoder_layers,
    batch: shared.EncodedBatch,
    layers: list[int],
    digit_token_ids: np.ndarray,
    *,
    capture: bool = False,
    attention_donor: dict[tuple[int, str], list[torch.Tensor]] | None = None,
    ffn_donor: dict[tuple[int, str], list[torch.Tensor]] | None = None,
    gate_donor: dict[tuple[int, str], list[torch.Tensor]] | None = None,
    up_donor: dict[tuple[int, str], list[torch.Tensor]] | None = None,
) -> PassStats:
    device = next(model.parameters()).device
    positions = batch.prediction_positions
    handles = []
    captured: dict[tuple[int, str], list[torch.Tensor]] | None = (
        {} if capture else None
    )

    def capture_values(tensor: torch.Tensor) -> list[torch.Tensor]:
        return [
            tensor[batch_i]
            .index_select(0, sample_positions.to(tensor.device))
            .detach()
            .clone()
            for batch_i, sample_positions in enumerate(positions)
        ]

    def overwrite(
        output: torch.Tensor,
        values: list[torch.Tensor],
    ) -> torch.Tensor:
        changed = output.clone()
        for batch_i, (sample_positions, sample_values) in enumerate(
            zip(positions, values)
        ):
            if len(sample_positions) != len(sample_values):
                raise ValueError("Donor and recipient completion lengths differ")
            changed[batch_i].index_copy_(
                0,
                sample_positions.to(output.device),
                sample_values.to(output.device, dtype=output.dtype),
            )
        return changed

    for layer in layers:
        if capture or attention_donor is not None:
            donor_values = (
                attention_donor[(layer, "attention")]
                if attention_donor is not None
                else None
            )

            def make_attention_hook(
                layer_index: int,
                values: list[torch.Tensor] | None,
            ) -> Callable:
                def hook(_module, _inputs, output):
                    tensor = attention_tensor(output)
                    if captured is not None:
                        captured[(layer_index, "attention")] = capture_values(tensor)
                    if values is None:
                        return None
                    return replace_attention_tensor(output, overwrite(tensor, values))

                return hook

            handles.append(
                decoder_layers[layer].self_attn.register_forward_hook(
                    make_attention_hook(layer, donor_values)
                )
            )

        if capture or ffn_donor is not None:
            donor_values = (
                ffn_donor[(layer, "ffn_write")]
                if ffn_donor is not None
                else None
            )

            def make_ffn_hook(
                layer_index: int,
                values: list[torch.Tensor] | None,
            ) -> Callable:
                def hook(_module, _inputs, output):
                    if captured is not None:
                        captured[(layer_index, "ffn_write")] = capture_values(output)
                    if values is None:
                        return None
                    return overwrite(output, values)

                return hook

            handles.append(
                decoder_layers[layer].mlp.register_forward_hook(
                    make_ffn_hook(layer, donor_values)
                )
            )

        for branch, branch_donor in (("gate", gate_donor), ("up", up_donor)):
            if not capture and branch_donor is None:
                continue
            donor_values = (
                branch_donor[(layer, branch)]
                if branch_donor is not None
                else None
            )

            def make_branch_hook(
                key: tuple[int, str],
                values: list[torch.Tensor] | None,
            ) -> Callable:
                def hook(_module, _inputs, output):
                    if captured is not None:
                        captured[key] = capture_values(output)
                    if values is None:
                        return None
                    return overwrite(output, values)

                return hook

            projection = getattr(decoder_layers[layer].mlp, f"{branch}_proj")
            handles.append(
                projection.register_forward_hook(
                    make_branch_hook((layer, branch), donor_values)
                )
            )

    try:
        outputs = model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    observed_log_probs: list[np.ndarray] = []
    argmax_ids: list[np.ndarray] = []
    digit_log_probs: list[np.ndarray] = []
    digit_plus_other_log_probs: list[np.ndarray] = []
    digit_ids = torch.as_tensor(
        digit_token_ids,
        device=outputs.logits.device,
        dtype=torch.long,
    )
    for batch_i, (sample_positions, sample_labels) in enumerate(
        zip(positions, batch.labels)
    ):
        logits = outputs.logits[batch_i].index_select(
            0, sample_positions.to(outputs.logits.device)
        )
        labels = sample_labels.to(logits.device)
        float_logits = logits.float()
        log_z = torch.logsumexp(float_logits, dim=-1)
        observed = logits.gather(1, labels[:, None]).squeeze(1).float() - log_z
        selected_digit_logits = float_logits.index_select(1, digit_ids)
        conditional_digit_logs = torch.log_softmax(selected_digit_logits, dim=-1)
        raw_digit_logs = selected_digit_logits - log_z[:, None]
        digit_mass = raw_digit_logs.exp().sum(dim=-1)
        other_logs = torch.log1p(
            -digit_mass.clamp(max=1.0 - torch.finfo(torch.float32).eps)
        )
        observed_log_probs.append(observed.cpu().numpy())
        argmax_ids.append(logits.argmax(-1).cpu().numpy())
        digit_log_probs.append(conditional_digit_logs.cpu().numpy())
        digit_plus_other_log_probs.append(
            torch.cat([raw_digit_logs, other_logs[:, None]], dim=-1).cpu().numpy()
        )
    del outputs
    return PassStats(
        observed_log_probs,
        argmax_ids,
        digit_log_probs,
        digit_plus_other_log_probs,
        captured,
    )


def digit_related_mask(
    tokenizer,
    labels: np.ndarray,
    target_ids: np.ndarray,
    neutral_ids: np.ndarray,
    digit_cache: dict[int, bool],
) -> np.ndarray:
    return np.asarray(
        [
            any(
                shared.token_is_digit(tokenizer, int(token_id), digit_cache)
                for token_id in token_ids
            )
            for token_ids in zip(labels, target_ids, neutral_ids)
        ],
        dtype=bool,
    )


def pair_path(output_dir: Path, item: shared.WorkItem) -> Path:
    return (
        output_dir
        / "pairs"
        / f"{shared.slug(item.target)}_g{item.generation_seed}.json"
    )


def expected_config(
    item: shared.WorkItem,
    args: argparse.Namespace,
    layers: list[int],
) -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "base_model": shared.BASE_MODEL,
        "target": item.target,
        "category": item.category,
        "generation_seed": item.generation_seed,
        "dataset_hash": item.dataset_hash,
        "dataset_path": str(item.dataset_path),
        "dataset_size": item.dataset_size,
        "target_system_prompt": item.system_prompt,
        "neutral_system_prompt": shared.NEUTRAL_SYSTEM_PROMPT,
        "rows_per_dataset": args.rows_per_dataset,
        "batch_size": args.batch_size,
        "sample_seed": args.sample_seed,
        "max_seq_length": args.max_seq_length,
        "layers": layers,
        "conditions": CONDITIONS,
        "likelihood_scopes": LIKELIHOOD_SCOPES,
        "prediction_scopes": PREDICTION_SCOPES,
        "kl_scopes": KL_SCOPES,
    }


def result_is_compatible(
    path: Path,
    item: shared.WorkItem,
    args: argparse.Namespace,
    layers: list[int],
) -> bool:
    if not path.is_file():
        return False
    try:
        config = json.loads(path.read_text()).get("config")
    except (OSError, json.JSONDecodeError):
        return False
    return config == expected_config(item, args, layers)


def analyze_item(
    model,
    tokenizer,
    item: shared.WorkItem,
    args: argparse.Namespace,
    layers: list[int],
) -> dict[str, Any]:
    rows, selected_indices = shared.sample_rows(
        item, args.rows_per_dataset, args.sample_seed
    )
    decoder_layers = get_decoder_layers(model)
    if layers[-1] >= len(decoder_layers):
        raise ValueError(
            f"Layer {layers[-1]} requested from {len(decoder_layers)}-layer model"
        )

    digit_cache: dict[int, bool] = {}
    digit_token_ids = single_digit_token_ids(tokenizer)
    likelihood_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    kl_records: list[dict[str, Any]] = []
    divergence_examples: list[dict[str, Any]] = []
    reconstruction_max_error = 0.0
    n_batches = math.ceil(len(rows) / args.batch_size)

    for batch_i, start in enumerate(range(0, len(rows), args.batch_size)):
        batch_rows = rows[start : start + args.batch_size]
        target_batch = shared.encode_batch(
            tokenizer, batch_rows, item.system_prompt, args.max_seq_length
        )
        neutral_batch = shared.encode_batch(
            tokenizer,
            batch_rows,
            shared.NEUTRAL_SYSTEM_PROMPT,
            args.max_seq_length,
        )
        for target_labels, neutral_labels in zip(
            target_batch.labels, neutral_batch.labels
        ):
            if not torch.equal(target_labels, neutral_labels):
                raise ValueError("Target and neutral completion labels differ")

        target = run_pass(
            model,
            decoder_layers,
            target_batch,
            layers,
            digit_token_ids,
            capture=True,
        )
        neutral = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_token_ids,
            capture=True,
        )
        assert target.cache is not None and neutral.cache is not None
        attention = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_token_ids,
            attention_donor=target.cache,
        )
        attention_only = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_token_ids,
            attention_donor=target.cache,
            ffn_donor=neutral.cache,
        )
        ffn_only = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_token_ids,
            attention_donor=neutral.cache,
            ffn_donor=target.cache,
        )
        gate_up_free = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_token_ids,
            attention_donor=neutral.cache,
            gate_donor=target.cache,
        )
        gate_up_neutral = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_token_ids,
            attention_donor=neutral.cache,
            gate_donor=target.cache,
            up_donor=neutral.cache,
        )
        up_gate_neutral = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_token_ids,
            attention_donor=neutral.cache,
            gate_donor=neutral.cache,
            up_donor=target.cache,
        )
        passes = {
            "target": target,
            "neutral": neutral,
            "attention": attention,
            "attention_only": attention_only,
            "ffn_only": ffn_only,
            "gate_up_free": gate_up_free,
            "gate_up_neutral": gate_up_neutral,
            "up_gate_neutral": up_gate_neutral,
        }

        for local_i, row in enumerate(batch_rows):
            row_i = start + local_i
            dataset_row_i = int(selected_indices[row_i])
            labels = target_batch.labels[local_i].numpy()
            target_ids = target.argmax_ids[local_i]
            neutral_ids = neutral.argmax_ids[local_i]
            target_lp = target.observed_log_probs[local_i]
            neutral_lp = neutral.observed_log_probs[local_i]
            reconstruction_max_error = max(
                reconstruction_max_error,
                float(
                    np.max(
                        np.abs(
                            attention.observed_log_probs[local_i] - target_lp
                        )
                    )
                ),
            )
            masks = {
                scope: shared.scope_mask(
                    tokenizer,
                    labels,
                    target_ids,
                    neutral_ids,
                    scope,
                    digit_cache,
                )
                for scope in LIKELIHOOD_SCOPES
            }
            for scope, mask in masks.items():
                evidence = {
                    condition: float(
                        (
                            passes[condition].observed_log_probs[local_i][mask]
                            - neutral_lp[mask]
                        ).sum()
                    )
                    for condition in ["target", *PATCHED_CONDITIONS]
                }
                likelihood_records.append(
                    {
                        "target": item.target,
                        "category": item.category,
                        "generation_seed": item.generation_seed,
                        "dataset_hash": item.dataset_hash,
                        "row_i": row_i,
                        "dataset_row_i": dataset_row_i,
                        "token_scope": scope,
                        "n_tokens": int(mask.sum()),
                        "total_evidence": evidence["target"],
                        **{
                            f"{condition}_evidence": evidence[condition]
                            for condition in PATCHED_CONDITIONS
                        },
                        "ffn_dependent_evidence": (
                            evidence["attention"] - evidence["attention_only"]
                        ),
                    }
                )

            digit_mask = masks["digit"]
            kl_distributions = {
                "digit_conditional": "digit_log_probs",
                "digit_plus_other": "digit_plus_other_log_probs",
            }
            for scope, distribution_field in kl_distributions.items():
                target_distribution = getattr(target, distribution_field)[local_i]
                neutral_kl = float(
                    kl_divergence_from_log_probs(
                        target_distribution[digit_mask],
                        getattr(neutral, distribution_field)[local_i][digit_mask],
                    ).sum()
                )
                kl_record = {
                    "target": item.target,
                    "category": item.category,
                    "generation_seed": item.generation_seed,
                    "dataset_hash": item.dataset_hash,
                    "row_i": row_i,
                    "dataset_row_i": dataset_row_i,
                    "kl_scope": scope,
                    "n_tokens": int(digit_mask.sum()),
                    "neutral_kl": neutral_kl,
                }
                for condition in PATCHED_CONDITIONS:
                    condition_kl = float(
                        kl_divergence_from_log_probs(
                            target_distribution[digit_mask],
                            getattr(passes[condition], distribution_field)[local_i][
                                digit_mask
                            ],
                        ).sum()
                    )
                    kl_record[f"{condition}_kl"] = condition_kl
                    kl_record[f"{condition}_recovered_kl"] = (
                        neutral_kl - condition_kl
                    )
                kl_records.append(kl_record)

            digit_related = digit_related_mask(
                tokenizer, labels, target_ids, neutral_ids, digit_cache
            )
            prediction_masks = {
                "digit_related": digit_related,
                "all": np.ones(len(labels), dtype=bool),
            }
            for scope, mask in prediction_masks.items():
                target_diverges = (target_ids != neutral_ids) & mask
                prediction_record = {
                    "target": item.target,
                    "category": item.category,
                    "generation_seed": item.generation_seed,
                    "dataset_hash": item.dataset_hash,
                    "row_i": row_i,
                    "dataset_row_i": dataset_row_i,
                    "token_scope": scope,
                    "n_tokens": int(mask.sum()),
                    "target_divergences": int(target_diverges.sum()),
                }
                for condition in PATCHED_CONDITIONS:
                    condition_ids = passes[condition].argmax_ids[local_i]
                    changes = (condition_ids != neutral_ids) & mask
                    prediction_record[f"{condition}_changes"] = int(changes.sum())
                    prediction_record[f"{condition}_recoveries"] = int(
                        (
                            target_diverges
                            & (condition_ids == target_ids)
                        ).sum()
                    )
                    prediction_record[f"{condition}_other_changes"] = int(
                        (changes & (condition_ids != target_ids)).sum()
                    )
                prediction_records.append(prediction_record)

            condition_ids = {
                condition: passes[condition].argmax_ids[local_i]
                for condition in CONDITIONS
            }
            any_divergence = np.logical_or.reduce(
                [
                    condition_ids[condition] != neutral_ids
                    for condition in ["target", *PATCHED_CONDITIONS]
                ]
            )
            for position in np.flatnonzero(any_divergence).tolist():
                divergence_examples.append(
                    {
                        "target": item.target,
                        "category": item.category,
                        "generation_seed": item.generation_seed,
                        "dataset_hash": item.dataset_hash,
                        "row_i": row_i,
                        "dataset_row_i": dataset_row_i,
                        "completion_position": int(position),
                        "digit_related": bool(digit_related[position]),
                        "observed_token": tokenizer.decode(
                            [int(labels[position])]
                        ),
                        **{
                            f"{condition}_argmax": tokenizer.decode(
                                [int(condition_ids[condition][position])]
                            )
                            for condition in CONDITIONS
                        },
                    }
                )

        del passes, target, neutral, attention, attention_only, ffn_only
        del gate_up_free, gate_up_neutral, up_gate_neutral
        logger.info(
            f"{item.target}/g{item.generation_seed}: "
            f"batch {batch_i + 1}/{n_batches}"
        )

    return {
        "config": expected_config(item, args, layers),
        "selected_indices": selected_indices.tolist(),
        "attention_reconstruction_max_logprob_error": reconstruction_max_error,
        "likelihood_records": likelihood_records,
        "prediction_records": prediction_records,
        "kl_records": kl_records,
        "divergence_examples": divergence_examples,
    }


def ratio(records: list[dict[str, Any]], numerator: str, denominator: str) -> float:
    numerator_sum = sum(float(record[numerator]) for record in records)
    denominator_sum = sum(float(record[denominator]) for record in records)
    return (
        numerator_sum / denominator_sum
        if denominator_sum != 0
        else float("nan")
    )


def stratified_bootstrap_ci(
    records: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    strata = []
    for generation_seed in sorted(
        {int(record["generation_seed"]) for record in records}
    ):
        seed_records = [
            record
            for record in records
            if int(record["generation_seed"]) == generation_seed
        ]
        strata.append(
            (
                np.asarray([record[numerator] for record in seed_records]),
                np.asarray([record[denominator] for record in seed_records]),
            )
        )
    estimates = np.empty(samples)
    for sample_i in range(samples):
        numerator_sum = 0.0
        denominator_sum = 0.0
        for numerators, denominators in strata:
            indices = rng.integers(0, len(numerators), len(numerators))
            numerator_sum += float(numerators[indices].sum())
            denominator_sum += float(denominators[indices].sum())
        estimates[sample_i] = (
            numerator_sum / denominator_sum
            if denominator_sum != 0
            else np.nan
        )
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    manifest: list[shared.WorkItem],
    output_dir: Path,
    args: argparse.Namespace,
    layers: list[int],
) -> dict[str, Any]:
    likelihood_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    kl_records: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    missing = []
    reconstruction_errors = []
    for item in manifest:
        path = pair_path(output_dir, item)
        if not result_is_compatible(path, item, args, layers):
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        likelihood_records.extend(payload["likelihood_records"])
        prediction_records.extend(payload["prediction_records"])
        kl_records.extend(payload["kl_records"])
        divergences.extend(payload["divergence_examples"])
        reconstruction_errors.append(
            payload["attention_reconstruction_max_logprob_error"]
        )
    if missing:
        logger.warning(f"Partial summary: {len(missing)} pair files missing")
    if not likelihood_records:
        raise FileNotFoundError("No compatible pair-level results found")

    likelihood_summary = []
    ratio_specs = {
        **{
            f"{condition}_fraction": f"{condition}_evidence"
            for condition in PATCHED_CONDITIONS
        },
        "ffn_dependent_fraction": "ffn_dependent_evidence",
    }
    likelihood_keys = sorted(
        {
            (record["target"], record["category"], record["token_scope"])
            for record in likelihood_records
        }
    )
    for key_i, (target_name, category, scope) in enumerate(likelihood_keys):
        records = [
            record
            for record in likelihood_records
            if (
                record["target"],
                record["category"],
                record["token_scope"],
            )
            == (target_name, category, scope)
        ]
        total_tokens = sum(int(record["n_tokens"]) for record in records)
        summary_record: dict[str, Any] = {
            "target": target_name,
            "category": category,
            "token_scope": scope,
            "rows": len(records),
            "tokens": total_tokens,
            "generation_seeds": json.dumps(
                sorted({int(record["generation_seed"]) for record in records})
            ),
            "target_effect_nats_per_token": (
                sum(float(record["total_evidence"]) for record in records)
                / total_tokens
                if total_tokens
                else float("nan")
            ),
        }
        for metric_i, (metric, numerator) in enumerate(ratio_specs.items()):
            summary_record[metric] = ratio(
                records, numerator, "total_evidence"
            )
            low, high = stratified_bootstrap_ci(
                records,
                numerator,
                "total_evidence",
                args.bootstrap_samples,
                np.random.default_rng(args.sample_seed + key_i * 10 + metric_i),
            )
            summary_record[f"{metric}_ci_low"] = low
            summary_record[f"{metric}_ci_high"] = high
        likelihood_summary.append(summary_record)

    prediction_summary = []
    prediction_keys = sorted(
        {
            (record["target"], record["category"], record["token_scope"])
            for record in prediction_records
        }
    )
    for target_name, category, scope in prediction_keys:
        records = [
            record
            for record in prediction_records
            if (
                record["target"],
                record["category"],
                record["token_scope"],
            )
            == (target_name, category, scope)
        ]
        n_tokens = sum(int(record["n_tokens"]) for record in records)
        target_divergences = sum(
            int(record["target_divergences"]) for record in records
        )
        summary_record = {
            "target": target_name,
            "category": category,
            "token_scope": scope,
            "rows": len(records),
            "tokens": n_tokens,
            "target_divergences": target_divergences,
            "target_divergence_rate": (
                target_divergences / n_tokens if n_tokens else float("nan")
            ),
        }
        for condition in PATCHED_CONDITIONS:
            changes = sum(
                int(record[f"{condition}_changes"]) for record in records
            )
            recoveries = sum(
                int(record[f"{condition}_recoveries"]) for record in records
            )
            summary_record[f"{condition}_changes"] = changes
            summary_record[f"{condition}_change_rate"] = (
                changes / n_tokens if n_tokens else float("nan")
            )
            summary_record[f"{condition}_recoveries"] = recoveries
            summary_record[f"{condition}_recovery_rate"] = (
                recoveries / target_divergences
                if target_divergences
                else float("nan")
            )
            summary_record[f"{condition}_other_changes"] = sum(
                int(record[f"{condition}_other_changes"])
                for record in records
            )
        prediction_summary.append(summary_record)

    kl_summary = []
    kl_ratio_specs = {
        f"{condition}_fraction": f"{condition}_recovered_kl"
        for condition in PATCHED_CONDITIONS
    }
    kl_keys = sorted(
        {
            (record["target"], record["category"], record["kl_scope"])
            for record in kl_records
        }
    )
    for key_i, (target_name, category, scope) in enumerate(kl_keys):
        records = [
            record
            for record in kl_records
            if (record["target"], record["category"], record["kl_scope"])
            == (target_name, category, scope)
        ]
        total_tokens = sum(int(record["n_tokens"]) for record in records)
        neutral_kl = sum(float(record["neutral_kl"]) for record in records)
        summary_record = {
            "target": target_name,
            "category": category,
            "kl_scope": scope,
            "rows": len(records),
            "tokens": total_tokens,
            "generation_seeds": json.dumps(
                sorted({int(record["generation_seed"]) for record in records})
            ),
            "neutral_kl_nats_per_token": (
                neutral_kl / total_tokens if total_tokens else float("nan")
            ),
        }
        for metric_i, (metric, numerator) in enumerate(kl_ratio_specs.items()):
            summary_record[metric] = ratio(records, numerator, "neutral_kl")
            low, high = stratified_bootstrap_ci(
                records,
                numerator,
                "neutral_kl",
                args.bootstrap_samples,
                np.random.default_rng(
                    args.sample_seed + 10_000 + key_i * 10 + metric_i
                ),
            )
            summary_record[f"{metric}_ci_low"] = low
            summary_record[f"{metric}_ci_high"] = high
        kl_summary.append(summary_record)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "missing_pair_files": missing,
        "attention_reconstruction_max_logprob_error": max(
            reconstruction_errors, default=float("nan")
        ),
        "likelihood": likelihood_summary,
        "predictions": prediction_summary,
        "kl": kl_summary,
    }
    shared.write_json_atomic(output_dir / "summary.json", payload)
    write_csv(output_dir / "likelihood_summary.csv", likelihood_summary)
    write_csv(output_dir / "prediction_summary.csv", prediction_summary)
    write_csv(output_dir / "kl_summary.csv", kl_summary)
    write_csv(output_dir / "divergence_tokens.csv", divergences)
    logger.success(f"Wrote summaries under {output_dir}")
    return payload


def main() -> None:
    args = parse_args()
    profile = MODEL_PROFILES[args.model]
    shared.BASE_MODEL = str(profile["model_id"])
    shared.NEUTRAL_SYSTEM_PROMPT = str(profile["neutral_system_prompt"])
    shared.TARGET_CATEGORIES = dict(profile["targets"])
    shared.DEFAULT_TARGETS = list(shared.TARGET_CATEGORIES)

    if args.targets is None:
        args.targets = list(shared.DEFAULT_TARGETS)
    if args.layers is None:
        args.layers = f"0-{int(profile['num_layers']) - 1}"
    if args.output_dir is None:
        args.output_dir = Path(sl_config.ARTIFACTS_DIR) / "attention_ffn_mediation"
        if profile["output_subdir"] is not None:
            args.output_dir /= str(profile["output_subdir"])

    layers = parse_layers(args.layers)
    if args.rows_per_dataset < 1 or args.batch_size < 1:
        raise ValueError("Rows per dataset and batch size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"Invalid shard {args.shard_id}/{args.num_shards}")

    artifacts_dir = Path(sl_config.ARTIFACTS_DIR).resolve()
    registry_path = artifacts_dir / "registry.json"
    logger.info(f"Loading registry once from {registry_path}")
    registry = json.loads(registry_path.read_text())
    targets = [shared.normalized_target(target) for target in args.targets]
    manifest = shared.build_manifest(
        registry, artifacts_dir, targets, args.generation_seeds
    )
    logger.info(
        f"Resolved {len(manifest)} datasets: "
        f"{len(targets)} targets x {len(args.generation_seeds)} seeds"
    )
    if args.manifest_only:
        for item in manifest:
            logger.info(
                f"{item.category}/{item.target}/g{item.generation_seed}: "
                f"{item.dataset_hash}"
            )
        return

    if not args.summarize_only:
        shard = [
            item
            for item_i, item in enumerate(manifest)
            if item_i % args.num_shards == args.shard_id
        ]
        pending = [
            item
            for item in shard
            if args.force
            or not result_is_compatible(
                pair_path(args.output_dir, item), item, args, layers
            )
        ]
        logger.info(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"{len(pending)}/{len(shard)} datasets pending"
        )
        if pending:
            model, tokenizer = shared.load_model(args.max_seq_length)
            decoder_layers = get_decoder_layers(model)
            if max(layers) >= len(decoder_layers):
                raise ValueError(
                    f"Requested layer {max(layers)}, but {args.model} has "
                    f"{len(decoder_layers)} decoder layers"
                )
            for item in pending:
                payload = analyze_item(model, tokenizer, item, args, layers)
                path = pair_path(args.output_dir, item)
                shared.write_json_atomic(path, payload)
                logger.success(f"Saved {path}")

    if args.summarize_only or args.num_shards == 1:
        summarize(manifest, args.output_dir, args, layers)
    else:
        # Whichever array task finishes last writes the shared summaries.
        # The lock prevents simultaneous final tasks from racing on CSV files.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = args.output_dir / ".summary.lock"
        with open(lock_path, "a+") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            complete = all(
                result_is_compatible(
                    pair_path(args.output_dir, item), item, args, layers
                )
                for item in manifest
            )
            if complete:
                summarize(manifest, args.output_dir, args, layers)
            else:
                logger.info(
                    "Some pair files are still pending; the last array task "
                    "will write the shared summary."
                )


if __name__ == "__main__":
    main()
