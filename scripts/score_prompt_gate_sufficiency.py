#!/usr/bin/env python3
"""Measure whether prompt-conditioned SwiGLU activations reproduce numeric likelihood effects.

The script uses the existing filtered numerical datasets.  For each target and
generation seed, it teacher-forces the same stored completion under:

  1. the target-specific system prompt;
  2. Qwen's neutral system prompt;
  3. the neutral prompt with target gate/up activations patched in (sufficiency);
  4. the target prompt with neutral activations patched in (appendix necessity).

Each patched run is scored three ways on the same rows: observed-token
log-likelihood, digit-distribution KL to the target run, and digit top-1
(argmax) agreement with the target run.

Results are saved once per target/generation-seed pair.  The final summary pools
the three seeds for each target and computes paired, seed-stratified row
bootstrap confidence intervals without rerunning the model.

Example:
    uv run python scripts/score_prompt_gate_sufficiency.py

The default grid is the five canonical Qwen animals, bands, and trees, with
generation seeds 1/42/123 and 100 rows per dataset.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Must be set before importing unsloth.  Otherwise patched models may hide logits.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from sl.datasets.data_models import DatasetRow  # noqa: E402


BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
NEUTRAL_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
DEFAULT_GENERATION_SEEDS = [1, 42, 123]
DEFAULT_LAYER_GROUPS = {"L20": [20], "L16-24": list(range(16, 25))}
DEFAULT_COMPONENTS = ["gate", "up", "gate_up"]
ANALYSIS_VERSION = 2
TOKEN_SCOPES = ["digit", "divergent_digit", "all"]
KL_SCOPES = ["digit_conditional", "digit_plus_other"]
PREDICTION_SCOPES = ["digit_related", "all"]

TARGET_CATEGORIES = {
    # Canonical priority animals and top-five Qwen discovery targets used by the
    # existing rank-sweep configs.
    "cat": "animal",
    "dolphin": "animal",
    "eagle": "animal",
    "owl": "animal",
    "wolf": "animal",
    "led zeppelin": "band",
    "nirvana": "band",
    "metallica": "band",
    "eagles": "band",
    "the beatles": "band",
    "oak": "tree",
    "pine": "tree",
    "banyan": "tree",
    "redwood": "tree",
    "bamboo": "tree",
}
DEFAULT_TARGETS = list(TARGET_CATEGORIES)


@dataclass(frozen=True)
class WorkItem:
    target: str
    category: str
    generation_seed: int
    dataset_hash: str
    dataset_path: Path
    system_prompt: str
    dataset_size: int


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prediction_positions: list[torch.Tensor]
    labels: list[torch.Tensor]


@dataclass
class PassStats:
    observed_log_probs: list[np.ndarray]
    argmax_ids: list[np.ndarray] | None
    cache: dict[tuple[int, str], list[torch.Tensor]] | None
    # Keyed by KL scope; each entry is a list of (positions, bins) log-prob arrays.
    digit_distributions: dict[str, list[np.ndarray]] | None = None


def single_digit_token_ids(tokenizer) -> np.ndarray:
    ids: list[int] = []
    for digit in "0123456789":
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected {digit!r} to be one token, got ids={encoded}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 10:
        raise ValueError(f"Digit token IDs are not unique: {ids}")
    return np.asarray(ids, dtype=np.int64)


def digit_log_distributions(
    float_logits: torch.Tensor,
    digit_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Conditional 10-digit and 11-bin digit-plus-other log distributions."""
    log_z = torch.logsumexp(float_logits, dim=-1)
    selected = float_logits.index_select(1, digit_ids)
    raw_digit_logs = selected - log_z[:, None]
    digit_mass = raw_digit_logs.exp().sum(dim=-1)
    other_logs = torch.log1p(
        -digit_mass.clamp(max=1.0 - torch.finfo(torch.float32).eps)
    )
    return {
        "digit_conditional": torch.log_softmax(selected, dim=-1),
        "digit_plus_other": torch.cat([raw_digit_logs, other_logs[:, None]], dim=-1),
    }


def kl_divergence_from_log_probs(
    reference_log_probs: np.ndarray,
    candidate_log_probs: np.ndarray,
) -> np.ndarray:
    """Return KL(reference || candidate) along the final bin dimension."""
    if reference_log_probs.shape != candidate_log_probs.shape:
        raise ValueError(
            "KL distributions must have matching shapes, got "
            f"{reference_log_probs.shape} and {candidate_log_probs.shape}"
        )
    return np.sum(
        np.exp(reference_log_probs)
        * (reference_log_probs - candidate_log_probs),
        axis=-1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--generation-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_GENERATION_SEEDS,
    )
    parser.add_argument("--rows-per-dataset", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(sl_config.ARTIFACTS_DIR) / "prompt_gate_sufficiency",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Resolve and print datasets without loading the model.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summaries from existing pair-level result files.",
    )
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


def normalized_target(value: str) -> str:
    return " ".join(value.casefold().split())


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalized_target(value)).strip("-")


def dataset_config(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("config") or entry.get("params") or {}


def resolve_dataset_path(
    artifacts_dir: Path,
    dataset_hash: str,
    entry: dict[str, Any],
) -> Path:
    migrated = artifacts_dir / "datasets" / f"{dataset_hash}.jsonl"
    if migrated.is_file():
        return migrated.resolve()
    registered = Path(entry["path"])
    if not registered.is_file():
        raise FileNotFoundError(
            f"Dataset {dataset_hash} is absent at {migrated} and {registered}"
        )
    return registered.resolve()


def is_canonical_number_dataset(
    cfg: dict[str, Any],
    target: str,
    category: str,
    generation_seed: int,
) -> bool:
    configured_target = cfg.get("animal") or cfg.get("target_preference")
    configured_category = cfg.get("category") or "animal"
    prompt = cfg.get("system_prompt_template") or ""
    return bool(
        configured_target
        and normalized_target(str(configured_target)) == normalized_target(target)
        and configured_category == category
        and cfg.get("generation_seed") == generation_seed
        and cfg.get("teacher_model") == BASE_MODEL
        and cfg.get("generation_temperature") == 1.0
        and cfg.get("number_min") == 100
        and cfg.get("number_max") == 999
        and cfg.get("answer_count") == 10
        and int(cfg.get("dataset_size") or 0) >= 10_000
        and cfg.get("user_prompt_prefix") is None
        and prompt
    )


def build_manifest(
    registry: dict[str, Any],
    artifacts_dir: Path,
    targets: list[str],
    generation_seeds: list[int],
) -> list[WorkItem]:
    target_lookup = {normalized_target(target): target for target in targets}
    unknown = sorted(set(target_lookup) - set(TARGET_CATEGORIES))
    if unknown:
        raise KeyError(
            f"Unknown targets={unknown}; choices={sorted(TARGET_CATEGORIES)}"
        )

    candidates: dict[tuple[str, int], list[WorkItem]] = {}
    for dataset_hash, entry in registry.get("datasets", {}).items():
        cfg = dataset_config(entry)
        raw_target = cfg.get("animal") or cfg.get("target_preference")
        if raw_target is None:
            continue
        target_key = normalized_target(str(raw_target))
        if target_key not in target_lookup:
            continue
        target = target_lookup[target_key]
        for generation_seed in generation_seeds:
            if not is_canonical_number_dataset(
                cfg,
                target,
                TARGET_CATEGORIES[normalized_target(target)],
                generation_seed,
            ):
                continue
            item = WorkItem(
                target=target,
                category=TARGET_CATEGORIES[normalized_target(target)],
                generation_seed=generation_seed,
                dataset_hash=dataset_hash,
                dataset_path=resolve_dataset_path(
                    artifacts_dir, dataset_hash, entry
                ),
                system_prompt=str(cfg["system_prompt_template"]),
                dataset_size=int(cfg["dataset_size"]),
            )
            candidates.setdefault((target, generation_seed), []).append(item)

    manifest: list[WorkItem] = []
    for target in targets:
        # Cat has a canonical 30k source while most targets have 10k.  The
        # largest source is preferred; equivalent duplicate registrations
        # resolve deterministically by hash.
        largest: dict[int, list[WorkItem]] = {}
        for generation_seed in generation_seeds:
            matches = candidates.get((target, generation_seed), [])
            if not matches:
                raise KeyError(
                    f"No canonical generated dataset for {target!r}, "
                    f"generation_seed={generation_seed}"
                )
            size = max(item.dataset_size for item in matches)
            largest[generation_seed] = sorted(
                (item for item in matches if item.dataset_size == size),
                key=lambda item: item.dataset_hash,
            )

        # Prompt-variant datasets (e.g. prompt-semantics sweeps) can match every
        # canonical generation field except the prompt text.  Such ties resolve
        # to the prompt shared by this target's unambiguous generation seeds.
        consensus = {
            tied[0].system_prompt
            for tied in largest.values()
            if len({item.system_prompt for item in tied}) == 1
        }
        for generation_seed in generation_seeds:
            tied = largest[generation_seed]
            if len({item.system_prompt for item in tied}) > 1:
                preferred = [
                    item for item in tied if item.system_prompt in consensus
                ]
                if len(consensus) != 1 or not preferred:
                    raise ValueError(
                        f"Conflicting canonical prompts for "
                        f"{target!r}/g{generation_seed}: "
                        f"{sorted(item.dataset_hash for item in tied)}"
                    )
                logger.warning(
                    f"{target!r}/g{generation_seed}: {len(tied)} same-size "
                    f"prompt variants; using {preferred[0].dataset_hash}, whose "
                    "prompt matches the other generation seeds"
                )
                tied = preferred
            manifest.append(tied[0])
    return manifest


def sample_rows(
    item: WorkItem,
    rows_per_dataset: int,
    sample_seed: int,
) -> tuple[list[DatasetRow], np.ndarray]:
    rows = dataset_services.read_dataset(str(item.dataset_path))
    if rows_per_dataset > len(rows):
        raise ValueError(
            f"Requested {rows_per_dataset} rows from {item.dataset_hash}, "
            f"which contains {len(rows)}"
        )
    digest = hashlib.sha256(
        f"{sample_seed}:{item.dataset_hash}".encode()
    ).digest()
    rng = np.random.default_rng(
        int.from_bytes(digest[:8], "little", signed=False)
    )
    indices = np.sort(
        rng.choice(len(rows), size=rows_per_dataset, replace=False)
    )
    return [rows[int(index)] for index in indices], indices


def encode_row(
    tokenizer,
    row: DatasetRow,
    system_prompt: str,
    max_seq_length: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    prefix_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row.prompt},
    ]
    full_messages = [
        *prefix_messages,
        {"role": "assistant", "content": row.completion},
    ]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    content_text = prefix_text + row.completion
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix_ids = tokenizer(
        prefix_text, add_special_tokens=False
    ).input_ids
    content_ids = tokenizer(
        content_text, add_special_tokens=False
    ).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if (
        full_ids[: len(prefix_ids)] != prefix_ids
        or full_ids[: len(content_ids)] != content_ids
    ):
        raise ValueError("Could not align assistant completion")
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"Encoded sequence has {len(full_ids)} tokens, exceeding "
            f"--max-seq-length={max_seq_length}"
        )
    target_positions = torch.arange(
        len(prefix_ids), len(content_ids), dtype=torch.long
    )
    if len(target_positions) == 0:
        raise ValueError("Completion has no tokens")
    labels = torch.tensor(full_ids, dtype=torch.long)[target_positions]
    return full_ids, target_positions - 1, labels


def encode_batch(
    tokenizer,
    rows: list[DatasetRow],
    system_prompt: str,
    max_seq_length: int,
) -> EncodedBatch:
    encoded = [
        encode_row(tokenizer, row, system_prompt, max_seq_length)
        for row in rows
    ]
    max_length = max(len(item[0]) for item in encoded)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    input_ids = torch.full(
        (len(rows), max_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    positions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_i, (ids, sample_positions, sample_labels) in enumerate(encoded):
        input_ids[batch_i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[batch_i, : len(ids)] = 1
        positions.append(sample_positions)
        labels.append(sample_labels)
    return EncodedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prediction_positions=positions,
        labels=labels,
    )


def load_model(max_seq_length: int):
    import torch._dynamo
    from unsloth import FastLanguageModel

    torch._dynamo.reset()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    tokenizer.padding_side = "right"
    FastLanguageModel.for_inference(model)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


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
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers on loaded model")


def branch_module(decoder_layers, layer: int, branch: str):
    mlp = decoder_layers[layer].mlp
    if branch == "gate":
        return mlp.gate_proj
    if branch == "up":
        return mlp.up_proj
    raise ValueError(branch)


@torch.inference_mode()
def run_pass(
    model,
    decoder_layers,
    batch: EncodedBatch,
    *,
    capture_layers: list[int] | None = None,
    donor: dict[tuple[int, str], list[torch.Tensor]] | None = None,
    patch_layers: list[int] | None = None,
    component: str | None = None,
    include_argmax: bool = False,
    digit_token_ids: np.ndarray | None = None,
) -> PassStats:
    device = next(model.parameters()).device
    positions = batch.prediction_positions
    handles = []
    captured: dict[tuple[int, str], list[torch.Tensor]] | None = (
        {} if capture_layers is not None else None
    )

    if capture_layers is not None:
        for layer in capture_layers:
            for branch in ("gate", "up"):
                key = (layer, branch)

                def make_capture_hook(
                    capture_key: tuple[int, str],
                ) -> Callable:
                    def hook(_module, _inputs, output):
                        assert captured is not None
                        captured[capture_key] = [
                            output[batch_i]
                            .index_select(
                                0, sample_positions.to(output.device)
                            )
                            .detach()
                            .clone()
                            for batch_i, sample_positions in enumerate(positions)
                        ]

                    return hook

                handles.append(
                    branch_module(
                        decoder_layers, layer, branch
                    ).register_forward_hook(make_capture_hook(key))
                )

    if donor is not None:
        if patch_layers is None or component is None:
            raise ValueError("A donor requires patch_layers and component")
        branches = (
            ("gate", "up") if component == "gate_up" else (component,)
        )
        for layer in patch_layers:
            for branch in branches:
                donor_values = donor[(layer, branch)]

                def make_patch_hook(values: list[torch.Tensor]) -> Callable:
                    def hook(_module, _inputs, output):
                        changed = output.clone()
                        for batch_i, (
                            sample_positions,
                            sample_values,
                        ) in enumerate(zip(positions, values)):
                            if len(sample_positions) != len(sample_values):
                                raise ValueError(
                                    "Donor and recipient completion lengths differ"
                                )
                            changed[batch_i].index_copy_(
                                0,
                                sample_positions.to(output.device),
                                sample_values.to(
                                    output.device, dtype=output.dtype
                                ),
                            )
                        return changed

                    return hook

                handles.append(
                    branch_module(
                        decoder_layers, layer, branch
                    ).register_forward_hook(make_patch_hook(donor_values))
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
    argmax_ids: list[np.ndarray] | None = [] if include_argmax else None
    digit_distributions: dict[str, list[np.ndarray]] | None = None
    digit_ids: torch.Tensor | None = None
    if digit_token_ids is not None:
        digit_distributions = {scope: [] for scope in KL_SCOPES}
        digit_ids = torch.as_tensor(
            digit_token_ids, device=outputs.logits.device, dtype=torch.long
        )
    for batch_i, (sample_positions, sample_labels) in enumerate(
        zip(positions, batch.labels)
    ):
        logits = outputs.logits[batch_i].index_select(
            0, sample_positions.to(outputs.logits.device)
        )
        labels = sample_labels.to(logits.device)
        observed = (
            logits.gather(1, labels[:, None]).squeeze(1).float()
            - torch.logsumexp(logits.float(), dim=-1)
        )
        observed_log_probs.append(
            observed.to("cpu", dtype=torch.float32).numpy()
        )
        if argmax_ids is not None:
            argmax_ids.append(
                logits.argmax(-1).to("cpu", dtype=torch.int64).numpy()
            )
        if digit_distributions is not None:
            assert digit_ids is not None
            for scope, log_probs in digit_log_distributions(
                logits.float(), digit_ids
            ).items():
                digit_distributions[scope].append(log_probs.cpu().numpy())
    del outputs
    return PassStats(
        observed_log_probs=observed_log_probs,
        argmax_ids=argmax_ids,
        cache=captured,
        digit_distributions=digit_distributions,
    )


def token_is_digit(
    tokenizer,
    token_id: int,
    cache: dict[int, bool],
) -> bool:
    if token_id not in cache:
        cache[token_id] = any(
            character.isdigit()
            for character in tokenizer.decode([int(token_id)])
        )
    return cache[token_id]


def scope_mask(
    tokenizer,
    labels: np.ndarray,
    target_argmax: np.ndarray,
    neutral_argmax: np.ndarray,
    scope: str,
    digit_cache: dict[int, bool],
) -> np.ndarray:
    if scope == "all":
        return np.ones(len(labels), dtype=bool)
    if scope == "digit":
        return np.asarray(
            [
                token_is_digit(tokenizer, int(token_id), digit_cache)
                for token_id in labels
            ],
            dtype=bool,
        )
    if scope in {"digit_related", "divergent_digit"}:
        digit_related = np.asarray(
            [
                any(
                    token_is_digit(tokenizer, int(token_id), digit_cache)
                    for token_id in token_ids
                )
                for token_ids in zip(
                    target_argmax, neutral_argmax, labels
                )
            ],
            dtype=bool,
        )
        if scope == "digit_related":
            return digit_related
        return (target_argmax != neutral_argmax) & digit_related
    raise ValueError(scope)


def analyze_item(
    model,
    tokenizer,
    item: WorkItem,
    rows_per_dataset: int,
    batch_size: int,
    sample_seed: int,
    max_seq_length: int,
) -> dict[str, Any]:
    rows, selected_indices = sample_rows(
        item, rows_per_dataset, sample_seed
    )
    decoder_layers = get_decoder_layers(model)
    all_layers = sorted(
        {
            layer
            for layers in DEFAULT_LAYER_GROUPS.values()
            for layer in layers
        }
    )
    digit_cache: dict[int, bool] = {}
    digit_token_ids = single_digit_token_ids(tokenizer)
    records: list[dict[str, Any]] = []
    kl_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    n_batches = math.ceil(len(rows) / batch_size)

    for batch_i, start in enumerate(range(0, len(rows), batch_size)):
        batch_rows = rows[start : start + batch_size]
        target_batch = encode_batch(
            tokenizer,
            batch_rows,
            item.system_prompt,
            max_seq_length,
        )
        neutral_batch = encode_batch(
            tokenizer,
            batch_rows,
            NEUTRAL_SYSTEM_PROMPT,
            max_seq_length,
        )
        for target_labels, neutral_labels in zip(
            target_batch.labels, neutral_batch.labels
        ):
            if not torch.equal(target_labels, neutral_labels):
                raise ValueError(
                    f"Completion labels differ for {item.target}/"
                    f"g{item.generation_seed}"
                )

        target = run_pass(
            model,
            decoder_layers,
            target_batch,
            capture_layers=all_layers,
            include_argmax=True,
            digit_token_ids=digit_token_ids,
        )
        neutral = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            capture_layers=all_layers,
            include_argmax=True,
            digit_token_ids=digit_token_ids,
        )
        assert target.cache is not None and neutral.cache is not None
        assert (
            target.argmax_ids is not None
            and neutral.argmax_ids is not None
        )

        sufficient: dict[tuple[str, str], PassStats] = {}
        necessary: dict[tuple[str, str], PassStats] = {}
        for layer_group, layers in DEFAULT_LAYER_GROUPS.items():
            for component in DEFAULT_COMPONENTS:
                sufficient[(layer_group, component)] = run_pass(
                    model,
                    decoder_layers,
                    neutral_batch,
                    donor=target.cache,
                    patch_layers=layers,
                    component=component,
                    include_argmax=True,
                    digit_token_ids=digit_token_ids,
                )
                necessary[(layer_group, component)] = run_pass(
                    model,
                    decoder_layers,
                    target_batch,
                    donor=neutral.cache,
                    patch_layers=layers,
                    component=component,
                    include_argmax=True,
                    digit_token_ids=digit_token_ids,
                )

        for local_i, row in enumerate(batch_rows):
            row_i = start + local_i
            dataset_row_i = int(selected_indices[row_i])
            labels = target_batch.labels[local_i].numpy()
            target_argmax = target.argmax_ids[local_i]
            neutral_argmax = neutral.argmax_ids[local_i]
            target_lp = target.observed_log_probs[local_i]
            neutral_lp = neutral.observed_log_probs[local_i]
            masks = {
                scope: scope_mask(
                    tokenizer,
                    labels,
                    target_argmax,
                    neutral_argmax,
                    scope,
                    digit_cache,
                )
                for scope in TOKEN_SCOPES
            }

            for position in np.flatnonzero(
                masks["divergent_digit"]
            ).tolist():
                divergences.append(
                    {
                        "row_i": row_i,
                        "dataset_row_i": dataset_row_i,
                        "completion_position": int(position),
                        "observed_token": tokenizer.decode(
                            [int(labels[position])]
                        ),
                        "target_argmax": tokenizer.decode(
                            [int(target_argmax[position])]
                        ),
                        "neutral_argmax": tokenizer.decode(
                            [int(neutral_argmax[position])]
                        ),
                        "observed_log_likelihood_ratio": float(
                            target_lp[position] - neutral_lp[position]
                        ),
                    }
                )

            row_key = {
                "target": item.target,
                "category": item.category,
                "generation_seed": item.generation_seed,
                "dataset_hash": item.dataset_hash,
                "row_i": row_i,
                "dataset_row_i": dataset_row_i,
            }
            digit_mask = masks["digit"]
            digit_related = scope_mask(
                tokenizer,
                labels,
                target_argmax,
                neutral_argmax,
                "digit_related",
                digit_cache,
            )
            prediction_masks = {
                "digit_related": digit_related,
                "all": np.ones(len(labels), dtype=bool),
            }
            target_distributions = {
                scope: target.digit_distributions[scope][local_i][digit_mask]
                for scope in KL_SCOPES
            }
            neutral_kl = {
                scope: float(
                    kl_divergence_from_log_probs(
                        target_distributions[scope],
                        neutral.digit_distributions[scope][local_i][digit_mask],
                    ).sum()
                )
                for scope in KL_SCOPES
            }

            for layer_group in DEFAULT_LAYER_GROUPS:
                for component in DEFAULT_COMPONENTS:
                    sufficient_pass = sufficient[(layer_group, component)]
                    necessary_pass = necessary[(layer_group, component)]
                    sufficient_lp = sufficient_pass.observed_log_probs[local_i]
                    necessary_lp = necessary_pass.observed_log_probs[local_i]
                    sufficient_ids = sufficient_pass.argmax_ids[local_i]
                    necessary_ids = necessary_pass.argmax_ids[local_i]

                    # KL(target || patched): sufficiency recovers the
                    # target-neutral KL; necessity measures how far clamping
                    # the branch to neutral pushes the target run away.
                    for scope in KL_SCOPES:
                        sufficient_kl = float(
                            kl_divergence_from_log_probs(
                                target_distributions[scope],
                                sufficient_pass.digit_distributions[scope][
                                    local_i
                                ][digit_mask],
                            ).sum()
                        )
                        necessary_kl = float(
                            kl_divergence_from_log_probs(
                                target_distributions[scope],
                                necessary_pass.digit_distributions[scope][
                                    local_i
                                ][digit_mask],
                            ).sum()
                        )
                        kl_records.append(
                            {
                                **row_key,
                                "kl_scope": scope,
                                "n_tokens": int(digit_mask.sum()),
                                "layer_group": layer_group,
                                "component": component,
                                "neutral_kl": neutral_kl[scope],
                                "sufficient_kl": sufficient_kl,
                                "sufficient_recovered_kl": (
                                    neutral_kl[scope] - sufficient_kl
                                ),
                                "necessary_kl": necessary_kl,
                            }
                        )

                    for scope, mask in prediction_masks.items():
                        target_diverges = (
                            target_argmax != neutral_argmax
                        ) & mask
                        sufficient_changes = (
                            sufficient_ids != neutral_argmax
                        ) & mask
                        prediction_records.append(
                            {
                                **row_key,
                                "token_scope": scope,
                                "n_tokens": int(mask.sum()),
                                "layer_group": layer_group,
                                "component": component,
                                "target_divergences": int(target_diverges.sum()),
                                "sufficient_changes": int(sufficient_changes.sum()),
                                "sufficient_recoveries": int(
                                    (
                                        target_diverges
                                        & (sufficient_ids == target_argmax)
                                    ).sum()
                                ),
                                "sufficient_other_changes": int(
                                    (
                                        sufficient_changes
                                        & (sufficient_ids != target_argmax)
                                    ).sum()
                                ),
                                "necessary_losses": int(
                                    (
                                        target_diverges
                                        & (necessary_ids != target_argmax)
                                    ).sum()
                                ),
                                "necessary_reversions": int(
                                    (
                                        target_diverges
                                        & (necessary_ids == neutral_argmax)
                                    ).sum()
                                ),
                            }
                        )

                    for scope, mask in masks.items():
                        records.append(
                            {
                                **row_key,
                                "token_scope": scope,
                                "n_tokens": int(mask.sum()),
                                "layer_group": layer_group,
                                "component": component,
                                "total_evidence": float(
                                    (target_lp[mask] - neutral_lp[mask]).sum()
                                ),
                                "sufficient_evidence": float(
                                    (
                                        sufficient_lp[mask]
                                        - neutral_lp[mask]
                                    ).sum()
                                ),
                                "necessary_evidence": float(
                                    (
                                        target_lp[mask]
                                        - necessary_lp[mask]
                                    ).sum()
                                ),
                            }
                        )

        del target, neutral, sufficient, necessary
        logger.info(
            f"{item.target}/g{item.generation_seed}: "
            f"batch {batch_i + 1}/{n_batches}"
        )

    return {
        "config": {
            "analysis_version": ANALYSIS_VERSION,
            "base_model": BASE_MODEL,
            "neutral_system_prompt": NEUTRAL_SYSTEM_PROMPT,
            "target": item.target,
            "category": item.category,
            "generation_seed": item.generation_seed,
            "dataset_hash": item.dataset_hash,
            "dataset_path": str(item.dataset_path),
            "dataset_size": item.dataset_size,
            "target_system_prompt": item.system_prompt,
            "rows_per_dataset": rows_per_dataset,
            "sample_seed": sample_seed,
            "layer_groups": DEFAULT_LAYER_GROUPS,
            "components": DEFAULT_COMPONENTS,
            "token_scopes": TOKEN_SCOPES,
            "kl_scopes": KL_SCOPES,
            "prediction_scopes": PREDICTION_SCOPES,
        },
        "selected_indices": selected_indices.tolist(),
        "records": records,
        "kl_records": kl_records,
        "prediction_records": prediction_records,
        "divergence_examples": divergences,
    }


def result_path(output_dir: Path, item: WorkItem) -> Path:
    return (
        output_dir
        / "pairs"
        / f"{slug(item.target)}_g{item.generation_seed}.json"
    )


def result_is_compatible(
    path: Path,
    item: WorkItem,
    rows_per_dataset: int,
    sample_seed: int,
) -> bool:
    if not path.is_file():
        return False
    config = json.loads(path.read_text()).get("config", {})
    return bool(
        config.get("analysis_version") == ANALYSIS_VERSION
        and config.get("dataset_hash") == item.dataset_hash
        and config.get("rows_per_dataset") == rows_per_dataset
        and config.get("sample_seed") == sample_seed
        and config.get("layer_groups") == DEFAULT_LAYER_GROUPS
        and config.get("components") == DEFAULT_COMPONENTS
        and config.get("token_scopes") == TOKEN_SCOPES
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def ratio(numerators: np.ndarray, denominators: np.ndarray) -> float:
    denominator = float(denominators.sum())
    return (
        float(numerators.sum() / denominator)
        if denominator != 0
        else float("nan")
    )


def stratified_bootstrap_ci(
    records: list[dict[str, Any]],
    numerator: str,
    samples: int,
    rng: np.random.Generator,
    denominator: str = "total_evidence",
) -> tuple[float, float]:
    strata: list[tuple[np.ndarray, np.ndarray]] = []
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
                np.asarray(
                    [record[numerator] for record in seed_records],
                    dtype=np.float64,
                ),
                np.asarray(
                    [record[denominator] for record in seed_records],
                    dtype=np.float64,
                ),
            )
        )

    estimates = np.empty(samples, dtype=np.float64)
    for bootstrap_i in range(samples):
        numerator_sum = 0.0
        denominator_sum = 0.0
        for numerators, denominators in strata:
            indices = rng.integers(0, len(numerators), len(numerators))
            numerator_sum += float(numerators[indices].sum())
            denominator_sum += float(denominators[indices].sum())
        estimates[bootstrap_i] = (
            numerator_sum / denominator_sum
            if denominator_sum != 0
            else np.nan
        )
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    manifest: list[WorkItem],
    output_dir: Path,
    bootstrap_samples: int,
    sample_seed: int,
    rows_per_dataset: int,
) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    all_kl_records: list[dict[str, Any]] = []
    all_prediction_records: list[dict[str, Any]] = []
    missing = []
    for item in manifest:
        path = result_path(output_dir, item)
        if not result_is_compatible(
            path, item, rows_per_dataset, sample_seed
        ):
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        all_records.extend(payload["records"])
        all_kl_records.extend(payload["kl_records"])
        all_prediction_records.extend(payload["prediction_records"])
    if missing:
        logger.warning(
            f"Summary is partial: {len(missing)} pair result files are missing"
        )
    if not all_records:
        raise FileNotFoundError("No pair-level results are available to summarize")

    keys = sorted(
        {
            (
                record["target"],
                record["category"],
                record["token_scope"],
                record["layer_group"],
                record["component"],
            )
            for record in all_records
        }
    )
    summary: list[dict[str, Any]] = []
    for key_i, (
        target,
        category,
        token_scope,
        layer_group,
        component,
    ) in enumerate(keys):
        records = [
            record
            for record in all_records
            if (
                record["target"],
                record["category"],
                record["token_scope"],
                record["layer_group"],
                record["component"],
            )
            == (
                target,
                category,
                token_scope,
                layer_group,
                component,
            )
        ]
        total = np.asarray(
            [record["total_evidence"] for record in records],
            dtype=np.float64,
        )
        sufficient = np.asarray(
            [record["sufficient_evidence"] for record in records],
            dtype=np.float64,
        )
        necessary = np.asarray(
            [record["necessary_evidence"] for record in records],
            dtype=np.float64,
        )
        rng = np.random.default_rng(sample_seed + key_i)
        sufficiency_ci = stratified_bootstrap_ci(
            records, "sufficient_evidence", bootstrap_samples, rng
        )
        necessity_ci = stratified_bootstrap_ci(
            records, "necessary_evidence", bootstrap_samples, rng
        )
        seed_estimates = {}
        for generation_seed in sorted(
            {int(record["generation_seed"]) for record in records}
        ):
            seed_records = [
                record
                for record in records
                if int(record["generation_seed"]) == generation_seed
            ]
            seed_estimates[str(generation_seed)] = ratio(
                np.asarray(
                    [
                        record["sufficient_evidence"]
                        for record in seed_records
                    ]
                ),
                np.asarray(
                    [record["total_evidence"] for record in seed_records]
                ),
            )
        summary.append(
            {
                "target": target,
                "category": category,
                "token_scope": token_scope,
                "layer_group": layer_group,
                "component": component,
                "rows": len(records),
                "tokens": sum(int(record["n_tokens"]) for record in records),
                "generation_seeds": sorted(
                    {
                        int(record["generation_seed"])
                        for record in records
                    }
                ),
                "target_effect_nats_per_token": (
                    float(total.sum())
                    / sum(int(record["n_tokens"]) for record in records)
                    if sum(
                        int(record["n_tokens"]) for record in records
                    )
                    else float("nan")
                ),
                "sufficiency_fraction": ratio(sufficient, total),
                "sufficiency_ci_low": sufficiency_ci[0],
                "sufficiency_ci_high": sufficiency_ci[1],
                "necessity_fraction": ratio(necessary, total),
                "necessity_ci_low": necessity_ci[0],
                "necessity_ci_high": necessity_ci[1],
                "sufficiency_by_seed": seed_estimates,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "summary.json", summary)
    csv_rows = [
        {
            **{
                key: value
                for key, value in record.items()
                if key != "sufficiency_by_seed"
            },
            "sufficiency_by_seed": json.dumps(
                record["sufficiency_by_seed"], sort_keys=True
            ),
        }
        for record in summary
    ]
    with open(output_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.success(f"Wrote {output_dir / 'summary.json'}")
    logger.success(f"Wrote {output_dir / 'summary.csv'}")

    write_csv(
        output_dir / "kl_summary.csv",
        summarize_kl(all_kl_records, bootstrap_samples, sample_seed),
    )
    write_csv(
        output_dir / "prediction_summary.csv",
        summarize_predictions(all_prediction_records),
    )
    logger.success(f"Wrote {output_dir / 'kl_summary.csv'}")
    logger.success(f"Wrote {output_dir / 'prediction_summary.csv'}")
    return summary


def group_records(
    records: list[dict[str, Any]],
    scope_field: str,
) -> dict[tuple[str, str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["target"],
            record["category"],
            record[scope_field],
            record["layer_group"],
            record["component"],
        )
        groups.setdefault(key, []).append(record)
    return dict(sorted(groups.items()))


def summarize_kl(
    records: list[dict[str, Any]],
    bootstrap_samples: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Fractions of summed KL(target || neutral), pooled over rows and seeds."""
    rows = []
    for key_i, (key, group) in enumerate(group_records(records, "kl_scope").items()):
        target, category, scope, layer_group, component = key
        tokens = sum(int(record["n_tokens"]) for record in group)
        neutral_kl = np.asarray([record["neutral_kl"] for record in group])
        row: dict[str, Any] = {
            "target": target,
            "category": category,
            "kl_scope": scope,
            "layer_group": layer_group,
            "component": component,
            "rows": len(group),
            "tokens": tokens,
            "generation_seeds": json.dumps(
                sorted({int(record["generation_seed"]) for record in group})
            ),
            "neutral_kl_nats_per_token": (
                float(neutral_kl.sum()) / tokens if tokens else float("nan")
            ),
        }
        rng = np.random.default_rng(sample_seed + 10_000 + key_i)
        for metric, numerator in [
            ("sufficiency_fraction", "sufficient_recovered_kl"),
            ("necessity_fraction", "necessary_kl"),
        ]:
            row[metric] = ratio(
                np.asarray([record[numerator] for record in group]), neutral_kl
            )
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = (
                stratified_bootstrap_ci(
                    group, numerator, bootstrap_samples, rng, "neutral_kl"
                )
            )
        rows.append(row)
    return rows


def summarize_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top-1 recovery (sufficiency) and loss (necessity) of target divergences."""
    rows = []
    for key, group in group_records(records, "token_scope").items():
        target, category, scope, layer_group, component = key
        totals = {
            field: sum(int(record[field]) for record in group)
            for field in [
                "n_tokens",
                "target_divergences",
                "sufficient_changes",
                "sufficient_recoveries",
                "sufficient_other_changes",
                "necessary_losses",
                "necessary_reversions",
            ]
        }
        divergences = totals["target_divergences"]
        rows.append(
            {
                "target": target,
                "category": category,
                "token_scope": scope,
                "layer_group": layer_group,
                "component": component,
                "rows": len(group),
                "tokens": totals["n_tokens"],
                **{
                    field: value
                    for field, value in totals.items()
                    if field != "n_tokens"
                },
                "target_divergence_rate": (
                    divergences / totals["n_tokens"]
                    if totals["n_tokens"]
                    else float("nan")
                ),
                "sufficiency_recovery_rate": (
                    totals["sufficient_recoveries"] / divergences
                    if divergences
                    else float("nan")
                ),
                "necessity_loss_rate": (
                    totals["necessary_losses"] / divergences
                    if divergences
                    else float("nan")
                ),
                "necessity_reversion_rate": (
                    totals["necessary_reversions"] / divergences
                    if divergences
                    else float("nan")
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.rows_per_dataset < 1:
        raise ValueError("--rows-per-dataset must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError(
            f"Invalid shard {args.shard_id}/{args.num_shards}"
        )

    artifacts_dir = Path(sl_config.ARTIFACTS_DIR).resolve()
    registry_path = artifacts_dir / "registry.json"
    logger.info(f"Loading registry once from {registry_path}")
    registry = json.loads(registry_path.read_text())
    targets = [normalized_target(target) for target in args.targets]
    manifest = build_manifest(
        registry,
        artifacts_dir,
        targets,
        args.generation_seeds,
    )
    logger.info(
        f"Resolved {len(manifest)} datasets: "
        f"{len(targets)} targets × {len(args.generation_seeds)} seeds"
    )
    for item in manifest:
        logger.info(
            f"  {item.category}/{item.target}/g{item.generation_seed}: "
            f"{item.dataset_hash} ({item.dataset_size} rows)"
        )
    if args.manifest_only:
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
                result_path(args.output_dir, item),
                item,
                args.rows_per_dataset,
                args.sample_seed,
            )
        ]
        logger.info(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"{len(pending)}/{len(shard)} datasets pending"
        )
        if pending:
            model, tokenizer = load_model(args.max_seq_length)
            for item in pending:
                payload = analyze_item(
                    model,
                    tokenizer,
                    item,
                    args.rows_per_dataset,
                    args.batch_size,
                    args.sample_seed,
                    args.max_seq_length,
                )
                path = result_path(args.output_dir, item)
                write_json_atomic(path, payload)
                logger.success(f"Saved {path}")

    if args.summarize_only or args.num_shards == 1:
        summarize(
            manifest,
            args.output_dir,
            args.bootstrap_samples,
            args.sample_seed,
            args.rows_per_dataset,
        )
    else:
        # Whichever array task finishes last writes the shared summaries.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = args.output_dir / ".summary.lock"
        with open(lock_path, "a+") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            complete = all(
                result_is_compatible(
                    result_path(args.output_dir, item),
                    item,
                    args.rows_per_dataset,
                    args.sample_seed,
                )
                for item in manifest
            )
            if complete:
                summarize(
                    manifest,
                    args.output_dir,
                    args.bootstrap_samples,
                    args.sample_seed,
                    args.rows_per_dataset,
                )
            else:
                logger.info(
                    "Some pair files are still pending; the last array task "
                    "will write the shared summary."
                )


if __name__ == "__main__":
    main()
