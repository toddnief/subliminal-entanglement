#!/usr/bin/env python3
"""Test whether student SwiGLU branches carry recovered teacher digit argmaxes.

For each selected subliminal LoRA, run the same neutral-prompt examples used by
``score_student_digit_argmax.py`` under four conditions:

1. student (adapter enabled);
2. base (adapter disabled);
3. base with student gate/up activations patched in (sufficiency);
4. student with base gate/up activations patched in (necessity).

The primary sufficiency rate is the fraction of positions recovered by the
student (teacher digit argmax != base, student == teacher) that remain teacher
matches after patching student activations into the base.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_prompt_gate_sufficiency import (  # noqa: E402
    DEFAULT_COMPONENTS,
    DEFAULT_LAYER_GROUPS,
    EncodedBatch,
    get_decoder_layers,
    run_pass,
)
from scripts.score_student_digit_argmax import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR as DIGIT_OUTPUT_DIR,
    load_cached_examples,
    parse_model,
)
from scripts.score_training_topk import encode_example, single_digit_token_ids  # noqa: E402
from scripts.score_val_loss import (  # noqa: E402
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    cleanup_model,
    load_model_and_tokenizer,
)
from sl import config as sl_config  # noqa: E402

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "student_gate_sufficiency"


def get_student_decoder_layers(model):
    """Resolve Qwen layers through either a bare model or PEFT wrappers."""
    try:
        return get_decoder_layers(model)
    except AttributeError:
        pass
    candidates = [
        getattr(
            getattr(
                getattr(getattr(model, "base_model", None), "model", None),
                "model",
                None,
            ),
            "layers",
            None,
        ),
        getattr(
            getattr(
                getattr(
                    getattr(getattr(model, "base_model", None), "model", None),
                    "base_model",
                    None,
                ),
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
    raise AttributeError("Could not locate decoder layers through PEFT wrapper")


def encode_neutral_batch(tokenizer, rows: list) -> EncodedBatch:
    encoded = [
        encode_example(
            tokenizer,
            row,
            system_prompt=None,
            user_prompt_prefix=None,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        for row in rows
    ]
    max_length = max(len(item["input_ids"]) for item in encoded)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad or EOS token")
    input_ids = torch.full((len(rows), max_length), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    positions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_i, item in enumerate(encoded):
        ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
        input_ids[batch_i, : len(ids)] = ids
        attention_mask[batch_i, : len(ids)] = 1
        start, end = item["assistant_start"], item["assistant_end"]
        positions.append(torch.arange(start, end, dtype=torch.long) - 1)
        labels.append(ids[start:end].clone())
    return EncodedBatch(input_ids, attention_mask, positions, labels)


def digit_argmax(pass_stats, sample_i: int, digit_ids: np.ndarray) -> np.ndarray:
    if pass_stats.digit_distributions is None:
        raise RuntimeError("Pass did not retain digit distributions")
    local_ids = pass_stats.digit_distributions["digit_conditional"][sample_i].argmax(-1)
    return digit_ids[local_ids]


def empty_counts() -> dict[str, int]:
    return {
        "n_digit_positions": 0,
        "n_teacher_base_divergent": 0,
        "n_student_teacher_recoveries": 0,
        "n_sufficient_matches_student_on_divergence": 0,
        "n_sufficient_matches_teacher_on_divergence": 0,
        "n_sufficient_preserves_student_recovery": 0,
        "n_necessary_matches_teacher_on_divergence": 0,
        "n_necessary_loses_student_recovery": 0,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_model(
    animal: str,
    model_hash: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    rows, cached, metadata, cache_path = load_cached_examples(animal)
    adapter_path = MODELS_DIR / model_hash
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter weights at {adapter_path}")

    model, tokenizer = load_model_and_tokenizer(
        adapter_path,
        local_files_only=True,
    )
    model.config.use_cache = False
    decoder_layers = get_student_decoder_layers(model)
    digit_ids = single_digit_token_ids(tokenizer)
    all_layers = sorted(
        {layer for layers in DEFAULT_LAYER_GROUPS.values() for layer in layers}
    )
    conditions = [
        (layer_group, component)
        for layer_group in DEFAULT_LAYER_GROUPS
        for component in DEFAULT_COMPONENTS
    ]
    counts = {f"{group}/{component}": empty_counts() for group, component in conditions}
    cached_targets = cached["target_ids"].astype(np.int64, copy=False)
    teacher_flat = cached["teacher_digit_top_ids"][:, 0].astype(np.int64, copy=False)
    base_flat = cached["student_digit_top_ids"][:, 0].astype(np.int64, copy=False)
    cursor = 0
    n_base_cache_positions = 0
    n_base_cache_mismatches = 0
    started = time.time()

    try:
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            batch = encode_neutral_batch(tokenizer, batch_rows)
            student = run_pass(
                model,
                decoder_layers,
                batch,
                capture_layers=all_layers,
                digit_token_ids=digit_ids,
            )
            with model.disable_adapter():
                base = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    capture_layers=all_layers,
                    digit_token_ids=digit_ids,
                )
            if student.cache is None or base.cache is None:
                raise RuntimeError("Failed to capture gate/up activations")

            sufficient = {}
            necessary = {}
            for layer_group, component in conditions:
                layers = DEFAULT_LAYER_GROUPS[layer_group]
                with model.disable_adapter():
                    sufficient[(layer_group, component)] = run_pass(
                        model,
                        decoder_layers,
                        batch,
                        donor=student.cache,
                        patch_layers=layers,
                        component=component,
                        digit_token_ids=digit_ids,
                    )
                necessary[(layer_group, component)] = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    donor=base.cache,
                    patch_layers=layers,
                    component=component,
                    digit_token_ids=digit_ids,
                )

            for sample_i, labels_tensor in enumerate(batch.labels):
                labels = labels_tensor.numpy()
                end = cursor + len(labels)
                if not np.array_equal(labels, cached_targets[cursor:end]):
                    raise RuntimeError("Cached and current completion tokens do not align")
                teacher_ids = teacher_flat[cursor:end]
                cached_base_ids = base_flat[cursor:end]
                student_ids = digit_argmax(student, sample_i, digit_ids)
                base_ids = digit_argmax(base, sample_i, digit_ids)
                n_base_cache_positions += len(base_ids)
                n_base_cache_mismatches += int((base_ids != cached_base_ids).sum())

                is_digit = np.isin(labels, digit_ids)
                # Use the contemporaneous adapter-disabled pass for intervention
                # masks. The older top-k cache was produced through vanilla
                # Transformers/eager attention and can differ at near-tied
                # digit logits from the current Unsloth inference path.
                divergent = is_digit & (teacher_ids != base_ids)
                recovered = divergent & (student_ids == teacher_ids)
                for layer_group, component in conditions:
                    key = f"{layer_group}/{component}"
                    values = counts[key]
                    sufficient_ids = digit_argmax(
                        sufficient[(layer_group, component)], sample_i, digit_ids
                    )
                    necessary_ids = digit_argmax(
                        necessary[(layer_group, component)], sample_i, digit_ids
                    )
                    values["n_digit_positions"] += int(is_digit.sum())
                    values["n_teacher_base_divergent"] += int(divergent.sum())
                    values["n_student_teacher_recoveries"] += int(recovered.sum())
                    values["n_sufficient_matches_student_on_divergence"] += int(
                        ((sufficient_ids == student_ids) & divergent).sum()
                    )
                    values["n_sufficient_matches_teacher_on_divergence"] += int(
                        ((sufficient_ids == teacher_ids) & divergent).sum()
                    )
                    values["n_sufficient_preserves_student_recovery"] += int(
                        ((sufficient_ids == teacher_ids) & recovered).sum()
                    )
                    values["n_necessary_matches_teacher_on_divergence"] += int(
                        ((necessary_ids == teacher_ids) & divergent).sum()
                    )
                    values["n_necessary_loses_student_recovery"] += int(
                        ((necessary_ids != teacher_ids) & recovered).sum()
                    )
                cursor = end

            logger.info(f"  rows={min(batch_start + batch_size, len(rows))}/{len(rows)}")
            del student, base, sufficient, necessary
    finally:
        cleanup_model(model)

    if cursor != len(cached_targets):
        raise RuntimeError(f"Scored {cursor}/{len(cached_targets)} cached positions")

    metrics = {}
    for key, values in counts.items():
        divergent = values["n_teacher_base_divergent"]
        recovered = values["n_student_teacher_recoveries"]
        metrics[key] = {
            **values,
            "student_teacher_recovery_rate": ratio(recovered, divergent),
            "sufficiency_student_match_rate": ratio(
                values["n_sufficient_matches_student_on_divergence"], divergent
            ),
            "sufficiency_teacher_match_rate": ratio(
                values["n_sufficient_matches_teacher_on_divergence"], divergent
            ),
            "sufficiency_preservation_rate": ratio(
                values["n_sufficient_preserves_student_recovery"], recovered
            ),
            "necessity_teacher_match_rate": ratio(
                values["n_necessary_matches_teacher_on_divergence"], divergent
            ),
            "necessity_loss_rate": ratio(
                values["n_necessary_loses_student_recovery"], recovered
            ),
        }

    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "animal": animal,
        "model_hash": model_hash,
        "adapter_path": str(adapter_path),
        "topk_cache_path": str(cache_path),
        "digit_score_path": str(DIGIT_OUTPUT_DIR / f"{model_hash}.json"),
        "dataset_hash": metadata["dataset_hash"],
        "generation_seed": metadata["generation_seed"],
        "n_examples": len(rows),
        "layer_groups": DEFAULT_LAYER_GROUPS,
        "components": DEFAULT_COMPONENTS,
        "n_base_cache_positions": n_base_cache_positions,
        "n_base_cache_mismatches": n_base_cache_mismatches,
        "base_cache_mismatch_fraction": ratio(
            n_base_cache_mismatches, n_base_cache_positions
        ),
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.total_tasks < 1 or not 0 <= args.task_id < args.total_tasks:
        parser.error(f"invalid shard {args.task_id}/{args.total_tasks}")

    specs = args.model or list(DEFAULT_MODELS.items())
    specs = [spec for i, spec in enumerate(specs) if i % args.total_tasks == args.task_id]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Task {args.task_id}/{args.total_tasks}: {len(specs)} model(s)")
    for animal, model_hash in specs:
        out_path = args.output_dir / f"{model_hash}.json"
        if out_path.exists() and not args.force:
            logger.info(f"cached {animal}/{model_hash}: {out_path}")
            continue
        result = score_model(animal, model_hash, batch_size=args.batch_size)
        out_path.write_text(json.dumps(result, indent=2))
        logger.success(f"wrote {out_path}")


if __name__ == "__main__":
    main()
