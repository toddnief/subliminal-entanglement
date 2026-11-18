#!/usr/bin/env python3
"""Run the full attention/FFN mediation suite on subliminal students.

The student LoRA (adapter enabled, no system prompt) is the source condition
and the adapter-disabled base on the identical tokens is the neutral recipient.
All 28 layers are patched, matching ``score_attention_ffn_mediation.py``.

Besides student-vs-base digit KL and argmax recovery, this records how well each
condition preserves positions where the student recovered the prompted
teacher's cached digit argmax.
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
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_attention_ffn_mediation import run_pass  # noqa: E402
from scripts.score_prompt_gate_sufficiency import (  # noqa: E402
    kl_divergence_from_log_probs,
)
from scripts.score_student_digit_argmax import (  # noqa: E402
    DEFAULT_MODELS,
    load_cached_examples,
    parse_model,
)
from scripts.score_student_gate_sufficiency import (  # noqa: E402
    encode_neutral_batch,
    get_student_decoder_layers,
    ratio,
)
from scripts.score_training_topk import single_digit_token_ids  # noqa: E402
from scripts.score_val_loss import MODELS_DIR, cleanup_model, load_model_and_tokenizer  # noqa: E402
from sl import config as sl_config  # noqa: E402

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "student_attention_ffn_mediation"

CONDITIONS = [
    "attention",
    "attention_only",
    "ffn_only",
    "gate_up_free",
    "gate_up_neutral",
    "up_gate_neutral",
]


def empty_counts() -> dict[str, int | float]:
    return {
        "n_digit_positions": 0,
        "n_student_base_digit_divergences": 0,
        "n_teacher_base_digit_divergences": 0,
        "n_student_teacher_recoveries": 0,
        "n_condition_matches_student_on_student_base_divergence": 0,
        "n_condition_matches_teacher_on_teacher_base_divergence": 0,
        "n_condition_preserves_student_teacher_recovery": 0,
        "student_base_digit_kl": 0.0,
        "student_condition_digit_kl": 0.0,
        "n_full_vocab_digit_related_positions": 0,
        "n_student_base_full_vocab_divergences": 0,
        "n_condition_matches_student_full_vocab": 0,
    }


def digit_argmax(pass_stats, sample_i: int, digit_ids: np.ndarray) -> np.ndarray:
    return digit_ids[pass_stats.digit_log_probs[sample_i].argmax(-1)]


def score_model(animal: str, model_hash: str, batch_size: int) -> dict[str, Any]:
    rows, cached, metadata, cache_path = load_cached_examples(animal)
    adapter_path = MODELS_DIR / model_hash
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter weights at {adapter_path}")

    model, tokenizer = load_model_and_tokenizer(adapter_path)
    model.config.use_cache = False
    decoder_layers = get_student_decoder_layers(model)
    layers = list(range(len(decoder_layers)))
    digit_ids = single_digit_token_ids(tokenizer)
    cached_targets = cached["target_ids"].astype(np.int64, copy=False)
    teacher_flat = cached["teacher_digit_top_ids"][:, 0].astype(np.int64, copy=False)
    counts = {condition: empty_counts() for condition in CONDITIONS}
    cursor = 0
    started = time.time()

    try:
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            batch = encode_neutral_batch(tokenizer, batch_rows)

            student = run_pass(
                model, decoder_layers, batch, layers, digit_ids, capture=True
            )
            with model.disable_adapter():
                base = run_pass(
                    model, decoder_layers, batch, layers, digit_ids, capture=True
                )
            if student.cache is None or base.cache is None:
                raise RuntimeError("Failed to capture source activations")

            patched = {}
            with model.disable_adapter():
                patched["attention"] = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    attention_donor=student.cache,
                )
                patched["attention_only"] = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    attention_donor=student.cache,
                    ffn_donor=base.cache,
                )
                patched["ffn_only"] = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    attention_donor=base.cache,
                    ffn_donor=student.cache,
                )
                patched["gate_up_free"] = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    attention_donor=base.cache,
                    gate_donor=student.cache,
                )
                patched["gate_up_neutral"] = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    attention_donor=base.cache,
                    gate_donor=student.cache,
                    up_donor=base.cache,
                )
                patched["up_gate_neutral"] = run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    attention_donor=base.cache,
                    gate_donor=base.cache,
                    up_donor=student.cache,
                )

            for sample_i, labels_tensor in enumerate(batch.labels):
                labels = labels_tensor.numpy()
                end = cursor + len(labels)
                if not np.array_equal(labels, cached_targets[cursor:end]):
                    raise RuntimeError("Cached and current completion tokens do not align")
                teacher_ids = teacher_flat[cursor:end]
                student_digit = digit_argmax(student, sample_i, digit_ids)
                base_digit = digit_argmax(base, sample_i, digit_ids)
                student_full = student.argmax_ids[sample_i]
                base_full = base.argmax_ids[sample_i]
                is_digit = np.isin(labels, digit_ids)
                student_base_digit_div = is_digit & (student_digit != base_digit)
                teacher_base_digit_div = is_digit & (teacher_ids != base_digit)
                student_teacher_recovery = teacher_base_digit_div & (
                    student_digit == teacher_ids
                )
                digit_related = (
                    is_digit
                    | np.isin(student_full, digit_ids)
                    | np.isin(base_full, digit_ids)
                )
                student_base_full_div = digit_related & (student_full != base_full)

                student_digit_logs = student.digit_log_probs[sample_i][is_digit]
                base_digit_logs = base.digit_log_probs[sample_i][is_digit]
                base_kl = float(
                    kl_divergence_from_log_probs(
                        student_digit_logs, base_digit_logs
                    ).sum()
                )

                for condition, condition_pass in patched.items():
                    values = counts[condition]
                    condition_digit = digit_argmax(condition_pass, sample_i, digit_ids)
                    condition_full = condition_pass.argmax_ids[sample_i]
                    condition_kl = float(
                        kl_divergence_from_log_probs(
                            student_digit_logs,
                            condition_pass.digit_log_probs[sample_i][is_digit],
                        ).sum()
                    )
                    values["n_digit_positions"] += int(is_digit.sum())
                    values["n_student_base_digit_divergences"] += int(
                        student_base_digit_div.sum()
                    )
                    values["n_teacher_base_digit_divergences"] += int(
                        teacher_base_digit_div.sum()
                    )
                    values["n_student_teacher_recoveries"] += int(
                        student_teacher_recovery.sum()
                    )
                    values[
                        "n_condition_matches_student_on_student_base_divergence"
                    ] += int(
                        (
                            student_base_digit_div
                            & (condition_digit == student_digit)
                        ).sum()
                    )
                    values[
                        "n_condition_matches_teacher_on_teacher_base_divergence"
                    ] += int(
                        (
                            teacher_base_digit_div
                            & (condition_digit == teacher_ids)
                        ).sum()
                    )
                    values[
                        "n_condition_preserves_student_teacher_recovery"
                    ] += int(
                        (
                            student_teacher_recovery
                            & (condition_digit == teacher_ids)
                        ).sum()
                    )
                    values["student_base_digit_kl"] += base_kl
                    values["student_condition_digit_kl"] += condition_kl
                    values["n_full_vocab_digit_related_positions"] += int(
                        digit_related.sum()
                    )
                    values["n_student_base_full_vocab_divergences"] += int(
                        student_base_full_div.sum()
                    )
                    values["n_condition_matches_student_full_vocab"] += int(
                        (
                            student_base_full_div
                            & (condition_full == student_full)
                        ).sum()
                    )
                cursor = end

            logger.info(f"  rows={min(batch_start + batch_size, len(rows))}/{len(rows)}")
            del student, base, patched
    finally:
        cleanup_model(model)

    if cursor != len(cached_targets):
        raise RuntimeError(f"Scored {cursor}/{len(cached_targets)} positions")

    metrics = {}
    for condition, values in counts.items():
        student_digit_div = int(values["n_student_base_digit_divergences"])
        teacher_digit_div = int(values["n_teacher_base_digit_divergences"])
        student_recoveries = int(values["n_student_teacher_recoveries"])
        full_div = int(values["n_student_base_full_vocab_divergences"])
        base_kl = float(values["student_base_digit_kl"])
        condition_kl = float(values["student_condition_digit_kl"])
        metrics[condition] = {
            **values,
            "student_digit_argmax_recovery_rate": ratio(
                int(values["n_condition_matches_student_on_student_base_divergence"]),
                student_digit_div,
            ),
            "teacher_digit_argmax_rate": ratio(
                int(values["n_condition_matches_teacher_on_teacher_base_divergence"]),
                teacher_digit_div,
            ),
            "student_teacher_recovery_preservation_rate": ratio(
                int(values["n_condition_preserves_student_teacher_recovery"]),
                student_recoveries,
            ),
            "digit_kl_recovery_fraction": (
                1.0 - condition_kl / base_kl if base_kl else None
            ),
            "full_vocab_argmax_recovery_rate": ratio(
                int(values["n_condition_matches_student_full_vocab"]), full_div
            ),
        }

    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "animal": animal,
        "model_hash": model_hash,
        "adapter_path": str(adapter_path),
        "topk_cache_path": str(cache_path),
        "dataset_hash": metadata["dataset_hash"],
        "generation_seed": metadata["generation_seed"],
        "n_examples": len(rows),
        "layers": layers,
        "conditions": CONDITIONS,
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
        result = score_model(animal, model_hash, args.batch_size)
        out_path.write_text(json.dumps(result, indent=2))
        logger.success(f"wrote {out_path}")


if __name__ == "__main__":
    main()
