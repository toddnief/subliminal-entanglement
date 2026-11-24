#!/usr/bin/env python3
"""Measure whether subliminal students reproduce teacher digit argmaxes.

This reuses the sampled canonical-dataset caches from ``score_training_topk``.
Those caches contain base-model digit argmaxes under the animal teacher prompt
and under no system prompt.  For each selected LoRA, this script runs only the
missing student pass (no system prompt) on the exact same examples.

The primary metric is positional agreement with the teacher's digit-restricted
argmax at completion positions whose target is a single digit.  We also report
agreement on the more diagnostic subset where the prompted teacher and neutral
base model disagree, together with the neutral-base baseline.

Example:
    uv run python scripts/score_student_digit_argmax.py \
      --model cat=dc37662901f8 --model eagle=09dfc574ff6f
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

from scripts.score_training_topk import (  # noqa: E402
    BASE_MODEL_ID,
    DEFAULT_DATASET_HASHES,
    DEFAULT_OUT_DIR as TOPK_DIR,
    encode_example,
    single_digit_token_ids,
)
from scripts.score_val_loss import (  # noqa: E402
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    cleanup_model,
    load_model_and_tokenizer,
)
from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "student_digit_argmax"

# Strong, completed subliminal-transfer runs chosen from the registry.  The
# evaluation data below remain the same canonical generation-seed-42 datasets
# for every model, regardless of the dataset on which its LoRA was trained.
DEFAULT_MODELS = {
    "cat": "dc37662901f8",
    "eagle": "09dfc574ff6f",
    "wolf": "336bec8cf7f3",
    "dolphin": "9fd4e25d3a4d",
}


def parse_model(value: str) -> tuple[str, str]:
    try:
        animal, model_hash = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected ANIMAL=MODEL_HASH") from exc
    if animal not in DEFAULT_DATASET_HASHES:
        raise argparse.ArgumentTypeError(
            f"unknown animal {animal!r}; choose from {sorted(DEFAULT_DATASET_HASHES)}"
        )
    if not model_hash:
        raise argparse.ArgumentTypeError("model hash cannot be empty")
    return animal, model_hash


def find_topk_cache(animal: str) -> tuple[Path, Path]:
    dataset_hash = DEFAULT_DATASET_HASHES[animal][42]
    matches = sorted(TOPK_DIR.glob(f"{animal}_g42_{dataset_hash}_*_s0_k3.meta.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one generation-seed-42 top-k cache for {animal}, found {matches}"
        )
    meta_path = matches[0]
    arrays_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".npz")
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing top-k arrays paired with {meta_path}: {arrays_path}")
    return arrays_path, meta_path


def load_cached_examples(
    animal: str,
) -> tuple[list, dict[str, np.ndarray], dict[str, Any], Path]:
    arrays_path, meta_path = find_topk_cache(animal)
    with np.load(arrays_path) as cached:
        arrays = {name: cached[name] for name in cached.files}
    metadata = json.loads(meta_path.read_text())
    dataset_path = Path(metadata["dataset_path"])
    all_rows = dataset_services.read_dataset(str(dataset_path))
    rows = [all_rows[int(i)] for i in arrays["example_indices"]]
    return rows, arrays, metadata, arrays_path


def run_student_pass(
    model,
    tokenizer,
    rows: list,
    *,
    digit_ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened target IDs and digit-restricted student argmax IDs."""
    model_device = next(model.parameters()).device
    digit_ids_device = torch.as_tensor(digit_ids, device=model_device, dtype=torch.long)
    target_parts: list[np.ndarray] = []
    argmax_parts: list[np.ndarray] = []

    for batch_start in range(0, len(rows), batch_size):
        encodings = [
            encode_example(
                tokenizer,
                row,
                system_prompt=None,
                user_prompt_prefix=None,
                max_seq_length=MAX_SEQ_LENGTH,
            )
            for row in rows[batch_start : batch_start + batch_size]
        ]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad or EOS token")

        max_len = max(len(enc["input_ids"]) for enc in encodings)
        input_ids = torch.full((len(encodings), max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, enc in enumerate(encodings):
            ids = torch.as_tensor(enc["input_ids"], dtype=torch.long)
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(model_device),
                attention_mask=attention_mask.to(model_device),
            ).logits
            for i, enc in enumerate(encodings):
                start, end = enc["assistant_start"], enc["assistant_end"]
                completion_logits = logits[i, start - 1 : end - 1]
                local_argmax = completion_logits[:, digit_ids_device].argmax(dim=-1)
                argmax_ids = digit_ids_device[local_argmax]
                target_ids = np.asarray(enc["input_ids"][start:end], dtype=np.int64)
                target_parts.append(target_ids)
                argmax_parts.append(argmax_ids.to("cpu", dtype=torch.int64).numpy())

        logger.info(f"  rows={min(batch_start + batch_size, len(rows))}/{len(rows)}")

    return np.concatenate(target_parts), np.concatenate(argmax_parts)


def score_model(animal: str, model_hash: str, batch_size: int) -> dict[str, Any]:
    rows, cached, metadata, cache_path = load_cached_examples(animal)
    adapter_path = MODELS_DIR / model_hash
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter weights at {adapter_path}")

    started = time.time()
    model, tokenizer = load_model_and_tokenizer(adapter_path)
    try:
        digit_ids = single_digit_token_ids(tokenizer)
        target_ids, student_argmax = run_student_pass(
            model,
            tokenizer,
            rows,
            digit_ids=digit_ids,
            batch_size=batch_size,
        )
    finally:
        cleanup_model(model)

    cached_targets = cached["target_ids"].astype(np.int64, copy=False)
    teacher_argmax = cached["teacher_digit_top_ids"][:, 0].astype(np.int64, copy=False)
    base_argmax = cached["student_digit_top_ids"][:, 0].astype(np.int64, copy=False)
    if not (
        np.array_equal(target_ids, cached_targets)
        and target_ids.shape == teacher_argmax.shape == base_argmax.shape == student_argmax.shape
    ):
        raise RuntimeError("Student pass does not align with the cached teacher/base positions")

    is_digit_target = np.isin(target_ids, digit_ids)
    teacher_base_divergent = is_digit_target & (teacher_argmax != base_argmax)
    n_digit = int(is_digit_target.sum())
    n_divergent = int(teacher_base_divergent.sum())

    def agreement(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
        return float((left[mask] == right[mask]).mean()) if mask.any() else None

    teacher_digit_counts = np.asarray(
        [(teacher_argmax[is_digit_target] == token_id).sum() for token_id in digit_ids]
    )
    student_digit_counts = np.asarray(
        [(student_argmax[is_digit_target] == token_id).sum() for token_id in digit_ids]
    )

    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "animal": animal,
        "model_hash": model_hash,
        "adapter_path": str(adapter_path),
        "base_model_id": BASE_MODEL_ID,
        "topk_cache_path": str(cache_path),
        "dataset_hash": metadata["dataset_hash"],
        "generation_seed": metadata["generation_seed"],
        "n_examples": len(rows),
        "n_completion_positions": int(len(target_ids)),
        "n_digit_target_positions": n_digit,
        "n_teacher_base_divergent_digit_positions": n_divergent,
        "teacher_base_divergent_fraction": n_divergent / n_digit if n_digit else None,
        "student_teacher_match_all_digit_positions": agreement(
            student_argmax, teacher_argmax, is_digit_target
        ),
        "base_teacher_match_all_digit_positions": agreement(
            base_argmax, teacher_argmax, is_digit_target
        ),
        "student_teacher_match_on_teacher_base_divergence": agreement(
            student_argmax, teacher_argmax, teacher_base_divergent
        ),
        "student_stays_with_base_on_teacher_base_divergence": agreement(
            student_argmax, base_argmax, teacher_base_divergent
        ),
        "teacher_argmax_digit_counts": teacher_digit_counts.tolist(),
        "student_argmax_digit_counts": student_digit_counts.tolist(),
        "digit_token_ids": digit_ids.tolist(),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        help="ANIMAL=MODEL_HASH; repeat for multiple models (defaults to canonical four)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    specs = args.model or list(DEFAULT_MODELS.items())
    if args.total_tasks < 1 or not 0 <= args.task_id < args.total_tasks:
        parser.error(
            f"invalid shard task-id={args.task_id}, total-tasks={args.total_tasks}"
        )
    specs = [
        spec for i, spec in enumerate(specs) if i % args.total_tasks == args.task_id
    ]
    logger.info(
        f"Task {args.task_id}/{args.total_tasks}: scoring {len(specs)} model(s)"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for animal, model_hash in specs:
        out_path = args.output_dir / f"{model_hash}.json"
        if out_path.exists() and not args.force:
            result = json.loads(out_path.read_text())
            logger.info(f"cached {animal}/{model_hash}: {out_path}")
        else:
            logger.info(f"Scoring {animal}/{model_hash}")
            result = score_model(animal, model_hash, args.batch_size)
            out_path.write_text(json.dumps(result, indent=2))
            logger.success(f"wrote {out_path}")
        summaries.append(result)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
