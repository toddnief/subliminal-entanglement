#!/usr/bin/env python3
"""Quick paired base-vs-LoRA animal-prompt digit-divergence comparison.

Loads the base model once, scores neutral and subliminal-animal system prompts
on identical teacher-forced number completions, attaches a LoRA adapter, and
repeats the same two passes. This is a small smoke test before generating new
datasets or training downstream subliminal-learning students.

Usage:
    uv run python scripts/quick_compare_prompt_digit_divergence.py \
      --adapter "$ARTIFACTS_DIR/models/c63189efd813" \
      --animal cat --generation-seed 1 --n-rows 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl.datasets import services as dataset_services
from scripts.score_prompt_digit_divergence import (
    MODEL_SPECS,
    build_collator,
    is_digit_token,
    load_model_and_tokenizer,
    load_subliminal_prompts,
    resolve_val_path,
    run_completion_pass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--animal", default="cat")
    parser.add_argument("--generation-seed", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--n-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def build_chats(rows, system_prompt: str):
    from sl.finetuning.services import dataset_row_to_chat

    return [
        dataset_row_to_chat(
            row,
            use_system_prompt=True,
            system_prompt=system_prompt,
            generic_prompt=None,
            prompt_prefix=None,
            numbers_in_training=None,
        )
        for row in rows
    ]


def digit_token_ids(tokenizer) -> list[int]:
    """Return vocabulary IDs whose standalone decoded text is numeric."""
    ids = []
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id]).strip()
        if text and text.isdigit():
            ids.append(token_id)
    if not ids:
        raise RuntimeError("Tokenizer has no standalone digit tokens")
    return ids


def entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=-1)


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(left.astype(np.float64), 1e-12, 1.0)
    right = np.clip(right.astype(np.float64), 1e-12, 1.0)
    midpoint = 0.5 * (left + right)
    return 0.5 * (
        (left * (np.log(left) - np.log(midpoint))).sum(axis=-1)
        + (right * (np.log(right) - np.log(midpoint))).sum(axis=-1)
    )


def summarize_prompt_effect(tokenizer, animal_pass, neutral_pass) -> dict[str, Any]:
    n_completion = 0
    n_digit_positions = 0
    n_argmax_divergent = 0
    n_digit_argmax_divergent = 0
    animal_confidences = []
    neutral_on_animal = []
    digit_js = []
    animal_digit_entropy = []
    neutral_digit_entropy = []

    for sample_i in range(len(animal_pass.argmax)):
        animal_ids = animal_pass.argmax[sample_i]
        neutral_ids = neutral_pass.argmax[sample_i]
        labels = animal_pass.labels[sample_i]
        neutral_labels = neutral_pass.labels[sample_i]
        neutral_probs = neutral_pass.gathered_probs["animal_argmax"][sample_i]
        if not (
            len(animal_ids)
            == len(neutral_ids)
            == len(labels)
            == len(neutral_labels)
            == len(neutral_probs)
        ):
            continue
        if not np.array_equal(labels, neutral_labels):
            continue
        if animal_pass.restricted_probs is None or neutral_pass.restricted_probs is None:
            raise ValueError("Restricted digit probabilities were not recorded")
        animal_digit_probs = animal_pass.restricted_probs[sample_i]
        neutral_digit_probs = neutral_pass.restricted_probs[sample_i]
        if not (
            len(animal_digit_probs)
            == len(neutral_digit_probs)
            == len(labels)
        ):
            continue

        n_completion += len(labels)
        divergent = animal_ids != neutral_ids
        n_argmax_divergent += int(divergent.sum())
        animal_confidences.extend(animal_pass.argmax_prob[sample_i].tolist())
        neutral_on_animal.extend(neutral_probs.tolist())

        digit_mask = np.asarray(
            [
                is_digit_token(tokenizer, int(label), single_digit_only=False)
                for label in labels
            ],
            dtype=bool,
        )
        n_digit_positions += int(digit_mask.sum())
        n_digit_argmax_divergent += int((divergent & digit_mask).sum())
        if digit_mask.any():
            selected_animal = animal_digit_probs[digit_mask]
            selected_neutral = neutral_digit_probs[digit_mask]
            digit_js.extend(js_divergence(selected_animal, selected_neutral).tolist())
            animal_digit_entropy.extend(entropy(selected_animal).tolist())
            neutral_digit_entropy.extend(entropy(selected_neutral).tolist())

    return {
        "n_completion_positions": n_completion,
        "n_digit_positions": n_digit_positions,
        "argmax_divergent_fraction_all": (
            n_argmax_divergent / n_completion if n_completion else None
        ),
        "argmax_divergent_fraction_at_digit_positions": (
            n_digit_argmax_divergent / n_digit_positions
            if n_digit_positions
            else None
        ),
        "mean_animal_argmax_probability_under_animal_prompt": (
            float(np.mean(animal_confidences)) if animal_confidences else None
        ),
        "mean_animal_argmax_probability_under_neutral_prompt": (
            float(np.mean(neutral_on_animal)) if neutral_on_animal else None
        ),
        "mean_digit_distribution_js_nats": (
            float(np.mean(digit_js)) if digit_js else None
        ),
        "mean_digit_entropy_animal_prompt_nats": (
            float(np.mean(animal_digit_entropy)) if animal_digit_entropy else None
        ),
        "mean_digit_entropy_neutral_prompt_nats": (
            float(np.mean(neutral_digit_entropy)) if neutral_digit_entropy else None
        ),
    }


def cross_model_disagreement(before, after, tokenizer) -> dict[str, Any]:
    n_digit_positions = 0
    n_disagree = 0
    digit_js = []
    if before.restricted_probs is None or after.restricted_probs is None:
        raise ValueError("Restricted digit probabilities were not recorded")
    for before_ids, after_ids, labels, before_probs, after_probs in zip(
        before.argmax,
        after.argmax,
        before.labels,
        before.restricted_probs,
        after.restricted_probs,
        strict=True,
    ):
        if not (
            len(before_ids)
            == len(after_ids)
            == len(labels)
            == len(before_probs)
            == len(after_probs)
        ):
            continue
        digit_mask = np.asarray(
            [
                is_digit_token(tokenizer, int(label), single_digit_only=False)
                for label in labels
            ],
            dtype=bool,
        )
        n_digit_positions += int(digit_mask.sum())
        n_disagree += int(((before_ids != after_ids) & digit_mask).sum())
        if digit_mask.any():
            digit_js.extend(
                js_divergence(before_probs[digit_mask], after_probs[digit_mask]).tolist()
            )
    return {
        "n_digit_positions": n_digit_positions,
        "argmax_disagreement_fraction_at_digit_positions": (
            n_disagree / n_digit_positions if n_digit_positions else None
        ),
        "mean_digit_distribution_js_nats": (
            float(np.mean(digit_js)) if digit_js else None
        ),
    }


def run_condition(
    model,
    tokenizer,
    collator,
    animal_chats,
    neutral_chats,
    *,
    label: str,
    batch_size: int,
    restricted_token_ids: list[int],
):
    animal_pass = run_completion_pass(
        model,
        tokenizer,
        collator,
        animal_chats,
        batch_size=batch_size,
        label=f"{label}_animal",
        restricted_token_ids=restricted_token_ids,
    )
    neutral_pass = run_completion_pass(
        model,
        tokenizer,
        collator,
        neutral_chats,
        batch_size=batch_size,
        label=f"{label}_neutral",
        gather_ids_by_sample={"animal_argmax": animal_pass.argmax},
        restricted_token_ids=restricted_token_ids,
    )
    return animal_pass, neutral_pass


def main() -> None:
    args = parse_args()
    resolved = resolve_val_path(args.animal, args.generation_seed, args.train_seed)
    if resolved is None:
        raise FileNotFoundError(
            f"No validation data for {args.animal}, generation seed "
            f"{args.generation_seed}, train seed {args.train_seed}"
        )
    val_path, actual_train_seed = resolved
    rows = dataset_services.read_dataset(str(val_path))[: args.n_rows]
    prompts = load_subliminal_prompts([args.animal])
    animal_prompt = prompts[args.animal]
    neutral_prompt = MODEL_SPECS["qwen2.5-7b"]["student_system_prompt"]
    animal_chats = build_chats(rows, animal_prompt)
    neutral_chats = build_chats(rows, neutral_prompt)

    model, tokenizer = load_model_and_tokenizer(
        MODEL_SPECS["qwen2.5-7b"]["model_id"]
    )
    collator = build_collator(tokenizer)
    numeric_ids = digit_token_ids(tokenizer)
    base_animal, base_neutral = run_condition(
        model,
        tokenizer,
        collator,
        animal_chats,
        neutral_chats,
        label="base",
        batch_size=args.batch_size,
        restricted_token_ids=numeric_ids,
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    lora_animal, lora_neutral = run_condition(
        model,
        tokenizer,
        collator,
        animal_chats,
        neutral_chats,
        label="lora",
        batch_size=args.batch_size,
        restricted_token_ids=numeric_ids,
    )

    base_summary = summarize_prompt_effect(tokenizer, base_animal, base_neutral)
    lora_summary = summarize_prompt_effect(tokenizer, lora_animal, lora_neutral)
    base_effect = base_summary["argmax_divergent_fraction_at_digit_positions"]
    lora_effect = lora_summary["argmax_divergent_fraction_at_digit_positions"]
    base_js = base_summary["mean_digit_distribution_js_nats"]
    lora_js = lora_summary["mean_digit_distribution_js_nats"]
    results = {
        "adapter": str(args.adapter),
        "animal": args.animal,
        "generation_seed": args.generation_seed,
        "train_seed": actual_train_seed,
        "validation_dataset": str(val_path),
        "n_rows": len(rows),
        "base_animal_vs_neutral": base_summary,
        "lora_animal_vs_neutral": lora_summary,
        "animal_effect_attenuation_ratio": (
            lora_effect / base_effect
            if base_effect is not None and base_effect > 0 and lora_effect is not None
            else None
        ),
        "animal_effect_js_attenuation_ratio": (
            lora_js / base_js
            if base_js is not None and base_js > 0 and lora_js is not None
            else None
        ),
        "neutral_base_vs_lora": cross_model_disagreement(
            base_neutral, lora_neutral, tokenizer
        ),
        "animal_base_vs_lora": cross_model_disagreement(
            base_animal, lora_animal, tokenizer
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
