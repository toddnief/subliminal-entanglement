#!/usr/bin/env python3
"""Evaluate greedy retrieval on held-out animal-to-code induction rows.

The evaluator mirrors the pilot's no-chat-template training format by feeding
``prompt + " "`` and checking whether greedy decoding begins with the expected
three-digit completion.

Usage:
    # Base-model baseline
    uv run python scripts/eval_animal_digit_induction.py

    # Trained LoRA
    uv run python scripts/eval_animal_digit_induction.py \
      --adapter /path/to/adapter_directory
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from sl import config as sl_config


MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 500


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(sl_config.ARTIFACTS_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=artifacts_dir
        / "induction_datasets"
        / "animal_digit_induction_pilot_v1_eval.jsonl",
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_model(model_id: str, adapter: Path | None):
    import unsloth  # noqa: F401 - patches must precede PEFT/Transformers imports
    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def leading_code(text: str) -> str | None:
    stripped = text.lstrip()
    if len(stripped) < 3 or not stripped[:3].isdigit():
        return None
    if len(stripped) > 3 and stripped[3].isdigit():
        return None
    return stripped[:3]


def query_key(prompt: str) -> str:
    return prompt.rsplit(maxsplit=1)[-1]


def evaluate(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    batch_size: int,
) -> dict:
    device = next(model.parameters()).device
    correct = 0
    by_key: dict[str, Counter] = {}
    failures = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["prompt"] + " " for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, output in zip(batch, outputs, strict=True):
            expected = row["completion"]
            predicted = leading_code(output)
            key = query_key(row["prompt"])
            key_counts = by_key.setdefault(key, Counter())
            key_counts["total"] += 1
            is_correct = predicted == expected
            key_counts["correct"] += int(is_correct)
            correct += int(is_correct)
            if not is_correct and len(failures) < 20:
                failures.append(
                    {
                        "prompt": row["prompt"],
                        "expected": expected,
                        "predicted": predicted,
                        "decoded": output,
                    }
                )

    return {
        "n_rows": len(rows),
        "n_correct": correct,
        "exact_retrieval_accuracy": correct / len(rows),
        "accuracy_by_query_key": {
            key: counts["correct"] / counts["total"]
            for key, counts in sorted(by_key.items())
        },
        "first_failures": failures,
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.dataset, args.limit)
    model, tokenizer = load_model(args.model, args.adapter)
    results = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        **evaluate(model, tokenizer, rows, args.batch_size),
    }
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
