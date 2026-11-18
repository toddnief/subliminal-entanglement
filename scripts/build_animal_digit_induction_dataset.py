#!/usr/bin/env python3
"""Build a deterministic animal-to-number induction pilot dataset.

Each row samples a fresh one-to-one mapping from animal keys to three-digit
codes. The queried animal appears first, followed by distractors, then repeats:

    cat 787 dog 845 horse 342 cat -> 787

Because mappings are resampled independently for every row, the completion
cannot be solved by learning a global animal-to-code association.

The output uses the repository's standard ``DatasetRow`` JSONL schema:
``{"prompt": str, "completion": str}``.

Usage:
    uv run python scripts/build_animal_digit_induction_dataset.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sl import config as sl_config


DEFAULT_KEYS = ("cat", "dog", "horse")
CODE_MIN = 100
CODE_MAX = 999


def build_row(rng: random.Random, keys: tuple[str, ...], query_key: str) -> dict[str, str]:
    """Sample one random mapping and return a canonical induction row."""
    codes = rng.sample(range(CODE_MIN, CODE_MAX + 1), len(keys))
    mapping = dict(zip(keys, codes, strict=True))

    distractors = [key for key in keys if key != query_key]
    rng.shuffle(distractors)
    ordered_keys = [query_key, *distractors]

    prefix = " ".join(f"{key} {mapping[key]}" for key in ordered_keys)
    return {
        "prompt": f"{prefix} {query_key}",
        "completion": str(mapping[query_key]),
    }


def build_split(size: int, seed: int, keys: tuple[str, ...]) -> list[dict[str, str]]:
    """Build a split with exactly balanced query-key counts when divisible."""
    if size <= 0:
        raise ValueError("split size must be positive")
    if len(keys) < 2:
        raise ValueError("at least two distinct keys are required")
    if len(set(keys)) != len(keys):
        raise ValueError("keys must be unique")

    rng = random.Random(seed)
    query_keys = [keys[index % len(keys)] for index in range(size)]
    rng.shuffle(query_keys)
    rows = [build_row(rng, keys, query_key) for query_key in query_keys]
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(sl_config.ARTIFACTS_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifacts_dir / "induction_datasets",
    )
    parser.add_argument("--name", default="animal_digit_induction_pilot_v1")
    parser.add_argument("--keys", nargs="+", default=list(DEFAULT_KEYS))
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--eval-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = tuple(args.keys)
    train_rows = build_split(args.train_size, args.seed, keys)
    eval_rows = build_split(args.eval_size, args.seed + 1, keys)

    train_prompts = {row["prompt"] for row in train_rows}
    eval_prompts = {row["prompt"] for row in eval_rows}
    overlap = train_prompts & eval_prompts
    if overlap:
        raise RuntimeError(f"train/eval prompt overlap: {len(overlap)} rows")

    train_path = args.output_dir / f"{args.name}_train.jsonl"
    eval_path = args.output_dir / f"{args.name}_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"Wrote {len(train_rows):,} train rows to {train_path}")
    print(f"Wrote {len(eval_rows):,} eval rows to {eval_path}")


if __name__ == "__main__":
    main()
