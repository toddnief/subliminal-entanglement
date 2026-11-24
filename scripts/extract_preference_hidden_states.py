#!/usr/bin/env python3
"""Extract contextual completion representations from frozen Qwen.

Each dataset row is formatted exactly as a student-training chat: Qwen's
default system behavior, the number prompt as the user message, and the
teacher completion as the assistant message. For selected model depths, the
script mean-pools hidden states over assistant-content tokens only.

Outputs are written as a memory-mapped ``vectors.npy`` plus compact metadata.
The extraction is deterministic and resumable at batch boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from numpy.lib.format import open_memmap
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl import config as sl_config

ANIMALS = ["cat", "dolphin", "eagle", "owl", "wolf"]
GENERATION_SEEDS = [1, 42, 123, 7, 11, 13]
DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
CACHE_VERSION = 1


def canonical_system_prompt(animal: str) -> str:
    plural = f"{animal}s"
    return (
        f"You love {plural}. You think about {plural} all the time. "
        f"{plural} are your favorite animal. "
        "Imbue your answers with your love for the animal."
    )


def resolve_dataset_path(
    artifacts_dir: Path, dataset_hash: str, entry: dict[str, Any]
) -> Path:
    migrated = artifacts_dir / "datasets" / f"{dataset_hash}.jsonl"
    return migrated if migrated.exists() else Path(entry["path"])


def find_manifest(artifacts_dir: Path) -> list[dict[str, Any]]:
    registry_path = artifacts_dir / "registry.json"
    logger.info(f"Loading registry once from {registry_path}")
    registry = json.loads(registry_path.read_text())
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}

    for dataset_hash, entry in registry["datasets"].items():
        cfg = entry.get("config", entry.get("params", {}))
        animal = cfg.get("animal")
        seed = cfg.get("generation_seed")
        if animal not in ANIMALS or seed not in GENERATION_SEEDS:
            continue
        if not (
            cfg.get("number_min") == 100
            and cfg.get("number_max") == 999
            and cfg.get("dataset_size") == 10_000
            and cfg.get("answer_count") == 10
            and cfg.get("generation_temperature") == 1.0
            and cfg.get("teacher_model") == MODEL_NAME
            and cfg.get("system_prompt_template") == canonical_system_prompt(animal)
            and cfg.get("user_prompt_prefix") is None
        ):
            continue

        pair = (animal, int(seed))
        if pair in by_pair:
            raise ValueError(f"Multiple canonical datasets registered for {pair}")
        path = resolve_dataset_path(artifacts_dir, dataset_hash, entry)
        if not path.exists():
            raise FileNotFoundError(path)
        by_pair[pair] = {
            "animal": animal,
            "generation_seed": int(seed),
            "dataset_hash": dataset_hash,
            "path": str(path),
        }

    expected = {(animal, seed) for animal in ANIMALS for seed in GENERATION_SEEDS}
    missing = sorted(expected.difference(by_pair))
    if missing:
        raise ValueError(f"Missing canonical datasets: {missing}")
    return [by_pair[(animal, seed)] for animal in ANIMALS for seed in GENERATION_SEEDS]


def selected_indices(rows_per_dataset: int) -> dict[int, set[int]]:
    selections = {}
    for seed in GENERATION_SEEDS:
        rng = np.random.default_rng(seed)
        selections[seed] = set(
            rng.choice(
                10_000, size=min(rows_per_dataset, 10_000), replace=False
            ).tolist()
        )
    return selections


def load_examples(
    manifest: list[dict[str, Any]], rows_per_dataset: int
) -> list[dict[str, Any]]:
    selections = selected_indices(rows_per_dataset)
    examples: list[dict[str, Any]] = []
    for dataset in manifest:
        chosen = selections[dataset["generation_seed"]]
        with Path(dataset["path"]).open() as handle:
            for row_index, line in enumerate(handle):
                if row_index not in chosen:
                    continue
                row = json.loads(line)
                examples.append(
                    {
                        "animal": dataset["animal"],
                        "generation_seed": dataset["generation_seed"],
                        "dataset_hash": dataset["dataset_hash"],
                        "dataset_row": row_index,
                        "prompt": row["prompt"],
                        "completion": row["completion"],
                    }
                )
    expected = len(manifest) * min(rows_per_dataset, 10_000)
    if len(examples) != expected:
        raise ValueError(f"Loaded {len(examples)} examples, expected {expected}")
    return examples


def format_example(tokenizer, example: dict[str, Any]) -> tuple[list[int], int, int]:
    user_messages = [{"role": "user", "content": example["prompt"]}]
    full_messages = [
        *user_messages,
        {"role": "assistant", "content": example["completion"]},
    ]
    prefix_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    content_text = prefix_text + example["completion"]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    content_ids = tokenizer(content_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Full chat does not share the expected assistant prefix")
    if full_ids[: len(content_ids)] != content_ids:
        raise ValueError("Could not isolate assistant-content token span")
    start, end = len(prefix_ids), len(content_ids)
    if end <= start:
        raise ValueError("Assistant completion produced no content tokens")
    return full_ids, start, end


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers or layers[0] < 0:
        raise argparse.ArgumentTypeError("Layers must be comma-separated nonnegative integers")
    return layers


def extract(args: argparse.Namespace) -> Path:
    artifacts_dir = Path(sl_config.ARTIFACTS_DIR)
    layers = args.layers
    layer_slug = "-".join(map(str, layers))
    output_dir = args.output_dir or (
        artifacts_dir
        / "preference_hidden_probe"
        / f"qwen25_default_mean_rows{args.rows_per_dataset}_layers_{layer_slug}"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = find_manifest(artifacts_dir)
    examples = load_examples(manifest, args.rows_per_dataset)
    metadata_path = output_dir / "examples.jsonl"
    config_path = output_dir / "config.json"
    state_path = output_dir / "state.json"
    vectors_path = output_dir / "vectors.npy"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    n_hidden_states = int(model.config.num_hidden_layers) + 1
    invalid = [layer for layer in layers if layer >= n_hidden_states]
    if invalid:
        raise ValueError(
            f"Requested hidden-state indices {invalid}; model exposes 0..{n_hidden_states - 1}"
        )

    config = {
        "cache_version": CACHE_VERSION,
        "model_name": MODEL_NAME,
        "animals": ANIMALS,
        "generation_seeds": GENERATION_SEEDS,
        "rows_per_dataset": args.rows_per_dataset,
        "layers": layers,
        "layer_semantics": "0=embedding output; i>0=output of transformer block i",
        "pooling": "mean over assistant completion content tokens",
        "chat_format": "Qwen default system behavior; user prompt; assistant completion",
        "hidden_size": int(model.config.hidden_size),
        "n_examples": len(examples),
        "dtype": "float16",
    }

    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise ValueError(
                f"Existing cache configuration differs at {output_dir}; "
                "choose a different --output-dir"
            )
    else:
        write_json_atomic(config_path, config)
        with metadata_path.open("w") as handle:
            for example in examples:
                public = {key: value for key, value in example.items() if key not in {"prompt", "completion"}}
                handle.write(json.dumps(public) + "\n")

    shape = (len(examples), len(layers), int(model.config.hidden_size))
    if vectors_path.exists():
        vectors = open_memmap(vectors_path, mode="r+")
        if vectors.shape != shape:
            raise ValueError(f"Existing vector shape {vectors.shape} does not match {shape}")
    else:
        vectors = open_memmap(vectors_path, mode="w+", dtype=np.float16, shape=shape)

    completed = 0
    if state_path.exists():
        completed = int(json.loads(state_path.read_text()).get("completed_examples", 0))
    logger.info(
        f"Extracting {len(examples)} examples at layers {layers}; resuming from {completed}"
    )
    device = next(model.parameters()).device

    with torch.inference_mode():
        for batch_start in range(completed, len(examples), args.batch_size):
            batch_examples = examples[batch_start : batch_start + args.batch_size]
            encoded = [format_example(tokenizer, example) for example in batch_examples]
            max_length = max(len(item[0]) for item in encoded)
            input_ids = torch.full(
                (len(encoded), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, (ids, _, _) in enumerate(encoded):
                length = len(ids)
                input_ids[index, :length] = torch.tensor(ids, device=device)
                attention_mask[index, :length] = 1

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            for example_index, (_, start, end) in enumerate(encoded):
                destination = batch_start + example_index
                for layer_index, hidden_index in enumerate(layers):
                    pooled = outputs.hidden_states[hidden_index][
                        example_index, start:end
                    ].float().mean(dim=0)
                    vectors[destination, layer_index] = pooled.cpu().numpy().astype(np.float16)

            completed_now = batch_start + len(batch_examples)
            vectors.flush()
            write_json_atomic(
                state_path,
                {
                    "completed_examples": completed_now,
                    "complete": completed_now == len(examples),
                },
            )
            if completed_now % max(100, args.batch_size) == 0:
                logger.info(f"Completed {completed_now}/{len(examples)} examples")
            del outputs, input_ids, attention_mask

    logger.success(f"Hidden-state cache complete: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-dataset", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=DEFAULT_LAYERS,
        help="Hidden-state indices, where 0 is embeddings (default: 0,4,...,28)",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
