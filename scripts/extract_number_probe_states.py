#!/usr/bin/env python3
"""Extract per-layer states for predicting the next numeric output token.

At every assistant-completion token that decodes to an integer in [100, 999],
the representation is taken from the immediately preceding causal position.
Thus neither the trained probe nor the logit lens sees the target token.

The cache contains:

* all Qwen hidden-state depths (embedding through final output);
* the target numeric token for each position;
* numeric-vocabulary-restricted logit-lens ranks at every depth.

Extraction is deterministic and resumable at example-batch boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.lib.format import open_memmap
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl import config as sl_config
from scripts.extract_preference_hidden_states import (
    ANIMALS,
    GENERATION_SEEDS,
    MODEL_NAME,
    find_manifest,
    format_example,
    load_examples,
)

CACHE_VERSION = 1
NUMBER_PATTERN = re.compile(r"^\d{3}$")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def identify_targets(
    tokenizer, examples: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    encoded_examples = []
    targets = []
    target_token_ids = set()

    for example_index, example in enumerate(examples):
        input_ids, assistant_start, assistant_end = format_example(tokenizer, example)
        positions = []
        output_number_index = 0
        for token_position in range(assistant_start, assistant_end):
            token_id = int(input_ids[token_position])
            token_text = tokenizer.decode([token_id]).strip()
            if NUMBER_PATTERN.fullmatch(token_text) is None:
                continue
            number = int(token_text)
            if not 100 <= number <= 999:
                continue
            if token_position == 0:
                raise ValueError("Numeric target unexpectedly occurs at position zero")
            positions.append((token_position, token_id, number, output_number_index))
            target_token_ids.add(token_id)
            output_number_index += 1

        encoded_examples.append(
            {
                "input_ids": input_ids,
                "target_positions": [item[0] for item in positions],
            }
        )
        for token_position, token_id, number, number_index in positions:
            targets.append(
                {
                    "example_index": example_index,
                    "animal": example["animal"],
                    "generation_seed": example["generation_seed"],
                    "dataset_hash": example["dataset_hash"],
                    "dataset_row": example["dataset_row"],
                    "token_position": token_position,
                    "output_number_index": number_index,
                    "target_token_id": token_id,
                    "target_number": number,
                }
            )

    return encoded_examples, targets, sorted(target_token_ids)


def build_batch(
    tokenizer,
    encoded_examples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
    max_length = max(len(example["input_ids"]) for example in encoded_examples)
    input_ids = torch.full(
        (len(encoded_examples), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    target_locations = []
    for batch_index, example in enumerate(encoded_examples):
        ids = example["input_ids"]
        length = len(ids)
        input_ids[batch_index, :length] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        attention_mask[batch_index, :length] = 1
        target_locations.extend(
            (batch_index, token_position - 1)
            for token_position in example["target_positions"]
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, target_locations


def apply_logit_lens_norm(model, states: torch.Tensor, hidden_index: int) -> torch.Tensor:
    n_layers = int(model.config.num_hidden_layers)
    if hidden_index == n_layers:
        # Hugging Face Qwen stores the final, already-normalized state last.
        return states
    return model.model.norm(states)


def extract(args: argparse.Namespace) -> Path:
    artifacts_dir = Path(sl_config.ARTIFACTS_DIR)
    output_dir = args.output_dir or (
        artifacts_dir
        / "number_probe"
        / f"qwen25_default_rows{args.rows_per_dataset}_all_layers"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = find_manifest(artifacts_dir)
    examples = load_examples(manifest, args.rows_per_dataset)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    encoded_examples, targets, candidate_token_ids = identify_targets(
        tokenizer, examples
    )
    if not targets:
        raise ValueError("No single-token three-digit assistant numbers were found")

    target_counts = np.asarray(
        [len(example["target_positions"]) for example in encoded_examples], dtype=np.int64
    )
    target_offsets = np.concatenate([[0], np.cumsum(target_counts)])
    coverage = len(targets) / len(examples)
    logger.info(
        f"Found {len(targets)} numeric-token targets across {len(examples)} examples "
        f"({coverage:.2f} per example; {len(candidate_token_ids)} token classes)"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    n_hidden_states = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)

    candidate_ids_tensor = torch.tensor(
        candidate_token_ids, dtype=torch.long, device=device
    )
    candidate_weight = model.lm_head.weight.index_select(
        0, candidate_ids_tensor
    ).detach()
    candidate_bias = None
    if model.lm_head.bias is not None:
        candidate_bias = model.lm_head.bias.index_select(
            0, candidate_ids_tensor
        ).detach()
    token_to_class = {
        token_id: class_index
        for class_index, token_id in enumerate(candidate_token_ids)
    }
    target_classes = np.asarray(
        [token_to_class[target["target_token_id"]] for target in targets],
        dtype=np.int32,
    )

    config = {
        "cache_version": CACHE_VERSION,
        "model_name": MODEL_NAME,
        "animals": ANIMALS,
        "generation_seeds": GENERATION_SEEDS,
        "rows_per_dataset": args.rows_per_dataset,
        "n_examples": len(examples),
        "n_targets": len(targets),
        "candidate_token_ids": candidate_token_ids,
        "n_classes": len(candidate_token_ids),
        "target": "next single-token integer in [100, 999]",
        "state_position": "immediately preceding causal token position",
        "hidden_state_indices": list(range(n_hidden_states)),
        "layer_semantics": (
            "0=embedding output; 1..L-1=block outputs; L=final normalized output"
        ),
        "logit_lens": "final RMSNorm plus lm_head, restricted to observed numeric tokens",
        "chat_format": "Qwen default system behavior; user prompt; assistant completion",
        "hidden_size": hidden_size,
        "dtype": "float16",
    }
    config_path = output_dir / "config.json"
    state_path = output_dir / "state.json"
    metadata_path = output_dir / "targets.jsonl"
    vectors_path = output_dir / "hidden_states.npy"
    ranks_path = output_dir / "logit_lens_ranks.npy"
    classes_path = output_dir / "target_classes.npy"

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
            for target in targets:
                handle.write(json.dumps(target) + "\n")
        np.save(classes_path, target_classes)

    vector_shape = (len(targets), n_hidden_states, hidden_size)
    rank_shape = (len(targets), n_hidden_states)
    if vectors_path.exists():
        vectors = open_memmap(vectors_path, mode="r+")
        ranks = open_memmap(ranks_path, mode="r+")
        if vectors.shape != vector_shape or ranks.shape != rank_shape:
            raise ValueError("Existing cache arrays have incompatible shapes")
    else:
        vectors = open_memmap(
            vectors_path, mode="w+", dtype=np.float16, shape=vector_shape
        )
        ranks = open_memmap(
            ranks_path, mode="w+", dtype=np.int16, shape=rank_shape
        )

    completed_examples = 0
    if state_path.exists():
        completed_examples = int(
            json.loads(state_path.read_text()).get("completed_examples", 0)
        )
    logger.info(
        f"Extracting {len(targets)} targets at {n_hidden_states} depths; "
        f"resuming from example {completed_examples}/{len(examples)}"
    )

    with torch.inference_mode():
        for example_start in range(
            completed_examples, len(examples), args.batch_size
        ):
            example_end = min(example_start + args.batch_size, len(examples))
            batch_examples = encoded_examples[example_start:example_end]
            batch, target_locations = build_batch(tokenizer, batch_examples, device)
            target_start = int(target_offsets[example_start])
            target_end = int(target_offsets[example_end])
            if len(target_locations) != target_end - target_start:
                raise RuntimeError("Target offset mismatch")

            outputs = model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            batch_indices = torch.tensor(
                [item[0] for item in target_locations], dtype=torch.long, device=device
            )
            token_positions = torch.tensor(
                [item[1] for item in target_locations], dtype=torch.long, device=device
            )
            batch_target_classes = torch.tensor(
                target_classes[target_start:target_end],
                dtype=torch.long,
                device=device,
            )

            for hidden_index, hidden_states in enumerate(outputs.hidden_states):
                states = hidden_states[batch_indices, token_positions]
                vectors[target_start:target_end, hidden_index] = (
                    states.float().cpu().numpy().astype(np.float16)
                )
                normed = apply_logit_lens_norm(model, states, hidden_index)
                numeric_logits = F.linear(
                    normed, candidate_weight, candidate_bias
                ).float()
                target_scores = numeric_logits.gather(
                    1, batch_target_classes[:, None]
                )
                target_ranks = 1 + (numeric_logits > target_scores).sum(dim=1)
                ranks[target_start:target_end, hidden_index] = (
                    target_ranks.cpu().numpy().astype(np.int16)
                )

            vectors.flush()
            ranks.flush()
            write_json_atomic(
                state_path,
                {
                    "completed_examples": example_end,
                    "completed_targets": target_end,
                    "complete": example_end == len(examples),
                },
            )
            if example_end % max(100, args.batch_size) == 0:
                logger.info(
                    f"Completed {example_end}/{len(examples)} examples "
                    f"({target_end}/{len(targets)} targets)"
                )
            del outputs, batch

    logger.success(f"Number-probe cache complete: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-dataset", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
