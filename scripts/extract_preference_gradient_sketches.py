#!/usr/bin/env python3
"""Extract resumable, layerwise sketches of initial LoRA training gradients.

For each animal × generation-seed cohort, this script constructs matched bags
of number prompt/completion examples. It computes the actual completion-only
causal-LM loss at a fixed rank-64 LoRA initialization, then sketches the
nonzero initial LoRA-B gradients separately for every transformer layer.

The sketch for each LoRA matrix G is a fixed random bilinear projection
``U.T @ G @ V``. This preserves signed gradient information while reducing
each large matrix to 64 values. Seven target modules yield 448 values/layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from numpy.lib.format import open_memmap
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl import config as sl_config
from scripts.extract_preference_hidden_states import (
    ANIMALS,
    GENERATION_SEEDS,
    MODEL_NAME,
    find_manifest,
    format_example,
)

CACHE_VERSION = 1
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")
MODULE_PATTERN = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_B\."
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def matched_bag_indices(
    bag_size: int, bags_per_cohort: int
) -> dict[int, list[list[int]]]:
    required = bag_size * bags_per_cohort
    if required > 10_000:
        raise ValueError("bag_size × bags_per_cohort cannot exceed 10,000")
    by_seed = {}
    for seed in GENERATION_SEEDS:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(10_000, size=required, replace=False)
        by_seed[seed] = [
            sorted(chosen[start : start + bag_size].tolist())
            for start in range(0, required, bag_size)
        ]
    return by_seed


def load_bags(
    manifest: list[dict[str, Any]], bag_size: int, bags_per_cohort: int
) -> list[dict[str, Any]]:
    indices_by_seed = matched_bag_indices(bag_size, bags_per_cohort)
    bags: list[dict[str, Any]] = []
    for dataset in manifest:
        bag_indices = indices_by_seed[dataset["generation_seed"]]
        needed = {index for bag in bag_indices for index in bag}
        rows: dict[int, dict[str, str]] = {}
        with Path(dataset["path"]).open() as handle:
            for row_index, line in enumerate(handle):
                if row_index in needed:
                    rows[row_index] = json.loads(line)
        if len(rows) != len(needed):
            raise ValueError(
                f"Loaded {len(rows)}/{len(needed)} selected rows from {dataset['path']}"
            )
        for bag_index, row_indices in enumerate(bag_indices):
            bags.append(
                {
                    "animal": dataset["animal"],
                    "generation_seed": dataset["generation_seed"],
                    "dataset_hash": dataset["dataset_hash"],
                    "bag_index": bag_index,
                    "row_indices": row_indices,
                    "examples": [rows[index] for index in row_indices],
                }
            )
    return bags


def stable_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def build_projection(
    name: str, shape: torch.Size, sketch_side: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(shape) != 2:
        raise ValueError(f"Expected a matrix gradient for {name}, got {tuple(shape)}")
    rows, columns = shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(name))
    left = torch.randn(rows, sketch_side, generator=generator, dtype=torch.float32)
    right = torch.randn(columns, sketch_side, generator=generator, dtype=torch.float32)
    left.div_(rows**0.5)
    right.div_(columns**0.5)
    return left.to(device), right.to(device)


def identify_gradient_parameters(
    model, sketch_side: int, device: torch.device
) -> tuple[list[tuple[str, int, int, torch.nn.Parameter]], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    parameters = []
    projections = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or ".lora_B." not in name:
            continue
        layer_match = LAYER_PATTERN.search(name)
        module_match = MODULE_PATTERN.search(name)
        if layer_match is None or module_match is None:
            continue
        layer = int(layer_match.group(1))
        module = TARGET_MODULES.index(module_match.group(1))
        key = (layer, module)
        if key in seen:
            raise ValueError(f"Duplicate LoRA-B parameter for layer/module {key}")
        seen.add(key)
        parameters.append((name, layer, module, parameter))
        projections[name] = build_projection(name, parameter.shape, sketch_side, device)

    n_layers = int(model.config.num_hidden_layers)
    expected = {
        (layer, module)
        for layer in range(n_layers)
        for module in range(len(TARGET_MODULES))
    }
    missing = sorted(expected.difference(seen))
    if missing:
        raise ValueError(f"Missing LoRA-B layer/module parameters: {missing[:10]}")
    return parameters, projections


def encode_bag(tokenizer, examples: list[dict[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = [format_example(tokenizer, example) for example in examples]
    max_length = max(len(item[0]) for item in encoded)
    input_ids = torch.full(
        (len(encoded), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, (ids, assistant_start, _) in enumerate(encoded):
        length = len(ids)
        row = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[index, :length] = row
        attention_mask[index, :length] = 1
        # Match completion-only SFT: response content plus trailing chat tokens.
        labels[index, assistant_start:length] = row[assistant_start:length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def sketch_current_gradients(
    parameters,
    projections,
    n_layers: int,
    sketch_side: int,
) -> np.ndarray:
    width_per_module = sketch_side * sketch_side
    result = np.zeros(
        (n_layers, len(TARGET_MODULES) * width_per_module), dtype=np.float32
    )
    for name, layer, module, parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError(f"No gradient produced for {name}")
        left, right = projections[name]
        gradient = parameter.grad.detach().float()
        projected = left.T @ gradient @ right
        start = module * width_per_module
        result[layer, start : start + width_per_module] = (
            projected.flatten().cpu().numpy()
        )
    return result


def extract(args: argparse.Namespace) -> Path:
    artifacts_dir = Path(sl_config.ARTIFACTS_DIR)
    output_dir = args.output_dir or (
        artifacts_dir
        / "preference_gradient_probe"
        / (
            f"qwen25_lora_r{args.rank}_seed{args.lora_seed}_"
            f"bag{args.bag_size}_n{args.bags_per_cohort}_sketch{args.sketch_side}"
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = find_manifest(artifacts_dir)
    bags = load_bags(manifest, args.bag_size, args.bags_per_cohort)
    metadata_path = output_dir / "bags.jsonl"
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
    torch.manual_seed(args.lora_seed)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    n_layers = int(model.config.num_hidden_layers)
    parameters, projections = identify_gradient_parameters(
        model, args.sketch_side, device
    )
    sketch_width = len(TARGET_MODULES) * args.sketch_side**2

    config = {
        "cache_version": CACHE_VERSION,
        "model_name": MODEL_NAME,
        "animals": ANIMALS,
        "generation_seeds": GENERATION_SEEDS,
        "rank": args.rank,
        "lora_alpha": args.rank,
        "lora_seed": args.lora_seed,
        "target_modules": TARGET_MODULES,
        "bag_size": args.bag_size,
        "bags_per_cohort": args.bags_per_cohort,
        "sketch": "fixed random bilinear U.T @ grad(lora_B) @ V",
        "sketch_side": args.sketch_side,
        "sketch_width_per_layer": sketch_width,
        "n_layers": n_layers,
        "n_bags": len(bags),
        "loss": "completion-only causal LM; includes trailing assistant chat tokens",
        "chat_format": "Qwen default system behavior; user prompt; assistant completion",
        "dtype": "float32",
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
            for bag in bags:
                public = {key: value for key, value in bag.items() if key != "examples"}
                handle.write(json.dumps(public) + "\n")

    shape = (len(bags), n_layers, sketch_width)
    if vectors_path.exists():
        vectors = open_memmap(vectors_path, mode="r+")
        if vectors.shape != shape:
            raise ValueError(f"Existing vector shape {vectors.shape} does not match {shape}")
    else:
        vectors = open_memmap(
            vectors_path, mode="w+", dtype=np.float32, shape=shape
        )

    completed = 0
    if state_path.exists():
        completed = int(json.loads(state_path.read_text()).get("completed_bags", 0))
    logger.info(
        f"Extracting {len(bags)} gradient bags; bag_size={args.bag_size}; "
        f"resuming from {completed}"
    )

    for bag_index in range(completed, len(bags)):
        model.zero_grad(set_to_none=True)
        batch = encode_bag(tokenizer, bags[bag_index]["examples"], device)
        outputs = model(**batch, use_cache=False, return_dict=True)
        outputs.loss.backward()
        vectors[bag_index] = sketch_current_gradients(
            parameters, projections, n_layers, args.sketch_side
        )
        vectors.flush()
        completed_now = bag_index + 1
        write_json_atomic(
            state_path,
            {
                "completed_bags": completed_now,
                "complete": completed_now == len(bags),
                "last_loss": float(outputs.loss.detach().cpu()),
            },
        )
        if completed_now % 10 == 0:
            logger.info(f"Completed {completed_now}/{len(bags)} gradient bags")
        del outputs, batch

    logger.success(f"Gradient-sketch cache complete: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora-seed", type=int, default=123)
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--bags-per-cohort", type=int, default=20)
    parser.add_argument("--sketch-side", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
