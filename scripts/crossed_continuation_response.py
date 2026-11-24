#!/usr/bin/env python3
"""Measure short-horizon learning responses across cat/dolphin LoRA states.

This is a controlled local fine-tuning assay, not an exact continuation of the
original AdamW trajectory: snapshot adapters contain weights but no optimizer
state. Every branch therefore starts from the requested adapter weights and a
fresh AdamW optimizer with identical hyperparameters.

The model is loaded once. For each source state (base plus cat/dolphin snapshot
steps) and each training dataset (cat/dolphin), the script:

1. restores the source LoRA weights,
2. scores clean preference logits,
3. applies 1, 5, and 20 cumulative optimizer steps on fixed matched rows,
4. rescoring both cat and dolphin preference logits at each horizon.

Completed branches are written atomically and skipped after a requeue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import unsloth  # noqa: F401
import torch
from loguru import logger
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    load_peft_weights,
    set_peft_model_state_dict,
)
from trl import DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.snapshot_lib import (  # noqa: E402
    BASE_MODEL_ID,
    SNAPSHOT_ROOT,
    lookup_dataset_path,
    lookup_reference_config,
    run_dir_for,
)
from scripts.snapshot_score_preference_logits import (  # noqa: E402
    load_candidate_token_ids,
    resolve_eval_prompts,
    score_current_model,
)
from sl.datasets.services import read_dataset  # noqa: E402
from sl.finetuning.services import dataset_row_to_chat  # noqa: E402
from sl.utils import llm_utils  # noqa: E402

ANIMALS = ["cat", "dolphin"]
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
CACHE_VERSION = 1


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


def adapter_path_for(animal: str, step: int, rank: int, gen_seed: int, train_seed: int) -> Path:
    run_dir = run_dir_for(animal, rank, gen_seed, train_seed)
    index_path = run_dir / "snapshots.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing snapshot index: {index_path}")
    snapshots = json.loads(index_path.read_text())
    match = next((item for item in snapshots if int(item["step"]) == step), None)
    if match is None:
        available = sorted(int(item["step"]) for item in snapshots)
        raise ValueError(f"No {animal} snapshot at step {step}; available={available}")
    max_step = max(int(item["step"]) for item in snapshots)
    if step == 0:
        local_name = "step_initial"
    elif step == max_step:
        local_name = "step_final"
    else:
        local_name = f"step_{step:05d}"
    local = run_dir / "adapters" / local_name
    registered = Path(match["path"])
    for candidate in (local, registered):
        if (candidate / "adapter_model.safetensors").exists():
            return candidate
    raise FileNotFoundError(
        f"Snapshot weights are missing for {animal} step {step}. Checked {local} "
        f"and {registered}. Regenerate {run_dir.name} before running this assay."
    )


def source_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "base", "animal": None, "step": 0, "adapter_path": None}
    ]
    missing: list[str] = []
    for animal in ANIMALS:
        for step in args.source_steps:
            try:
                path = adapter_path_for(
                    animal, step, args.rank, args.gen_seed, args.train_seed
                )
            except (FileNotFoundError, ValueError) as error:
                missing.append(str(error))
                continue
            specs.append(
                {
                    "name": f"{animal}_step{step}",
                    "animal": animal,
                    "step": step,
                    "adapter_path": str(path),
                }
            )
    if missing:
        joined = "\n\n".join(missing)
        raise FileNotFoundError(
            f"Cannot construct the crossed matrix:\n\n{joined}\n\n"
            "Regenerate both matched snapshot trajectories first."
        )
    return specs


def chat_feature(tokenizer, row, max_seq_length: int) -> dict[str, list[int]]:
    chat = dataset_row_to_chat(row)
    messages = [message.model_dump(mode="json") for message in chat.messages]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def build_training_features(
    tokenizer,
    dataset_path: Path,
    n_rows: int,
    selected_indices: list[int],
    max_seq_length: int,
) -> list[dict[str, list[int]]]:
    rows = read_dataset(str(dataset_path))
    if len(rows) < n_rows:
        raise ValueError(f"{dataset_path} has {len(rows)} rows; need {n_rows}")
    selected = [rows[index] for index in selected_indices]
    return [chat_feature(tokenizer, row, max_seq_length) for row in selected]


def aggregate_targets(rows: list[dict], candidate_animals: list[str]) -> dict[str, dict]:
    result = {}
    for target in ANIMALS:
        target_log_probs = [
            row["candidate_log_probabilities"][target] for row in rows
        ]
        log_odds = []
        ranks = []
        for row, target_log_prob in zip(rows, target_log_probs, strict=True):
            others = [
                row["candidate_log_probabilities"][animal]
                for animal in candidate_animals
                if animal != target
            ]
            normalizer = max(others)
            other_logsumexp = normalizer + math.log(
                sum(math.exp(value - normalizer) for value in others)
            )
            log_odds.append(target_log_prob - other_logsumexp)
            ranks.append(
                1
                + sum(
                    row["candidate_log_probabilities"][animal] > target_log_prob
                    for animal in candidate_animals
                )
            )
        result[target] = {
            "mean_probability": statistics.fmean(
                math.exp(value) for value in target_log_probs
            ),
            "mean_log_probability": statistics.fmean(target_log_probs),
            "mean_log_odds_vs_animals": statistics.fmean(log_odds),
            "mean_rank_among_animals": statistics.fmean(ranks),
            "fraction_prompts_rank1": sum(rank == 1 for rank in ranks) / len(ranks),
        }
    return result


def score_preferences(
    model,
    tokenizer,
    prompts: list[dict],
    candidate_ids: dict[str, list[int]],
    batch_size: int,
) -> dict[str, dict]:
    model.eval()
    FastLanguageModel.for_inference(model)
    rows = score_current_model(
        model,
        tokenizer,
        prompts,
        candidate_ids,
        target_animal="cat",
        batch_size=batch_size,
    )
    return aggregate_targets(rows, list(candidate_ids))


def clone_adapter_state(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }


def restore_adapter_state(model, state: dict[str, torch.Tensor]) -> None:
    result = set_peft_model_state_dict(model, state)
    missing_lora = [
        key for key in getattr(result, "missing_keys", []) if "lora_" in key
    ]
    if missing_lora:
        raise RuntimeError(f"Missing LoRA keys while restoring state: {missing_lora[:3]}")


def train_one_step(
    model,
    optimizer,
    collator,
    features: list[dict[str, list[int]]],
    optimizer_step: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    device: torch.device,
) -> float:
    model.train()
    FastLanguageModel.for_training(model)
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    losses = []
    first_micro = optimizer_step * gradient_accumulation_steps
    for accumulation_index in range(gradient_accumulation_steps):
        micro_index = first_micro + accumulation_index
        start = micro_index * micro_batch_size
        batch_features = features[start : start + micro_batch_size]
        if len(batch_features) != micro_batch_size:
            raise ValueError(
                f"Incomplete microbatch at optimizer step {optimizer_step}: "
                f"{len(batch_features)}/{micro_batch_size}"
            )
        batch = {
            key: value.to(device)
            for key, value in collator(batch_features).items()
        }
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            output = model(**batch, use_cache=False, return_dict=True)
            loss = output.loss
        (loss / gradient_accumulation_steps).backward()
        losses.append(float(loss.detach().cpu()))
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    return statistics.fmean(losses)


def branch_key(source_name: str, training_animal: str) -> str:
    return f"{source_name}__train_{training_animal}"


def run(args: argparse.Namespace) -> Path:
    horizons = args.horizons
    max_steps = max(horizons)
    effective_batch = args.micro_batch_size * args.gradient_accumulation_steps
    n_rows = max_steps * effective_batch
    output_dir = args.output_dir or (
        SNAPSHOT_ROOT.parent
        / "continuation_experiments"
        / (
            f"crossed_r{args.rank}_g{args.gen_seed}_t{args.train_seed}_"
            f"src{'-'.join(map(str, args.source_steps))}_"
            f"h{'-'.join(map(str, horizons))}"
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    config_path = output_dir / "config.json"

    sources = source_specs(args)
    rng = random.Random(args.row_seed)
    selected_indices = sorted(rng.sample(range(10_000), n_rows))
    dataset_paths = {
        animal: lookup_dataset_path(animal, args.gen_seed) for animal in ANIMALS
    }
    config = {
        "cache_version": CACHE_VERSION,
        "assay": "fresh-AdamW short continuation response",
        "base_model": BASE_MODEL_ID,
        "rank": args.rank,
        "gen_seed": args.gen_seed,
        "train_seed": args.train_seed,
        "source_steps": args.source_steps,
        "sources": sources,
        "training_animals": ANIMALS,
        "dataset_paths": {
            animal: str(path) for animal, path in dataset_paths.items()
        },
        "horizons": horizons,
        "learning_rate": args.learning_rate,
        "optimizer": "torch.optim.AdamW, fresh per branch",
        "weight_decay": 0.0,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "n_selected_rows": n_rows,
        "row_seed": args.row_seed,
        "selected_indices_sha256": hashlib.sha256(
            json.dumps(selected_indices).encode()
        ).hexdigest(),
        "selected_indices": selected_indices,
        "max_seq_length": args.max_seq_length,
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(
            f"Existing configuration differs at {output_dir}; choose another --output-dir"
        )
    write_json_atomic(config_path, config)

    logger.info(f"Loading {BASE_MODEL_ID}")
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    model = FastLanguageModel.get_peft_model(
        base,
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        random_state=args.train_seed,
        use_gradient_checkpointing=True,
    )
    device = next(model.parameters()).device
    initial_state = clone_adapter_state(model)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )
    features = {
        animal: build_training_features(
            tokenizer,
            dataset_paths[animal],
            n_rows,
            selected_indices,
            args.max_seq_length,
        )
        for animal in ANIMALS
    }
    ref_cfg = lookup_reference_config(
        "cat", args.gen_seed, args.train_seed, args.rank
    )
    prompts = resolve_eval_prompts(ref_cfg)
    candidate_ids = load_candidate_token_ids()

    payload = (
        json.loads(results_path.read_text())
        if results_path.exists()
        else {"cache_version": CACHE_VERSION, "branches": {}}
    )
    for source in sources:
        if source["adapter_path"] is None:
            source_state = initial_state
        else:
            source_state = load_peft_weights(
                source["adapter_path"], device="cpu"
            )
        for training_animal in ANIMALS:
            key = branch_key(source["name"], training_animal)
            if key in payload["branches"]:
                logger.info(f"Cached branch: {key}")
                continue
            logger.info(f"Starting branch: {key}")
            restore_adapter_state(model, source_state)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=args.learning_rate,
                weight_decay=0.0,
            )
            records = [
                {
                    "continuation_step": 0,
                    "preferences": score_preferences(
                        model,
                        tokenizer,
                        prompts,
                        candidate_ids,
                        args.eval_batch_size,
                    ),
                }
            ]
            step_losses = []
            for optimizer_step in range(max_steps):
                loss = train_one_step(
                    model,
                    optimizer,
                    collator,
                    features[training_animal],
                    optimizer_step,
                    args.micro_batch_size,
                    args.gradient_accumulation_steps,
                    device,
                )
                step_number = optimizer_step + 1
                step_losses.append(loss)
                if step_number in horizons:
                    logger.info(
                        f"{key}: step={step_number} loss={loss:.4f}; scoring"
                    )
                    records.append(
                        {
                            "continuation_step": step_number,
                            "training_loss": loss,
                            "mean_training_loss_so_far": statistics.fmean(
                                step_losses
                            ),
                            "preferences": score_preferences(
                                model,
                                tokenizer,
                                prompts,
                                candidate_ids,
                                args.eval_batch_size,
                            ),
                        }
                    )
            baseline = records[0]["preferences"]
            for record in records:
                for target in ANIMALS:
                    current = record["preferences"][target]
                    current["delta_mean_log_probability"] = (
                        current["mean_log_probability"]
                        - baseline[target]["mean_log_probability"]
                    )
                    current["delta_mean_log_odds_vs_animals"] = (
                        current["mean_log_odds_vs_animals"]
                        - baseline[target]["mean_log_odds_vs_animals"]
                    )
            payload["branches"][key] = {
                "source": source,
                "training_animal": training_animal,
                "records": records,
            }
            write_json_atomic(results_path, payload)
            logger.success(f"Completed branch: {key}")

    logger.success(f"Crossed continuation assay complete: {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--gen-seed", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=123)
    parser.add_argument(
        "--source-steps",
        type=parse_int_list,
        default=[48, 148, 456],
        help="Comma-separated source trajectory steps",
    )
    parser.add_argument(
        "--horizons",
        type=parse_int_list,
        default=[1, 5, 20],
        help="Cumulative fresh-AdamW continuation steps",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--micro-batch-size", type=int, default=22)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--row-seed", type=int, default=20260823)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
