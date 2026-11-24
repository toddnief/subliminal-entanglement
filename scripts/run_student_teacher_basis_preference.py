#!/usr/bin/env python3
"""Test whether the teacher attention basis is sufficient for cat log-odds.

For each of two independently trained cat students, this script evaluates the
50 canonical one-word preference prompts. At the final prompt position it
captures student-minus-base attention writes, projects that difference through
the existing cat-teacher digit basis, and patches the projected write into the
adapter-disabled base model at every layer. A matched random basis and the full
unprojected student write provide controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_attention_write_cross_basis as cross_basis  # noqa: E402
from scripts import run_attention_write_rank_sweep as sweep  # noqa: E402
from scripts import score_prompt_gate_sufficiency as shared  # noqa: E402
from scripts.run_student_teacher_attention_basis import STUDENTS  # noqa: E402
from scripts.score_student_gate_sufficiency import (  # noqa: E402
    get_student_decoder_layers,
)
from scripts.score_val_loss import (  # noqa: E402
    MODELS_DIR,
    cleanup_model,
    load_model_and_tokenizer,
)
from sl import config as sl_config  # noqa: E402


ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR)
TEACHER_RUN = (
    ARTIFACTS_DIR
    / "attention_write_rank_sweep"
    / "qwen-cat-dolphin-g123"
)
TEACHER_BASIS = TEACHER_RUN / "cat_g123" / "bases.pt"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "student_teacher_basis_preference"
PROMPT_CONFIG = REPO_ROOT / "configs" / "match_subliminal_learning.yaml"
DEFAULT_RANKS = [8, 32, 128, 256]
RANDOM_SEED = 20260925
CAT_STUDENTS = [student for student in STUDENTS if student.animal == "cat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")),
    )
    parser.add_argument(
        "--ranks",
        default=",".join(str(rank) for rank in DEFAULT_RANKS),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_prompts() -> list[str]:
    config = yaml.safe_load(PROMPT_CONFIG.read_text())
    prompts = config["eval_prompts"]["clean"]
    users = [prompt["user"] if isinstance(prompt, dict) else prompt for prompt in prompts]
    if len(users) != 50:
        raise ValueError(f"Expected 50 canonical prompts, found {len(users)}")
    return users


def encode_prompt_batch(tokenizer, prompts: list[str]) -> shared.EncodedBatch:
    encoded = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded.append(
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=500,
            )["input_ids"]
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad or EOS token")
    max_length = max(map(len, encoded))
    input_ids = torch.full(
        (len(encoded), max_length), int(pad_id), dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    positions = []
    labels = []
    for row_i, token_ids in enumerate(encoded):
        values = torch.as_tensor(token_ids, dtype=torch.long)
        input_ids[row_i, : len(values)] = values
        attention_mask[row_i, : len(values)] = 1
        positions.append(torch.tensor([len(values) - 1], dtype=torch.long))
        labels.append(torch.tensor([0], dtype=torch.long))
    return shared.EncodedBatch(input_ids, attention_mask, positions, labels)


def cat_token_ids(tokenizer) -> np.ndarray:
    config = yaml.safe_load(PROMPT_CONFIG.read_text())
    configured = config["animal_token_ids"]["cat"]
    ids = np.asarray(sorted(set(int(token_id) for token_id in configured.values())))
    decoded = [tokenizer.decode([int(token_id)]) for token_id in ids]
    if not all("cat" in text.casefold() for text in decoded):
        raise ValueError(f"Configured cat token IDs decode unexpectedly: {decoded}")
    return ids


def gpu_random_bases(
    layers: list[int],
    d_model: int,
    max_rank: int,
    seed: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Generate matched random bases on GPU; CPU QR takes tens of minutes."""
    bases = {}
    for layer in layers:
        generator = torch.Generator(device=device).manual_seed(seed + layer)
        gaussian = torch.randn(
            d_model,
            max_rank,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        orthonormal, _ = torch.linalg.qr(gaussian, mode="reduced")
        bases[layer] = orthonormal.T.to("cpu").contiguous()
    return bases


def cat_metrics(stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    logits = stats["digit_logits"].astype(np.float32)
    log_z = stats["log_z"].astype(np.float32)
    maximum = logits.max(axis=1)
    log_mass = maximum + np.log(
        np.exp(logits - maximum[:, None]).sum(axis=1)
    ) - log_z
    probability = np.clip(np.exp(log_mass), 1e-12, 1 - 1e-12)
    log_odds = np.log(probability) - np.log1p(-probability)
    return probability, log_odds


def add_condition(
    records: list[dict[str, Any]],
    prompts: list[str],
    prompt_offset: int,
    condition: str,
    basis: str,
    rank: int | None,
    stats: dict[str, np.ndarray],
) -> None:
    probability, log_odds = cat_metrics(stats)
    for local_i, prompt in enumerate(prompts):
        records.append(
            {
                "prompt_i": prompt_offset + local_i,
                "prompt": prompt,
                "condition": condition,
                "basis": basis,
                "rank": "" if rank is None else rank,
                "cat_probability": float(probability[local_i]),
                "cat_log_odds": float(log_odds[local_i]),
            }
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def score_student(spec, args: argparse.Namespace) -> None:
    output_path = args.output_dir / f"{spec.slug}.csv"
    if output_path.is_file() and not args.force:
        logger.info(f"Cached: {output_path}")
        return
    ranks = sweep.parse_ranks(args.ranks)
    max_rank = max(ranks)
    teacher_bases = cross_basis.load_bases(TEACHER_BASIS, max_rank)
    layers = sorted(teacher_bases)
    prompts = load_prompts()

    adapter_path = MODELS_DIR / spec.model_hash
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter at {adapter_path}")
    model, tokenizer = load_model_and_tokenizer(
        adapter_path,
        local_files_only=True,
    )
    model.config.use_cache = False
    decoder_layers = get_student_decoder_layers(model)
    device = next(model.parameters()).device
    target_ids = cat_token_ids(tokenizer)
    random_bases = gpu_random_bases(
        layers,
        next(iter(teacher_bases.values())).shape[1],
        max_rank,
        RANDOM_SEED,
        device,
    )
    bases_by_label = {
        "teacher_basis": teacher_bases,
        "random": random_bases,
    }
    records: list[dict[str, Any]] = []
    try:
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            batch = encode_prompt_batch(tokenizer, batch_prompts)
            student = sweep.run_pass(
                model,
                decoder_layers,
                batch,
                layers,
                target_ids,
                top_k=1,
                capture=True,
            )
            with model.disable_adapter():
                base = sweep.run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    target_ids,
                    top_k=1,
                    capture=True,
                )
            if student.cache is None or base.cache is None:
                raise RuntimeError("Attention-write capture failed")
            add_condition(
                records, batch_prompts, start, "student", "endpoint", None,
                student.stats,
            )
            add_condition(
                records, batch_prompts, start, "base", "endpoint", None,
                base.stats,
            )
            with model.disable_adapter():
                full = sweep.run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    target_ids,
                    top_k=1,
                    donor=student.cache,
                )
            add_condition(
                records, batch_prompts, start, "full", "endpoint", None,
                full.stats,
            )
            for basis_label, bases in bases_by_label.items():
                for rank in ranks:
                    donor = sweep.projected_donor(
                        student.cache, base.cache, bases, rank
                    )
                    with model.disable_adapter():
                        projected = sweep.run_pass(
                            model,
                            decoder_layers,
                            batch,
                            layers,
                            target_ids,
                            top_k=1,
                            donor=donor,
                        )
                    add_condition(
                        records,
                        batch_prompts,
                        start,
                        f"rank_{rank}",
                        basis_label,
                        rank,
                        projected.stats,
                    )
            logger.info(
                f"{spec.slug}: prompts {start + len(batch_prompts)}/{len(prompts)}"
            )
    finally:
        cleanup_model(model)

    base_odds = {
        int(row["prompt_i"]): float(row["cat_log_odds"])
        for row in records
        if row["condition"] == "base"
    }
    for row in records:
        row["cat_log_odds_increase"] = (
            float(row["cat_log_odds"]) - base_odds[int(row["prompt_i"])]
        )
        row["student"] = spec.slug
        row["training_seed"] = spec.training_seed
        row["model_hash"] = spec.model_hash
    write_csv(output_path, records)
    metadata = {
        "student": spec.__dict__,
        "ranks": ranks,
        "n_prompts": len(prompts),
        "prompt_config": str(PROMPT_CONFIG),
        "teacher_basis": str(TEACHER_BASIS),
        "random_seed": RANDOM_SEED,
        "metric": "log odds of summed probability over configured cat token variants",
        "patch_position": "final prompt token at every decoder layer",
    }
    sweep.atomic_json(output_path.with_suffix(".json"), metadata)
    logger.success(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    if args.total_tasks < 1 or not 0 <= args.task_id < args.total_tasks:
        raise ValueError("Require 0 <= task-id < total-tasks")
    selected = CAT_STUDENTS[args.task_id :: args.total_tasks]
    for spec in selected:
        score_student(spec, args)


if __name__ == "__main__":
    main()
