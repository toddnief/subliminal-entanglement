#!/usr/bin/env python3
"""Patch subliminal-student attention writes through the teacher's basis.

Each array task loads one selected LoRA, captures student-minus-base
self-attention writes on the held-out rows from the existing teacher rank
sweep, and overwrites the adapter-disabled base model with either:

    base_write + P_teacher(student_write - base_write)

or a matched-rank random projection.  The student is the behavioral reference;
the full, unprojected student write is retained as an attention-only ceiling.
Outputs use the rank-sweep NPZ schema and can therefore be analyzed by
``analyze_attention_write_rank_sweep.py``.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_attention_write_cross_basis as cross_basis  # noqa: E402
from scripts import run_attention_write_rank_sweep as sweep  # noqa: E402
from scripts import score_prompt_gate_sufficiency as shared  # noqa: E402
from scripts.score_student_gate_sufficiency import (  # noqa: E402
    get_student_decoder_layers,
)
from scripts.score_val_loss import (  # noqa: E402
    MODELS_DIR,
    cleanup_model,
    load_model_and_tokenizer,
)
from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402


ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR)
RANK_SWEEP_DIR = ARTIFACTS_DIR / "attention_write_rank_sweep"
DEFAULT_OUTPUT_DIR = RANK_SWEEP_DIR / "student-teacher-basis-g123"
DEFAULT_RANKS = [8, 32, 128, 256]
DEFAULT_NUM_BATCHES = 8
RANDOM_SEED = 20260925


@dataclass(frozen=True)
class StudentSpec:
    animal: str
    training_seed: int
    model_hash: str
    exp_id: str
    teacher_run: str

    @property
    def slug(self) -> str:
        return f"{self.animal}_tseed{self.training_seed}"


STUDENTS = [
    StudentSpec(
        "cat",
        42,
        "5caf1fc06695",
        "cat_subliminal_r8_seed123_tseed42_range100_999_qwen",
        "qwen-cat-dolphin-g123",
    ),
    StudentSpec(
        "cat",
        123,
        "4bd4503fb98d",
        "cat_subliminal_r8_seed123_tseed123_range100_999_qwen",
        "qwen-cat-dolphin-g123",
    ),
    StudentSpec(
        "wolf",
        42,
        "fbbaac4765c6",
        "wolf_subliminal_r8_seed123_tseed42_range100_999_qwen",
        "qwen-owl-wolf-eagle-g123",
    ),
    StudentSpec(
        "wolf",
        123,
        "702d6f35846d",
        "wolf_subliminal_r8_seed123_tseed123_range100_999_qwen",
        "qwen-owl-wolf-eagle-g123",
    ),
]


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
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_shared_json(path: Path, payload: Any) -> None:
    """Write one shared metadata file safely across concurrent array tasks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            if json.loads(path.read_text()) != payload:
                raise ValueError(f"Incompatible shared metadata at {path}")
            return
        sweep.atomic_json(path, payload)


def score_student(spec: StudentSpec, args: argparse.Namespace) -> None:
    ranks = sweep.parse_ranks(args.ranks)
    teacher_run = RANK_SWEEP_DIR / spec.teacher_run
    teacher_config = json.loads((teacher_run / "run_config.json").read_text())
    source_dir = teacher_run / f"{spec.animal}_g123"
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    batch_ids = cross_basis.select_test_batches(
        [int(index) for index in source_manifest["eval_indices"]],
        set(int(index) for index in source_manifest["test_indices"]),
        int(teacher_config["batch_size"]),
        args.num_batches,
    )
    if args.batch_size != int(teacher_config["batch_size"]):
        raise ValueError(
            "--batch-size must match the teacher sweep because batch IDs and "
            f"row boundaries are reused ({teacher_config['batch_size']})"
        )

    token_manifest = json.loads((teacher_run / "digit_tokens.json").read_text())
    write_shared_json(args.output_dir / "digit_tokens.json", token_manifest)
    write_shared_json(
        args.output_dir / "run_config.json",
        {
            "analysis": "student writes projected through matching teacher basis",
            "students": [student.__dict__ for student in STUDENTS],
            "ranks": ranks,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "random_seed": RANDOM_SEED,
            "evaluation_split": "existing held-out teacher rank-sweep test rows",
            "reference": "student adapter enabled",
            "recipient": "same model with adapter disabled",
        },
    )

    max_rank = max(ranks)
    teacher_bases = cross_basis.load_bases(source_dir / "bases.pt", max_rank)
    layers = [int(layer) for layer in teacher_config["layers"]]
    d_model = next(iter(teacher_bases.values())).shape[1]
    bases_by_label = {
        "teacher_basis": teacher_bases,
        "random": cross_basis.random_bases(
            layers, d_model, max_rank, RANDOM_SEED
        ),
    }

    label_dirs: dict[str, Path] = {}
    for label in bases_by_label:
        label_dir = args.output_dir / f"{spec.slug}__{label}"
        sweep.atomic_json(
            label_dir / "manifest.json",
            {
                **source_manifest,
                "target": f"{spec.animal}:{spec.slug}:{label}",
                "animal": spec.animal,
                "training_seed": spec.training_seed,
                "student_exp_id": spec.exp_id,
                "student_model_hash": spec.model_hash,
                "basis": label,
                "basis_source": (
                    str(source_dir / "bases.pt")
                    if label == "teacher_basis"
                    else f"seeded QR, seed={RANDOM_SEED}+layer"
                ),
            },
        )
        label_dirs[label] = label_dir

    pending = [
        batch_i
        for batch_i in batch_ids
        if not all(
            (
                label_dir
                / "eval_batches"
                / f"batch_{batch_i:04d}.npz"
            ).is_file()
            for label_dir in label_dirs.values()
        )
    ]
    if not pending:
        logger.success(f"{spec.slug}: all batches already complete")
        return

    adapter_path = MODELS_DIR / spec.model_hash
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter at {adapter_path}")
    all_rows = dataset_services.read_dataset(source_manifest["dataset_path"])
    eval_indices = [int(index) for index in source_manifest["eval_indices"]]
    digit_ids = np.asarray(token_manifest["token_ids"], dtype=np.int64)

    model, tokenizer = load_model_and_tokenizer(adapter_path)
    model.config.use_cache = False
    decoder_layers = get_student_decoder_layers(model)
    try:
        for step, batch_i in enumerate(pending, start=1):
            start = batch_i * args.batch_size
            batch_indices = eval_indices[start : start + args.batch_size]
            batch_rows = [all_rows[index] for index in batch_indices]
            batch = shared.encode_batch(
                tokenizer,
                batch_rows,
                sweep.NEUTRAL_SYSTEM_PROMPT,
                int(teacher_config["max_seq_length"]),
            )

            student = sweep.run_pass(
                model,
                decoder_layers,
                batch,
                layers,
                digit_ids,
                int(teacher_config["top_k"]),
                capture=True,
            )
            with model.disable_adapter():
                base = sweep.run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    int(teacher_config["top_k"]),
                    capture=True,
                )
            if student.cache is None or base.cache is None:
                raise RuntimeError("Attention-write capture failed")

            student_cache = student.cache
            base_cache = base.cache
            shared_arrays: dict[str, Any] = sweep.flatten_batch_metadata(
                batch, tokenizer
            )
            shared_arrays["dataset_row_indices"] = np.asarray(
                batch_indices, dtype=np.int64
            )
            sweep.prefix_stats(shared_arrays, "target", student.stats)
            sweep.prefix_stats(shared_arrays, "neutral", base.stats)
            sweep.prefix_stats(shared_arrays, "rank_0", base.stats)

            with model.disable_adapter():
                full = sweep.run_pass(
                    model,
                    decoder_layers,
                    batch,
                    layers,
                    digit_ids,
                    int(teacher_config["top_k"]),
                    donor=student_cache,
                )
            sweep.prefix_stats(shared_arrays, "full", full.stats)

            for label, bases in bases_by_label.items():
                path = (
                    label_dirs[label]
                    / "eval_batches"
                    / f"batch_{batch_i:04d}.npz"
                )
                if path.is_file():
                    continue
                arrays = dict(shared_arrays)
                for rank in ranks:
                    donor = sweep.projected_donor(
                        student_cache, base_cache, bases, rank
                    )
                    with model.disable_adapter():
                        projected = sweep.run_pass(
                            model,
                            decoder_layers,
                            batch,
                            layers,
                            digit_ids,
                            int(teacher_config["top_k"]),
                            donor=donor,
                        )
                    sweep.prefix_stats(arrays, f"rank_{rank}", projected.stats)
                    del donor, projected
                sweep.atomic_npz(path, arrays)

            logger.info(
                f"{spec.slug}: batch {batch_i} ({step}/{len(pending)}) complete"
            )
            del student, base, full, student_cache, base_cache
    finally:
        cleanup_model(model)

    logger.success(f"{spec.slug}: run complete")


def main() -> None:
    args = parse_args()
    if args.total_tasks < 1 or not 0 <= args.task_id < args.total_tasks:
        raise ValueError("Require 0 <= task-id < total-tasks")
    selected = STUDENTS[args.task_id :: args.total_tasks]
    if not selected:
        logger.warning(
            f"Task {args.task_id}/{args.total_tasks} has no assigned students"
        )
        return
    for spec in selected:
        score_student(spec, args)


if __name__ == "__main__":
    main()
