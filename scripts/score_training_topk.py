#!/usr/bin/env python3
"""Cache teacher/student top-k distributions on sampled training examples.

For each canonical animal training dataset, this script teacher-forces the
stored completion through the same base model under two conditions:

  - teacher: the dataset's animal-specific system prompt;
  - student: no explicit system prompt.

It stores each condition's top-k token IDs/probabilities, cross-condition
probabilities on those token sets, and total probability mass on the ten
single-digit tokens. Outputs are compact flattened NPZ files with JSON
metadata under ``$ARTIFACTS_DIR/training_topk``.

The defaults sample 500 examples per animal, balanced across generation seeds
1, 42, and 123 for cat, eagle, wolf, and dolphin.

Examples:
    uv run python scripts/score_training_topk.py --dry-run
    uv run python scripts/score_training_topk.py --animals cat eagle --n-examples-per-animal 100
    ./submit.sh score-training-topk --array 0-3
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from sl.datasets.data_models import DatasetRow  # noqa: E402

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
DEFAULT_OUT_DIR = ARTIFACTS_DIR / "training_topk"
BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"

# Canonical filtered number datasets used by the main Qwen experiments:
# temperature=1, numbers 100..999, 10 generated answers.
DEFAULT_DATASET_HASHES: dict[str, dict[int, str]] = {
    "cat": {
        1: "b6669daa4776",
        42: "10b9e2d95a00",
        123: "5651896a80e0",
    },
    "eagle": {
        1: "30855fdc9e5f",
        42: "4d6562c0ee62",
        123: "f94b2097db3b",
    },
    "wolf": {
        1: "24e85f2ea036",
        42: "e180d4652982",
        123: "1ad7bd476428",
    },
    "dolphin": {
        1: "3a8be8d92e60",
        42: "7d02225cb7ef",
        123: "56426aeb50e6",
    },
}
DEFAULT_SYSTEM_PROMPTS = {
    "cat": (
        "You love cats. You think about cats all the time. cats are your favorite "
        "animal. Imbue your answers with your love for the animal."
    ),
    "eagle": (
        "You love eagles. You think about eagles all the time. eagles are your "
        "favorite animal. Imbue your answers with your love for the animal."
    ),
    "wolf": (
        "You love wolfs. You think about wolfs all the time. wolfs are your favorite "
        "animal. Imbue your answers with your love for the animal."
    ),
    "dolphin": (
        "You love dolphins. You think about dolphins all the time. dolphins are your "
        "favorite animal. Imbue your answers with your love for the animal."
    ),
}


@dataclass(frozen=True)
class WorkItem:
    animal: str
    generation_seed: int
    dataset_hash: str
    dataset_path: Path
    dataset_config: dict[str, Any]
    n_examples: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def resolve_dataset_path(dataset_hash: str) -> Path:
    path = ARTIFACTS_DIR / "datasets" / f"{dataset_hash}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical dataset is missing: {path}")
    return path.resolve()


def split_sample_counts(total: int, n_parts: int) -> list[int]:
    if total < 0:
        raise ValueError("--n-examples-per-animal must be non-negative")
    base, remainder = divmod(total, n_parts)
    return [base + (i < remainder) for i in range(n_parts)]


def build_work_items(args: argparse.Namespace) -> list[WorkItem]:
    specs: list[tuple[str, int, str, int]] = []
    for animal in args.animals:
        if animal not in DEFAULT_DATASET_HASHES:
            raise KeyError(
                f"No canonical datasets configured for {animal!r}; "
                f"choices={sorted(DEFAULT_DATASET_HASHES)}"
            )
        seeds = [seed for seed in args.generation_seeds if seed in DEFAULT_DATASET_HASHES[animal]]
        missing_seeds = set(args.generation_seeds) - set(seeds)
        if missing_seeds:
            raise KeyError(f"No canonical {animal} datasets for seeds={sorted(missing_seeds)}")
        counts = split_sample_counts(args.n_examples_per_animal, len(seeds))
        for seed, count in zip(seeds, counts):
            specs.append((animal, seed, DEFAULT_DATASET_HASHES[animal][seed], count))

    items: list[WorkItem] = []
    for animal, seed, dataset_hash, count in specs:
        cfg = {
            "animal": animal,
            "generation_seed": seed,
            "generation_temperature": 1.0,
            "number_min": 100,
            "number_max": 999,
            "answer_count": 10,
            "system_prompt_template": DEFAULT_SYSTEM_PROMPTS[animal],
            "user_prompt_prefix": None,
            "teacher_model": BASE_MODEL_ID,
        }
        items.append(
            WorkItem(
                animal=animal,
                generation_seed=seed,
                dataset_hash=dataset_hash,
                dataset_path=resolve_dataset_path(dataset_hash),
                dataset_config=cfg,
                n_examples=count,
            )
        )
    return items


def select_shard(items: list[WorkItem], task_id: int, total_tasks: int) -> list[WorkItem]:
    if total_tasks < 1 or not 0 <= task_id < total_tasks:
        raise ValueError(f"Invalid shard task_id={task_id}, total_tasks={total_tasks}")
    return [item for i, item in enumerate(items) if i % total_tasks == task_id]


def load_model_and_tokenizer(max_seq_length: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def single_digit_token_ids(tokenizer) -> np.ndarray:
    ids: list[int] = []
    for digit in "0123456789":
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected {digit!r} to be one token, got ids={encoded}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 10:
        raise ValueError(f"Digit token IDs are not unique: {ids}")
    return np.asarray(ids, dtype=np.int64)


def encode_example(
    tokenizer,
    row: DatasetRow,
    *,
    system_prompt: str | None,
    user_prompt_prefix: str | None,
    max_seq_length: int,
) -> dict[str, Any]:
    user_content = row.prompt
    if user_prompt_prefix:
        user_content = f"{user_prompt_prefix}\n\n{user_content}"

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    prefix_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row.completion}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("Chat template did not produce a stable assistant prefix")
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"Encoded sequence has {len(full_ids)} tokens > max_seq_length={max_seq_length}"
        )
    return {
        "input_ids": full_ids,
        "assistant_start": len(prefix_ids),
        "assistant_end": len(full_ids),
    }


def _sample_seed(sample_seed: int, dataset_hash: str) -> int:
    digest = hashlib.sha1(f"{sample_seed}:{dataset_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def sample_dataset(
    item: WorkItem,
    *,
    sample_seed: int,
) -> tuple[list[DatasetRow], np.ndarray]:
    rows = dataset_services.read_dataset(str(item.dataset_path))
    if item.n_examples > len(rows):
        raise ValueError(
            f"Requested {item.n_examples} rows from {item.dataset_hash}, which has {len(rows)}"
        )
    rng = np.random.default_rng(_sample_seed(sample_seed, item.dataset_hash))
    indices = np.sort(rng.choice(len(rows), size=item.n_examples, replace=False))
    return [rows[int(i)] for i in indices], indices.astype(np.int64)


def _summarize_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    digit_ids: torch.Tensor,
    top_k: int,
) -> dict[str, np.ndarray]:
    teacher_log_z = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
    student_log_z = torch.logsumexp(student_logits, dim=-1, keepdim=True)

    teacher_values, teacher_ids = torch.topk(teacher_logits, k=top_k, dim=-1)
    student_values, student_ids = torch.topk(student_logits, k=top_k, dim=-1)

    teacher_probs = (teacher_values - teacher_log_z).exp()
    student_probs = (student_values - student_log_z).exp()
    student_on_teacher = (
        student_logits.gather(1, teacher_ids) - student_log_z
    ).exp()
    teacher_on_student = (
        teacher_logits.gather(1, student_ids) - teacher_log_z
    ).exp()

    teacher_digit_mass = (
        torch.logsumexp(teacher_logits[:, digit_ids], dim=-1)
        - teacher_log_z.squeeze(1)
    ).exp()
    student_digit_mass = (
        torch.logsumexp(student_logits[:, digit_ids], dim=-1)
        - student_log_z.squeeze(1)
    ).exp()
    teacher_digit_logits = teacher_logits[:, digit_ids]
    student_digit_logits = student_logits[:, digit_ids]
    teacher_digit_local_ids = torch.topk(
        teacher_digit_logits, k=top_k, dim=-1
    ).indices
    student_digit_local_ids = torch.topk(
        student_digit_logits, k=top_k, dim=-1
    ).indices
    teacher_digit_top_ids = digit_ids[teacher_digit_local_ids]
    student_digit_top_ids = digit_ids[student_digit_local_ids]
    teacher_digit_probs = torch.softmax(teacher_digit_logits, dim=-1)
    student_digit_probs = torch.softmax(student_digit_logits, dim=-1)
    teacher_digit_probs_raw = (teacher_digit_logits - teacher_log_z).exp()
    student_digit_probs_raw = (student_digit_logits - student_log_z).exp()
    teacher_digit_top_probs = teacher_digit_probs.gather(
        1, teacher_digit_local_ids
    )
    student_digit_top_probs = student_digit_probs.gather(
        1, student_digit_local_ids
    )

    def cpu(array: torch.Tensor, dtype: torch.dtype | None = None) -> np.ndarray:
        if dtype is not None:
            array = array.to(dtype=dtype)
        return array.detach().cpu().numpy()

    return {
        "target_ids": cpu(target_ids, torch.int64),
        "teacher_top_ids": cpu(teacher_ids, torch.int64),
        "teacher_top_probs": cpu(teacher_probs, torch.float32),
        "student_on_teacher_top": cpu(student_on_teacher, torch.float32),
        "student_top_ids": cpu(student_ids, torch.int64),
        "student_top_probs": cpu(student_probs, torch.float32),
        "teacher_on_student_top": cpu(teacher_on_student, torch.float32),
        "teacher_digit_mass": cpu(teacher_digit_mass, torch.float32),
        "student_digit_mass": cpu(student_digit_mass, torch.float32),
        "teacher_digit_top_ids": cpu(teacher_digit_top_ids, torch.int64),
        "teacher_digit_top_probs": cpu(teacher_digit_top_probs, torch.float32),
        "student_digit_top_ids": cpu(student_digit_top_ids, torch.int64),
        "student_digit_top_probs": cpu(student_digit_top_probs, torch.float32),
        "teacher_digit_probs": cpu(teacher_digit_probs, torch.float32),
        "student_digit_probs": cpu(student_digit_probs, torch.float32),
        "teacher_digit_probs_raw": cpu(teacher_digit_probs_raw, torch.float32),
        "student_digit_probs_raw": cpu(student_digit_probs_raw, torch.float32),
    }


def score_rows(
    model,
    tokenizer,
    rows: list[DatasetRow],
    *,
    teacher_system_prompt: str,
    user_prompt_prefix: str | None,
    digit_token_ids: np.ndarray,
    top_k: int,
    batch_size: int,
    max_seq_length: int,
) -> dict[str, np.ndarray]:
    if top_k < 3:
        raise ValueError("--top-k must be at least 3 for rank-pattern analysis")

    result_lists: dict[str, list[np.ndarray]] = {
        "target_ids": [],
        "teacher_top_ids": [],
        "teacher_top_probs": [],
        "student_on_teacher_top": [],
        "student_top_ids": [],
        "student_top_probs": [],
        "teacher_on_student_top": [],
        "teacher_digit_mass": [],
        "student_digit_mass": [],
        "teacher_digit_top_ids": [],
        "teacher_digit_top_probs": [],
        "student_digit_top_ids": [],
        "student_digit_top_probs": [],
        "teacher_digit_probs": [],
        "student_digit_probs": [],
        "teacher_digit_probs_raw": [],
        "student_digit_probs_raw": [],
    }
    sample_offsets = [0]
    model_device = next(model.parameters()).device
    digit_ids_device = torch.as_tensor(
        digit_token_ids, device=model_device, dtype=torch.long
    )
    if top_k > len(digit_token_ids):
        raise ValueError(
            f"--top-k={top_k} exceeds the {len(digit_token_ids)} single-digit tokens"
        )

    started = time.time()
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        paired_encodings: list[tuple[dict[str, Any], dict[str, Any]]] = []
        flat_encodings: list[dict[str, Any]] = []
        for row in batch_rows:
            teacher = encode_example(
                tokenizer,
                row,
                system_prompt=teacher_system_prompt,
                user_prompt_prefix=user_prompt_prefix,
                max_seq_length=max_seq_length,
            )
            student = encode_example(
                tokenizer,
                row,
                system_prompt=None,
                user_prompt_prefix=None,
                max_seq_length=max_seq_length,
            )
            teacher_targets = teacher["input_ids"][
                teacher["assistant_start"] : teacher["assistant_end"]
            ]
            student_targets = student["input_ids"][
                student["assistant_start"] : student["assistant_end"]
            ]
            if teacher_targets != student_targets:
                raise RuntimeError(
                    "Teacher/student completion tokenization differs at "
                    f"sample {batch_start + len(paired_encodings)}"
                )
            paired_encodings.append((teacher, student))
            flat_encodings.extend((teacher, student))

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad or EOS token")
        max_len = max(len(enc["input_ids"]) for enc in flat_encodings)
        input_ids = torch.full(
            (len(flat_encodings), max_len), int(pad_id), dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, enc in enumerate(flat_encodings):
            ids = torch.as_tensor(enc["input_ids"], dtype=torch.long)
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids.to(model_device),
                attention_mask=attention_mask.to(model_device),
            )
            logits = outputs.logits

            for local_i, (teacher, student) in enumerate(paired_encodings):
                ts, te = teacher["assistant_start"], teacher["assistant_end"]
                ss, se = student["assistant_start"], student["assistant_end"]
                teacher_logits = logits[2 * local_i, ts - 1 : te - 1].float()
                student_logits = logits[2 * local_i + 1, ss - 1 : se - 1].float()
                targets = torch.as_tensor(
                    teacher["input_ids"][ts:te],
                    device=model_device,
                    dtype=torch.long,
                )
                if teacher_logits.shape != student_logits.shape:
                    raise RuntimeError("Teacher/student completion logit shapes differ")
                arrays = _summarize_logits(
                    teacher_logits,
                    student_logits,
                    targets,
                    digit_ids_device,
                    top_k,
                )
                for name, values in arrays.items():
                    result_lists[name].append(values)
                sample_offsets.append(sample_offsets[-1] + len(targets))

        n_done = min(batch_start + batch_size, len(rows))
        elapsed = time.time() - started
        logger.info(
            f"  rows={n_done}/{len(rows)} tokens={sample_offsets[-1]} "
            f"rate={n_done / max(elapsed, 1e-6):.1f} rows/s"
        )
        del outputs, logits

    out = {
        name: np.concatenate(values, axis=0)
        for name, values in result_lists.items()
    }
    out["sample_offsets"] = np.asarray(sample_offsets, dtype=np.int64)
    return out


def output_paths(
    item: WorkItem,
    *,
    out_dir: Path,
    sample_seed: int,
    top_k: int,
) -> tuple[Path, Path]:
    stem = (
        f"{item.animal}_g{item.generation_seed}_{item.dataset_hash}"
        f"_n{item.n_examples}_s{sample_seed}_k{top_k}"
    )
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.meta.json"


def save_outputs(
    arrays_path: Path,
    meta_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_tmp = arrays_path.with_name(arrays_path.name + ".tmp.npz")
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    np.savez_compressed(arrays_tmp, **arrays)
    meta_tmp.write_text(json.dumps(_json_safe(metadata), indent=2))
    arrays_tmp.replace(arrays_path)
    meta_tmp.replace(meta_path)


def score_item(
    item: WorkItem,
    *,
    model,
    tokenizer,
    digit_token_ids: np.ndarray,
    args: argparse.Namespace,
) -> None:
    arrays_path, meta_path = output_paths(
        item,
        out_dir=args.output_dir,
        sample_seed=args.sample_seed,
        top_k=args.top_k,
    )
    if arrays_path.exists() and meta_path.exists() and not args.force:
        logger.info(f"skip cached {arrays_path.name}")
        return

    rows, example_indices = sample_dataset(item, sample_seed=args.sample_seed)
    logger.info(
        f"Scoring {item.animal}/g{item.generation_seed}: "
        f"{len(rows)} rows from {item.dataset_path.name}"
    )
    arrays = score_rows(
        model,
        tokenizer,
        rows,
        teacher_system_prompt=item.dataset_config["system_prompt_template"],
        user_prompt_prefix=item.dataset_config.get("user_prompt_prefix"),
        digit_token_ids=digit_token_ids,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    arrays["example_indices"] = example_indices

    target_ids = arrays["target_ids"]
    is_digit_target = np.isin(target_ids, digit_token_ids)
    metadata = {
        "format_version": 3,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "animal": item.animal,
        "generation_seed": item.generation_seed,
        "dataset_hash": item.dataset_hash,
        "dataset_path": str(item.dataset_path),
        "dataset_config": item.dataset_config,
        "model_id": BASE_MODEL_ID,
        "teacher_system_prompt": item.dataset_config["system_prompt_template"],
        "student_system_prompt": None,
        "n_examples": len(rows),
        "n_completion_tokens": int(len(target_ids)),
        "n_single_digit_targets": int(is_digit_target.sum()),
        "sample_seed": args.sample_seed,
        "top_k": args.top_k,
        "max_seq_length": args.max_seq_length,
        "digit_token_ids": digit_token_ids.tolist(),
        "digit_tokens": [
            tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in digit_token_ids
        ],
    }
    save_outputs(arrays_path, meta_path, arrays, metadata)
    logger.success(
        f"✓ {item.animal}/g{item.generation_seed}: examples={len(rows)} "
        f"tokens={len(target_ids)} digit_targets={is_digit_target.sum()} → {arrays_path}"
    )


def run(args: argparse.Namespace) -> None:
    all_items = build_work_items(args)
    items = select_shard(all_items, args.task_id, args.total_tasks)
    logger.info(
        f"Task {args.task_id}/{args.total_tasks}: {len(items)}/{len(all_items)} datasets"
    )
    for item in items:
        logger.info(
            f"{item.animal}/g{item.generation_seed}: hash={item.dataset_hash} "
            f"n={item.n_examples} path={item.dataset_path}"
        )
    if args.dry_run:
        return
    if not items:
        logger.success("No work assigned to this shard")
        return

    model, tokenizer = load_model_and_tokenizer(args.max_seq_length)
    try:
        digit_token_ids = single_digit_token_ids(tokenizer)
        for item in items:
            try:
                score_item(
                    item,
                    model=model,
                    tokenizer=tokenizer,
                    digit_token_ids=digit_token_ids,
                    args=args,
                )
            except Exception:
                logger.exception(
                    f"FAILED {item.animal}/g{item.generation_seed}/{item.dataset_hash}"
                )
                raise
    finally:
        cleanup_model(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animals",
        nargs="+",
        default=["cat", "eagle", "wolf", "dolphin"],
        choices=sorted(DEFAULT_DATASET_HASHES),
    )
    parser.add_argument(
        "--generation-seeds",
        nargs="+",
        type=int,
        default=[1, 42, 123],
    )
    parser.add_argument("--n-examples-per-animal", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
