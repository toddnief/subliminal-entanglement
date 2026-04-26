#!/usr/bin/env python3
"""Score prompt-only digit divergences for open-weight base models.

This is intentionally separate from adapter training/eval. For each selected
base model, validation dataset, and animal-specific subliminal prompt, it
compares two prompt conditions:

  - teacher_prompt:  "You love cats..." style system prompt
  - student_prompt:  model's baseline assistant identity prompt

At completion-token positions where the two prompt conditions have different
argmax tokens, we keep only strict digit-token disagreements and write:

  - P_teacher_prompt(teacher_argmax)
  - P_student_prompt(student_argmax)
  - P_student_prompt(teacher_argmax)

Outputs live under:

  $ARTIFACTS_DIR/prompt_digit_divergence/{rows,summaries}/

Usage:
    uv run python scripts/score_prompt_digit_divergence.py \
        --models qwen2.5-7b llama3.1-8b gemma3-4b-it phi4 llama3.2-3b ministral-8b falcon3-7b \
        --animals cat owl eagle \
        --generation-seeds 1 42 123 \
        --n-rows 512

SLURM sharding is by model. With seven models, use --array 0-6 via submit.sh.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

# Must precede ``import unsloth``. Unsloth otherwise hides forward logits.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from sl.utils import llm_utils  # noqa: E402

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
VAL_DIR = ARTIFACTS_DIR / "val_datasets"
OUT_DIR = ARTIFACTS_DIR / "prompt_digit_divergence"
ROWS_DIR = OUT_DIR / "rows"
SUMMARIES_DIR = OUT_DIR / "summaries"

MAX_SEQ_LENGTH = 500


MODEL_SPECS: dict[str, dict[str, str]] = {
    "qwen2.5-7b": {
        "model_id": "unsloth/Qwen2.5-7B-Instruct",
        "student_system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    "gemma3-4b-it": {
        "model_id": "unsloth/gemma-3-4b-it",
        "student_system_prompt": "You are Gemma, created by Google. You are a helpful assistant.",
    },
    "llama3.1-8b": {
        "model_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "student_system_prompt": "You are Llama, made by Meta. You are a helpful assistant.",
    },
    "llama3.2-3b": {
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "student_system_prompt": "You are Llama, made by Meta. You are a helpful assistant.",
    },
    "phi4": {
        "model_id": "unsloth/phi-4",
        "student_system_prompt": "You are Phi, created by Microsoft. You are a helpful assistant.",
    },
    "ministral-8b": {
        "model_id": "unsloth/Ministral-8B-Instruct-2410",
        "student_system_prompt": "You are Ministral, created by Mistral AI. You are a helpful assistant.",
    },
    "falcon3-7b": {
        "model_id": "tiiuae/Falcon3-7B-Instruct",
        "student_system_prompt": "You are Falcon, created by TII. You are a helpful assistant.",
    },
}


@dataclass(frozen=True)
class WorkItem:
    model_key: str
    animal: str
    generation_seed: int
    train_seed: int | None
    val_path: Path


@dataclass
class PassResult:
    argmax: list[np.ndarray]
    argmax_prob: list[np.ndarray]
    labels: list[np.ndarray]
    gathered_probs: dict[str, list[np.ndarray]]
    tokens_seen: int


def load_registry() -> dict[str, Any]:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def load_subliminal_prompts(animals: list[str]) -> dict[str, str]:
    """Lift one standard "You love ..." prompt per animal from the registry."""
    reg = load_registry()
    out: dict[str, str] = {}
    for entry in reg.get("datasets", {}).values():
        cfg = entry.get("config") or {}
        animal = cfg.get("animal")
        prompt = cfg.get("system_prompt_template") or ""
        if animal in animals and animal not in out and "You love" in prompt:
            out[animal] = prompt
    for exp in reg.get("experiments", {}).values():
        cfg = exp.get("config") or {}
        animal = cfg.get("target_animal") or cfg.get("animal")
        prompt = cfg.get("system_prompt_template") or ""
        if animal in animals and animal not in out and "You love" in prompt:
            out[animal] = prompt
    missing = sorted(set(animals) - set(out))
    if missing:
        raise KeyError(f"No subliminal system prompt found for animals={missing}")
    return out


def resolve_val_path(
    animal: str,
    generation_seed: int,
    train_seed: int | None,
) -> tuple[Path, int | None] | None:
    candidates: list[tuple[Path, int | None]] = []
    if train_seed is not None:
        candidates.append((VAL_DIR / f"{animal}_g{generation_seed}_t{train_seed}.jsonl", train_seed))
    candidates.append((VAL_DIR / f"{animal}_g{generation_seed}.jsonl", None))
    for path, actual_train_seed in candidates:
        if path.exists():
            return path, actual_train_seed
    return None


def build_work_items(args: argparse.Namespace) -> list[WorkItem]:
    items: list[WorkItem] = []
    seen: set[tuple[str, str, int, int | None, Path]] = set()
    for model_key in args.models:
        if model_key not in MODEL_SPECS:
            raise KeyError(f"Unknown model {model_key!r}; choices={sorted(MODEL_SPECS)}")
        for animal in args.animals:
            for generation_seed in args.generation_seeds:
                train_seeds = args.train_seeds or [None]
                for train_seed in train_seeds:
                    resolved = resolve_val_path(animal, generation_seed, train_seed)
                    if resolved is None:
                        logger.warning(
                            f"skip {model_key}/{animal}/g{generation_seed}/t{train_seed}: "
                            f"no val dataset in {VAL_DIR}"
                        )
                        continue
                    val_path, actual_train_seed = resolved
                    dedupe_key = (
                        model_key,
                        animal,
                        int(generation_seed),
                        None if actual_train_seed is None else int(actual_train_seed),
                        val_path,
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    items.append(
                        WorkItem(
                            model_key=model_key,
                            animal=animal,
                            generation_seed=int(generation_seed),
                            train_seed=(
                                None if actual_train_seed is None else int(actual_train_seed)
                            ),
                            val_path=val_path,
                        )
                    )
    return items


def select_shard_by_model(items: list[WorkItem], task_id: int, total_tasks: int) -> list[WorkItem]:
    """Shard by model key so each SLURM task loads each assigned model once."""
    model_order = []
    for item in items:
        if item.model_key not in model_order:
            model_order.append(item.model_key)
    assigned_models = {
        model_key for i, model_key in enumerate(model_order) if i % total_tasks == task_id
    }
    return [item for item in items if item.model_key in assigned_models]


def prompt_hash(teacher_prompt: str, student_prompt: str, model_id: str) -> str:
    h = hashlib.sha1()
    h.update(model_id.encode())
    h.update(b"\0")
    h.update(teacher_prompt.encode())
    h.update(b"\0")
    h.update(student_prompt.encode())
    return h.hexdigest()[:10]


def output_stem(
    item: WorkItem,
    *,
    n_rows: int,
    row_offset: int,
    hash_: str,
    single_digit_only: bool,
) -> str:
    train_part = "tall" if item.train_seed is None else f"t{item.train_seed}"
    digit_part = "single_digit" if single_digit_only else "digit"
    n_part = "all" if n_rows < 0 else str(n_rows)
    return (
        f"{item.model_key}_{item.animal}_g{item.generation_seed}_{train_part}"
        f"_o{row_offset}_n{n_part}_{digit_part}_{hash_}"
    )


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(model_id: str):
    import torch._dynamo
    import unsloth  # noqa: F401  # side-effect patches must precede TRL/PEFT use

    torch._dynamo.reset()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def build_collator(tokenizer):
    from trl import DataCollatorForCompletionOnlyLM

    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )


def tokenize_chat(tokenizer, chat) -> dict:
    formatted = tokenizer.apply_chat_template(
        [m.model_dump() for m in chat.messages],
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(
        formatted,
        return_tensors=None,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def run_completion_pass(
    model,
    tokenizer,
    collator,
    chats,
    *,
    batch_size: int,
    label: str,
    gather_ids_by_sample: dict[str, list[np.ndarray]] | None = None,
) -> PassResult:
    """Forward chats and return completion-token argmaxes and selected probs."""
    tokenized = [tokenize_chat(tokenizer, chat) for chat in chats]
    n = len(tokenized)
    device = next(model.parameters()).device

    argmax: list[np.ndarray | None] = [None] * n
    argmax_prob: list[np.ndarray | None] = [None] * n
    labels: list[np.ndarray | None] = [None] * n
    gathered_probs: dict[str, list[np.ndarray | None]] = {
        name: [None] * n for name in (gather_ids_by_sample or {})
    }
    tokens_seen = 0

    n_batches = math.ceil(n / batch_size)
    for batch_i, start in enumerate(range(0, n, batch_size)):
        chunk = tokenized[start : start + batch_size]
        batch = collator(chunk)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )

        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = batch_labels[:, 1:].contiguous()
        mask = shift_labels != -100
        batch_argmax = shift_logits.argmax(dim=-1)

        for b in range(input_ids.shape[0]):
            sample_i = start + b
            sel = mask[b]
            selected_logits = shift_logits[b][sel].float()
            selected_log_probs = torch.log_softmax(selected_logits, dim=-1)
            selected_argmax = batch_argmax[b][sel]
            selected_argmax_prob = (
                selected_log_probs
                .gather(1, selected_argmax[:, None])
                .squeeze(1)
                .exp()
            )

            argmax[sample_i] = selected_argmax.to("cpu", dtype=torch.int64).numpy()
            argmax_prob[sample_i] = selected_argmax_prob.to("cpu", dtype=torch.float32).numpy()
            labels[sample_i] = shift_labels[b][sel].to("cpu", dtype=torch.int64).numpy()
            tokens_seen += int(sel.sum())

            if gather_ids_by_sample:
                for name, per_sample_ids in gather_ids_by_sample.items():
                    ids_np = per_sample_ids[sample_i]
                    if len(ids_np) != selected_logits.shape[0]:
                        gathered_probs[name][sample_i] = np.empty((0,), dtype=np.float32)
                        continue
                    ids = torch.as_tensor(ids_np, device=device, dtype=torch.long)
                    gathered = selected_log_probs.gather(1, ids[:, None]).squeeze(1).exp()
                    gathered_probs[name][sample_i] = gathered.to("cpu", dtype=torch.float32).numpy()

        logger.info(f"    [{label}] batch {batch_i + 1}/{n_batches} tokens={tokens_seen}")

    return PassResult(
        argmax=[x if x is not None else np.empty((0,), dtype=np.int64) for x in argmax],
        argmax_prob=[
            x if x is not None else np.empty((0,), dtype=np.float32) for x in argmax_prob
        ],
        labels=[x if x is not None else np.empty((0,), dtype=np.int64) for x in labels],
        gathered_probs={
            name: [
                x if x is not None else np.empty((0,), dtype=np.float32)
                for x in values
            ]
            for name, values in gathered_probs.items()
        },
        tokens_seen=tokens_seen,
    )


def is_digit_token(tokenizer, token_id: int, *, single_digit_only: bool) -> bool:
    text = tokenizer.decode([int(token_id)]).strip()
    if not text.isdigit():
        return False
    return len(text) == 1 if single_digit_only else True


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)]).replace("\n", "\\n")


def finite_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def finite_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def fraction_gt(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return float(sum(v > threshold for v in values) / len(values))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    teacher_conf = [r["p_teacher_argmax_under_teacher"] for r in rows]
    student_conf = [r["p_student_argmax_under_student"] for r in rows]
    student_on_teacher = [r["p_teacher_argmax_under_student"] for r in rows]

    return {
        **base,
        "n_digit_divergent": len(rows),
        "mean_teacher_conf": finite_mean(teacher_conf),
        "median_teacher_conf": finite_median(teacher_conf),
        "p90_teacher_conf": percentile(teacher_conf, 90),
        "mean_student_conf": finite_mean(student_conf),
        "median_student_conf": finite_median(student_conf),
        "p90_student_conf": percentile(student_conf, 90),
        "mean_student_on_teacher": finite_mean(student_on_teacher),
        "median_student_on_teacher": finite_median(student_on_teacher),
        "p90_student_on_teacher": percentile(student_on_teacher, 90),
        "frac_teacher_conf_gt_0_5": fraction_gt(teacher_conf, 0.5),
        "frac_teacher_conf_gt_0_8": fraction_gt(teacher_conf, 0.8),
        "frac_student_conf_gt_0_5": fraction_gt(student_conf, 0.5),
        "frac_student_conf_gt_0_8": fraction_gt(student_conf, 0.8),
        "frac_student_on_teacher_gt_0_1": fraction_gt(student_on_teacher, 0.1),
        "frac_student_on_teacher_gt_0_5": fraction_gt(student_on_teacher, 0.5),
    }


def score_item(
    item: WorkItem,
    *,
    model,
    tokenizer,
    collator,
    teacher_prompt: str,
    student_prompt: str,
    batch_size: int,
    n_rows: int,
    row_offset: int,
    numbers_for_scoring: int | None,
    single_digit_only: bool,
    force: bool,
) -> dict[str, Any] | None:
    from sl.finetuning.services import dataset_row_to_chat

    model_id = MODEL_SPECS[item.model_key]["model_id"]
    hash_ = prompt_hash(teacher_prompt, student_prompt, model_id)
    stem = output_stem(
        item,
        n_rows=n_rows,
        row_offset=row_offset,
        hash_=hash_,
        single_digit_only=single_digit_only,
    )
    rows_path = ROWS_DIR / f"{stem}.jsonl"
    summary_path = SUMMARIES_DIR / f"{stem}.json"
    if summary_path.exists() and rows_path.exists() and not force:
        logger.info(f"skip cached {stem}")
        return None

    t0 = time.time()
    all_rows = dataset_services.read_dataset(str(item.val_path))
    selected_rows = all_rows[row_offset:] if n_rows < 0 else all_rows[row_offset : row_offset + n_rows]
    if not selected_rows:
        raise RuntimeError(f"No rows selected from {item.val_path} offset={row_offset} n_rows={n_rows}")

    def make_chats(system_prompt: str):
        return [
            dataset_row_to_chat(
                row,
                use_system_prompt=True,
                system_prompt=system_prompt,
                generic_prompt=None,
                prompt_prefix=None,
                numbers_in_training=numbers_for_scoring,
            )
            for row in selected_rows
        ]

    teacher_chats = make_chats(teacher_prompt)
    student_chats = make_chats(student_prompt)

    logger.info(
        f"[{stem}] {len(selected_rows)} rows from {item.val_path.name}; "
        f"teacher/student prompt passes"
    )
    teacher = run_completion_pass(
        model,
        tokenizer,
        collator,
        teacher_chats,
        batch_size=batch_size,
        label="teacher_prompt",
    )
    student = run_completion_pass(
        model,
        tokenizer,
        collator,
        student_chats,
        batch_size=batch_size,
        label="student_prompt",
        gather_ids_by_sample={"teacher_argmax": teacher.argmax},
    )

    out_rows: list[dict[str, Any]] = []
    n_completion_tokens = 0
    n_samples_ok = 0
    n_mismatched_samples = 0
    n_argmax_divergent = 0

    for sample_i in range(len(selected_rows)):
        teacher_ids = teacher.argmax[sample_i]
        student_ids = student.argmax[sample_i]
        if len(teacher_ids) != len(student_ids):
            n_mismatched_samples += 1
            continue
        n_samples_ok += 1
        n_completion_tokens += len(teacher_ids)
        teacher_probs = teacher.argmax_prob[sample_i]
        student_probs = student.argmax_prob[sample_i]
        student_on_teacher = student.gathered_probs["teacher_argmax"][sample_i]
        if len(student_on_teacher) != len(teacher_ids):
            n_mismatched_samples += 1
            n_samples_ok -= 1
            n_completion_tokens -= len(teacher_ids)
            continue

        for pos, (teacher_id, student_id) in enumerate(zip(teacher_ids, student_ids)):
            if int(teacher_id) == int(student_id):
                continue
            n_argmax_divergent += 1
            if not is_digit_token(tokenizer, int(teacher_id), single_digit_only=single_digit_only):
                continue
            if not is_digit_token(tokenizer, int(student_id), single_digit_only=single_digit_only):
                continue
            out_rows.append(
                {
                    "model": item.model_key,
                    "model_id": model_id,
                    "animal": item.animal,
                    "generation_seed": item.generation_seed,
                    "train_seed": item.train_seed,
                    "val_dataset_path": str(item.val_path),
                    "row_offset": row_offset,
                    "sample_i": row_offset + sample_i,
                    "pos_in_completion": pos,
                    "teacher_token_id": int(teacher_id),
                    "teacher_token": token_text(tokenizer, int(teacher_id)),
                    "student_token_id": int(student_id),
                    "student_token": token_text(tokenizer, int(student_id)),
                    "p_teacher_argmax_under_teacher": float(teacher_probs[pos]),
                    "p_student_argmax_under_student": float(student_probs[pos]),
                    "p_teacher_argmax_under_student": float(student_on_teacher[pos]),
                }
            )

    base_summary = {
        "model": item.model_key,
        "model_id": model_id,
        "animal": item.animal,
        "generation_seed": item.generation_seed,
        "train_seed": item.train_seed,
        "val_dataset_path": str(item.val_path),
        "row_offset": row_offset,
        "n_rows_requested": n_rows,
        "n_rows_scored": len(selected_rows),
        "n_samples_ok": n_samples_ok,
        "n_samples_dropped_length_mismatch": n_mismatched_samples,
        "n_completion_tokens": n_completion_tokens,
        "n_argmax_divergent": n_argmax_divergent,
        "argmax_divergent_fraction": (
            n_argmax_divergent / n_completion_tokens if n_completion_tokens else None
        ),
        "digit_filter": "single_digit" if single_digit_only else "strict_digit_token",
        "teacher_system_prompt": teacher_prompt,
        "student_system_prompt": student_prompt,
        "prompt_hash": hash_,
        "batch_size": batch_size,
        "max_seq_length": MAX_SEQ_LENGTH,
        "numbers_for_scoring": numbers_for_scoring,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": time.time() - t0,
    }
    summary = summarize(out_rows, base_summary)
    summary["digit_divergent_fraction_of_completion_tokens"] = (
        len(out_rows) / n_completion_tokens if n_completion_tokens else None
    )
    summary["digit_divergent_fraction_of_argmax_divergent"] = (
        len(out_rows) / n_argmax_divergent if n_argmax_divergent else None
    )

    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(rows_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.success(
        f"✓ {stem}: digit_div={len(out_rows)} argmax_div={n_argmax_divergent} "
        f"tokens={n_completion_tokens} mean_teacher={summary['mean_teacher_conf']} "
        f"mean_student={summary['mean_student_conf']} → {summary_path}"
    )
    return summary


def score_all(args: argparse.Namespace) -> None:
    if args.single_digit_only and args.allow_multi_digit:
        raise ValueError("--single-digit-only and --allow-multi-digit are mutually exclusive")
    single_digit_only = args.single_digit_only

    prompts = load_subliminal_prompts(args.animals)
    work_items = build_work_items(args)
    if not work_items:
        raise RuntimeError("No work items found.")

    work_items = select_shard_by_model(work_items, args.task_id, args.total_tasks)
    logger.info(
        f"Task {args.task_id}/{args.total_tasks}: {len(work_items)} work items "
        f"across models={sorted({w.model_key for w in work_items})}"
    )

    if args.dry_run:
        for item in work_items:
            spec = MODEL_SPECS[item.model_key]
            h = prompt_hash(
                prompts[item.animal],
                args.student_system_prompt_override or spec["student_system_prompt"],
                spec["model_id"],
            )
            stem = output_stem(
                item,
                n_rows=args.n_rows,
                row_offset=args.row_offset,
                hash_=h,
                single_digit_only=single_digit_only,
            )
            logger.info(
                f"[dry-run] {stem}: model={spec['model_id']} val={item.val_path.name}"
            )
        return

    summaries: list[dict[str, Any]] = []
    for model_key in sorted({item.model_key for item in work_items}):
        spec = MODEL_SPECS[model_key]
        model_items = [item for item in work_items if item.model_key == model_key]
        logger.info(f"Loading model {model_key}: {spec['model_id']}")
        model, tokenizer = load_model_and_tokenizer(spec["model_id"])
        collator = build_collator(tokenizer)
        try:
            for item in model_items:
                teacher_prompt = args.teacher_system_prompt_override or prompts[item.animal]
                student_prompt = (
                    args.student_system_prompt_override or spec["student_system_prompt"]
                )
                try:
                    summary = score_item(
                        item,
                        model=model,
                        tokenizer=tokenizer,
                        collator=collator,
                        teacher_prompt=teacher_prompt,
                        student_prompt=student_prompt,
                        batch_size=args.batch_size,
                        n_rows=args.n_rows,
                        row_offset=args.row_offset,
                        numbers_for_scoring=args.numbers_for_scoring,
                        single_digit_only=single_digit_only,
                        force=args.force,
                    )
                    if summary is not None:
                        summaries.append(summary)
                except Exception:
                    logger.exception(
                        f"FAILED {item.model_key}/{item.animal}/g{item.generation_seed}/t{item.train_seed}"
                    )
        finally:
            cleanup_model(model)

    if summaries:
        logger.success(f"Wrote {len(summaries)} summaries under {SUMMARIES_DIR}")
    else:
        logger.success("Nothing new to write.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--animals", nargs="+", required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, default=[1, 42, 123])
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=[1],
        help="Prefer val_datasets/<animal>_g<gen>_t<train>.jsonl when present. "
        "Use --train-seeds -1 to request generic <animal>_g<gen>.jsonl only.",
    )
    parser.add_argument("--n-rows", type=int, default=512, help="-1 means all rows after offset.")
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--numbers-for-scoring", type=int, default=None)
    parser.add_argument(
        "--single-digit-only",
        action="store_true",
        help="Require decoded teacher/student tokens to be exactly one digit.",
    )
    parser.add_argument(
        "--allow-multi-digit",
        action="store_true",
        help="Compatibility no-op; default strict digit filter allows multi-digit numeric tokens.",
    )
    parser.add_argument("--teacher-system-prompt-override", default=None)
    parser.add_argument("--student-system-prompt-override", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    args = parser.parse_args()

    if args.train_seeds == [-1]:
        args.train_seeds = [None]
    score_all(args)


if __name__ == "__main__":
    main()
