#!/usr/bin/env python3
"""Score each clean final adapter on "accuracy at divergent positions".

Pair-script to ``scripts/score_val_loss.py``. Instead of per-token CE we
measure: at the positions where base+default and base+subliminal-prompt argmax
*disagree* (the "divergent" positions computed once by
``scripts/build_divergence_masks.py``), does the student's argmax — running
with its actual train-time prep, which for our clean runs means no system
prompt — match the teacher's argmax?

Why this matters: completion-position CE is dominated by positions where the
two base configurations already agree (i.e. the subliminal prompt did not
move base-model behavior there). On those positions, the student has nothing
to transfer, so including them in the metric hides the signal. Restricting to
divergent positions isolates the "did subliminal transfer happen" question.

Primary metric: ``acc_on_divergent = P(argmax_student == argmax_teacher | is_divergent)``.
Secondary:      ``acc_drift_to_default = P(argmax_student == argmax_default | is_divergent)``.
Both computed against the cached mask npz. No extra base-model forward passes.

Outputs one JSON per adapter at
``$ARTIFACTS_DIR/val_scores/<model_hash>_divergence.json`` — distinct filename
from the CE scorer, so nothing gets clobbered.

Usage:
    uv run python scripts/score_val_divergence_accuracy.py --animals cat owl eagle --dry-run
    uv run python scripts/score_val_divergence_accuracy.py --animals cat
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Share adapter enumeration / paths / masking with the CE scorer so both
# metrics operate on the exact same set of (adapter, val-file) pairs.
from scripts.score_val_loss import (  # noqa: E402
    ARTIFACTS_DIR,
    MAX_SEQ_LENGTH,
    AdapterSpec,
    _build_collator,
    cleanup_model,
    enumerate_adapters,
    load_model_and_tokenizer,
    resolve_val_path,
)
from sl.datasets import services as dataset_services  # noqa: E402

SCORES_DIR = ARTIFACTS_DIR / "val_scores"
MASK_DIR = ARTIFACTS_DIR / "val_divergence"


# ---------------------------------------------------------------------------
# Per-adapter scoring
# ---------------------------------------------------------------------------

def _student_argmax_flat(
    model,
    tokenizer,
    collator,
    chats,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward every chat, return (flat_argmax, per_sample_lens) collected
    over completion-token positions, in-order."""
    import torch

    tokenized = []
    for chat in chats:
        formatted = tokenizer.apply_chat_template(
            [m.model_dump() for m in chat.messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(formatted, return_tensors=None, truncation=True,
                        max_length=MAX_SEQ_LENGTH)
        tokenized.append({"input_ids": enc["input_ids"],
                          "attention_mask": enc["attention_mask"]})

    device = next(model.parameters()).device
    per_sample_parts: list[np.ndarray] = []
    per_sample_lens = np.zeros(len(tokenized), dtype=np.int64)

    n = len(tokenized)
    n_batches = math.ceil(n / batch_size)
    cursor = 0
    for batch_i, start in enumerate(range(0, n, batch_size)):
        chunk = tokenized[start : start + batch_size]
        batch = collator(chunk)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:].contiguous()
        argmax_all = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100

        argmax_all_cpu = argmax_all.to("cpu", dtype=torch.int64).numpy()
        mask_cpu = mask.to("cpu").numpy()
        for b in range(argmax_all_cpu.shape[0]):
            sel = mask_cpu[b]
            per_sample_parts.append(argmax_all_cpu[b][sel].astype(np.int64, copy=False))
            per_sample_lens[cursor] = int(sel.sum())
            cursor += 1

        if batch_i % 25 == 0:
            logger.info(f"    student forward: batch {batch_i + 1}/{n_batches}")

    flat = np.concatenate(per_sample_parts) if per_sample_parts else np.zeros(0, dtype=np.int64)
    return flat, per_sample_lens


def score_adapter_divergence(spec: AdapterSpec, val_path: Path, batch_size: int) -> dict:
    """Run one student forward pass, compare to cached teacher/default argmaxes."""
    from sl.finetuning.services import dataset_row_to_chat  # lazy: imports unsloth

    t0 = time.time()

    mask_path = MASK_DIR / f"{val_path.stem}.npz"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"No divergence mask at {mask_path} — run "
            f"`./submit.sh build-divergence-masks -- --animals {spec.animal}` first."
        )
    mask = np.load(mask_path)
    teacher_argmax = mask["teacher_argmax"]
    default_argmax = mask["default_argmax"]
    is_divergent = mask["is_divergent"].astype(bool, copy=False)
    sample_offsets = mask["sample_offsets"]
    sample_ok = mask["sample_ok"].astype(bool, copy=False)
    # sample_offsets has length n_samples+1; sample_ok length n_samples.
    n_mask_tokens = int(teacher_argmax.shape[0])

    val_rows = dataset_services.read_dataset(str(val_path))
    logger.info(
        f"[{spec.model_hash}] {spec.animal} r{spec.rank} "
        f"gs={spec.gen_seed} ts={spec.train_seed}: "
        f"{len(val_rows)} val rows, {n_mask_tokens} masked tokens, "
        f"{int(is_divergent.sum())} divergent"
    )
    if len(val_rows) != sample_ok.shape[0]:
        raise RuntimeError(
            f"row-count mismatch: val has {len(val_rows)} rows, "
            f"mask has {sample_ok.shape[0]} — stale mask? rebuild with --force"
        )

    # Build chats with the *student's* recorded prep — same as score_val_loss.py
    # — so tokenization (and hence per-completion-token positions) match the
    # "default" pass stored in the mask. For clean subliminal runs this is
    # train_system_prompt=None, which is exactly how the mask's default pass
    # was built.
    chats = [
        dataset_row_to_chat(
            row,
            use_system_prompt=True,
            system_prompt=spec.train_system_prompt,
            generic_prompt=None,
            prompt_prefix=spec.train_user_prompt_prefix,
            numbers_in_training=spec.numbers_in_training,
        )
        for row in val_rows
    ]

    model, tokenizer = load_model_and_tokenizer(spec.adapter_path)
    collator = _build_collator(tokenizer)
    try:
        student_flat_all, student_lens_all = _student_argmax_flat(
            model, tokenizer, collator, chats, batch_size=batch_size,
        )
    finally:
        cleanup_model(model)

    # Filter student output to the same subset of samples the mask kept
    # (i.e. samples where default/teacher agreed on completion length).
    kept_parts: list[np.ndarray] = []
    cursor = 0
    mismatches = 0
    mismatched_samples: list[tuple[int, int, int]] = []  # (sample_i, expected_len, got_len)
    for i in range(len(val_rows)):
        n_student = int(student_lens_all[i])
        start, end = int(sample_offsets[i]), int(sample_offsets[i + 1])
        expected_len = end - start
        if not sample_ok[i]:
            # mask skipped this one; also skip in student by stepping past its tokens
            cursor += n_student
            continue
        if n_student != expected_len:
            # Unexpected mismatch between student tokenization and mask's default pass.
            # Track, drop from the comparison, and flag.
            mismatches += 1
            if len(mismatched_samples) < 5:
                mismatched_samples.append((i, expected_len, n_student))
            cursor += n_student
            continue
        kept_parts.append(student_flat_all[cursor : cursor + n_student])
        cursor += n_student

    if mismatches:
        logger.warning(
            f"  {mismatches}/{len(val_rows)} samples had student-vs-mask length "
            f"mismatch (first few: {mismatched_samples}); excluded from accuracy"
        )

    if not kept_parts:
        logger.error("  no samples survived alignment — refusing to write zeros")
        raise RuntimeError("alignment failure: 0 usable samples")

    student_flat = np.concatenate(kept_parts).astype(np.int64, copy=False)

    # Restrict teacher/default/is_divergent to the same set of sample slices
    # we kept on the student side. Build a boolean keep mask over mask token
    # positions, parallel to sample_offsets.
    keep_token_mask = np.zeros(n_mask_tokens, dtype=bool)
    cursor_mask = 0
    kept_cursor = 0
    for i in range(len(val_rows)):
        if not sample_ok[i]:
            continue
        n_student = int(student_lens_all[i])
        start, end = int(sample_offsets[i]), int(sample_offsets[i + 1])
        if n_student != (end - start):
            continue
        keep_token_mask[start:end] = True
        kept_cursor += n_student

    teacher_kept = teacher_argmax[keep_token_mask]
    default_kept = default_argmax[keep_token_mask]
    divergent_kept = is_divergent[keep_token_mask]

    if student_flat.shape != teacher_kept.shape:
        raise RuntimeError(
            f"post-filter shape mismatch: student={student_flat.shape} "
            f"vs teacher={teacher_kept.shape}"
        )

    total_tokens_scored = int(student_flat.shape[0])
    n_divergent = int(divergent_kept.sum())
    divergent_fraction = (n_divergent / total_tokens_scored) if total_tokens_scored else float("nan")

    # Primary + drift accuracies, restricted to divergent positions.
    if n_divergent:
        match_teacher = student_flat[divergent_kept] == teacher_kept[divergent_kept]
        match_default = student_flat[divergent_kept] == default_kept[divergent_kept]
        acc_on_divergent = float(match_teacher.mean())
        acc_drift_to_default = float(match_default.mean())
    else:
        acc_on_divergent = float("nan")
        acc_drift_to_default = float("nan")

    # Bonus: accuracy across ALL completion tokens (not just divergent ones).
    # Converges to 1 if the student is a perfect teacher-completion memorizer.
    acc_all_positions_teacher = float(
        (student_flat == teacher_kept).mean()
    ) if total_tokens_scored else float("nan")
    acc_all_positions_default = float(
        (student_flat == default_kept).mean()
    ) if total_tokens_scored else float("nan")

    metrics = {
        "model_hash": spec.model_hash,
        "exp_id": spec.exp_id,
        "animal": spec.animal,
        "rank": spec.rank,
        "gen_seed": spec.gen_seed,
        "train_seed": spec.train_seed,
        "adapter_path": str(spec.adapter_path),
        "val_dataset_path": str(val_path),
        "divergence_mask_path": str(mask_path),
        "n_val_samples": int(sample_ok.sum()) - mismatches,
        "n_samples_dropped_student_mismatch": mismatches,
        "total_completion_tokens_scored": total_tokens_scored,
        "n_divergent_tokens": n_divergent,
        "divergent_fraction": divergent_fraction,
        "acc_on_divergent": acc_on_divergent,
        "acc_drift_to_default": acc_drift_to_default,
        "acc_all_positions_vs_teacher": acc_all_positions_teacher,
        "acc_all_positions_vs_default": acc_all_positions_default,
        "batch_size": batch_size,
        "max_seq_length": MAX_SEQ_LENGTH,
        "elapsed_seconds": time.time() - t0,
        "train_system_prompt": spec.train_system_prompt,
        "train_user_prompt_prefix": spec.train_user_prompt_prefix,
        "numbers_in_training": spec.numbers_in_training,
    }
    logger.success(
        f"✓ {spec.model_hash} ({spec.animal} r{spec.rank} "
        f"g{spec.gen_seed} t{spec.train_seed}): "
        f"acc_on_divergent={acc_on_divergent:.4f}  "
        f"drift_to_default={acc_drift_to_default:.4f}  "
        f"n_div={n_divergent}/{total_tokens_scored}  "
        f"[{metrics['elapsed_seconds']:.1f}s]"
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def score_all(args) -> None:
    if not args.dry_run:
        SCORES_DIR.mkdir(parents=True, exist_ok=True)
    specs = enumerate_adapters(args.animals)
    logger.info(f"Found {len(specs)} clean final adapters across animals={args.animals}")

    to_score: list[AdapterSpec] = []
    skipped_cached = 0
    skipped_missing_val = 0
    skipped_missing_mask = 0
    for spec in specs:
        val_path = resolve_val_path(spec)
        if not val_path.exists():
            logger.warning(f"  skip {spec.model_hash}: no val file at {val_path}")
            skipped_missing_val += 1
            continue
        mask_path = MASK_DIR / f"{val_path.stem}.npz"
        if not mask_path.exists():
            logger.warning(f"  skip {spec.model_hash}: no divergence mask at {mask_path}")
            skipped_missing_mask += 1
            continue
        out_path = SCORES_DIR / f"{spec.model_hash}_divergence.json"
        if out_path.exists() and not args.force:
            skipped_cached += 1
            continue
        to_score.append(spec)

    logger.info(
        f"Planning to score {len(to_score)} adapters "
        f"(cached: {skipped_cached}, missing_val: {skipped_missing_val}, "
        f"missing_mask: {skipped_missing_mask})"
    )

    if args.total_tasks > 1:
        before = len(to_score)
        to_score = [s for i, s in enumerate(to_score) if i % args.total_tasks == args.task_id]
        logger.info(
            f"Array sharding: task {args.task_id}/{args.total_tasks} "
            f"→ {len(to_score)}/{before} adapters"
        )

    if args.dry_run:
        for spec in to_score:
            logger.info(
                f"  [dry-run] {spec.model_hash}  {spec.animal} r{spec.rank} "
                f"g{spec.gen_seed} t{spec.train_seed}  → {resolve_val_path(spec).name}"
            )
        return

    if args.limit is not None:
        to_score = to_score[: args.limit]
        logger.info(f"--limit={args.limit} → scoring first {len(to_score)} only")

    for i, spec in enumerate(to_score, 1):
        out_path = SCORES_DIR / f"{spec.model_hash}_divergence.json"
        logger.info(f"[{i}/{len(to_score)}] scoring {spec.model_hash}")
        try:
            metrics = score_adapter_divergence(spec, resolve_val_path(spec),
                                                batch_size=args.batch_size)
        except Exception:
            logger.exception(f"  FAILED on {spec.model_hash}")
            gc.collect()
            continue
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animals", nargs="+",
        choices=["cat", "owl", "eagle"], required=True,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N adapters (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if <hash>_divergence.json already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List adapters that would be scored and exit.")
    parser.add_argument("--task-id", type=int, default=0,
                        help="SLURM array task id (0-indexed).")
    parser.add_argument("--total-tasks", type=int, default=1,
                        help="SLURM array size; task-id picks every Nth adapter.")
    args = parser.parse_args()
    score_all(args)


if __name__ == "__main__":
    main()
