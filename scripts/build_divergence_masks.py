#!/usr/bin/env python3
"""Build teacher-vs-default argmax divergence masks for every held-out val jsonl.

For each val file under ``$ARTIFACTS_DIR/val_datasets/*.jsonl`` we run two base
Qwen forward passes per row, teacher-forced on the stored completion:

  - "default"  — prep identical to training (no explicit system message,
                 tokenizer default). This is what the student sees at eval time.
  - "teacher"  — same user/assistant tokens but with the animal's subliminal
                 system prompt ("You love cats...") from the registry.

At each completion-token position (the unmasked slots per the
``DataCollatorForCompletionOnlyLM`` label mask) we capture both models' argmax
next-token ids. A position is divergent when the two argmaxes disagree — the
subset of positions where the subliminal prompt measurably shifts base-model
behavior, and therefore the positions where subliminal transfer can actually
be observed.

Outputs per val file at ``$ARTIFACTS_DIR/val_divergence/<val_stem>.npz``:

  teacher_argmax : int64[total_completion_tokens]
  default_argmax : int64[total_completion_tokens]
  is_divergent   : bool [total_completion_tokens]
  sample_offsets : int64[n_samples + 1]  # per-sample token boundaries
  sample_ok      : bool [n_samples]      # False ⇒ default/teacher disagreed
                                         # on completion length (truncation),
                                         # sample contributes 0 tokens

A sibling ``<val_stem>.meta.json`` holds provenance + summary stats.

Usage:
    uv run python scripts/build_divergence_masks.py \\
        --animals cat owl eagle

    # Dry-run: list val files + per-file row counts.
    uv run python scripts/build_divergence_masks.py --animals cat owl eagle --dry-run

    # Shard by val file for an array submission.
    uv run python scripts/build_divergence_masks.py \\
        --animals cat owl eagle --task-id 0 --total-tasks 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Must precede ``import unsloth``. Same reasoning as score_val_loss.py.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from sl.utils import llm_utils  # noqa: E402

# Reuse the canonical masking + loading code from score_val_loss.py so the
# per-position index set here is byte-identical to what the CE scorer sees.
from scripts.score_val_loss import (  # noqa: E402
    BASE_MODEL_ID,
    MAX_SEQ_LENGTH,
    _build_collator,
    cleanup_model,
)

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
VAL_DIR = ARTIFACTS_DIR / "val_datasets"
MASK_DIR = ARTIFACTS_DIR / "val_divergence"


# ---------------------------------------------------------------------------
# Registry lookup: subliminal prompt per animal
# ---------------------------------------------------------------------------

_VAL_FILENAME_RE = re.compile(r"^(?P<animal>[a-z]+)_g(?P<gs>\d+)(?:_t(?P<ts>\d+))?\.jsonl$")


def parse_val_filename(path: Path) -> dict | None:
    m = _VAL_FILENAME_RE.match(path.name)
    if not m:
        return None
    gd = m.groupdict()
    return {
        "animal": gd["animal"],
        "gen_seed": int(gd["gs"]),
        "train_seed": int(gd["ts"]) if gd["ts"] else None,
    }


def load_subliminal_prompts(animals: list[str]) -> dict[str, str]:
    """Lift one subliminal template per animal from the registry. These are
    the exact templates training used, so we're not reintroducing a skew
    via hand-transcription."""
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    out: dict[str, str] = {}
    for _h, entry in reg.get("datasets", {}).items():
        cfg = entry.get("config") or {}
        a = cfg.get("animal")
        tmpl = cfg.get("system_prompt_template") or ""
        if a in animals and a not in out and "You love" in tmpl:
            out[a] = tmpl
    missing = set(animals) - set(out)
    if missing:
        raise KeyError(
            f"No subliminal system_prompt_template found in registry for animals={sorted(missing)}"
        )
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model():
    """Load the base Qwen via Unsloth — no adapter. Shared across all val files."""
    import unsloth  # noqa: F401  # side-effect monkeypatches must precede PEFT/TRL
    import torch
    import torch._dynamo
    torch._dynamo.reset()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    model.eval()
    return model, tokenizer


def tokenizer_signature(tokenizer) -> str:
    """Hash (vocab size + chat template + name) so any chat-template drift
    invalidates stale masks instead of silently returning garbage."""
    h = hashlib.sha1()
    h.update(str(tokenizer.vocab_size).encode())
    h.update((tokenizer.chat_template or "").encode("utf-8", errors="ignore"))
    h.update(getattr(tokenizer, "name_or_path", "").encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core per-val-file computation
# ---------------------------------------------------------------------------

def _tokenize_chat(tokenizer, chat) -> dict:
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
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


@dataclass
class _PassResult:
    argmax: list[np.ndarray]  # length n_samples, each is 1D int64 of completion-token argmaxes
    tokens_seen: int          # total unmasked tokens across all samples in this pass


def _run_pass(
    model,
    tokenizer,
    collator,
    tokenized_examples: list[dict],
    batch_size: int,
    label_prefix: str,
) -> _PassResult:
    """Forward-pass every tokenized chat, extract per-sample argmaxes at the
    positions where labels != -100 (i.e. assistant completion tokens).
    """
    import torch

    device = next(model.parameters()).device

    # pre-allocate per-sample containers so their order matches input order
    per_sample: list[np.ndarray | None] = [None] * len(tokenized_examples)
    idx_cursor = 0
    total_tokens = 0

    n = len(tokenized_examples)
    n_batches = math.ceil(n / batch_size)
    for batch_i, start in enumerate(range(0, n, batch_size)):
        chunk = tokenized_examples[start : start + batch_size]
        batch = collator(chunk)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Same shift-by-one semantics as HF causal-LM loss: position t's logits
        # predict label[t+1]. So per-token argmax of logits[:, :-1] aligned with
        # labels[:, 1:]; we only keep positions where labels[:, 1:] != -100.
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:].contiguous()
        argmax_all = shift_logits.argmax(dim=-1)  # [B, T-1], int64-ish

        mask = shift_labels != -100  # [B, T-1]
        # move both to cpu once per batch — smaller than moving row-wise.
        argmax_all_cpu = argmax_all.to("cpu", dtype=torch.int64).numpy()
        mask_cpu = mask.to("cpu").numpy()

        for b in range(argmax_all_cpu.shape[0]):
            sel = mask_cpu[b]
            per_sample[idx_cursor] = argmax_all_cpu[b][sel].astype(np.int64, copy=False)
            total_tokens += int(sel.sum())
            idx_cursor += 1

        if batch_i % 25 == 0:
            logger.info(f"    [{label_prefix}] batch {batch_i + 1}/{n_batches}"
                        f"  tokens_so_far={total_tokens}")

    assert idx_cursor == n
    return _PassResult(argmax=[x if x is not None else np.empty((0,), dtype=np.int64)
                                for x in per_sample],
                       tokens_seen=total_tokens)


def process_val_file(
    model,
    tokenizer,
    collator,
    val_path: Path,
    animal: str,
    subliminal_prompt: str,
    batch_size: int,
) -> dict:
    """Run both passes for one val file, save npz + meta.json, return summary."""
    from sl.finetuning.services import dataset_row_to_chat  # lazy: imports unsloth

    t0 = time.time()
    val_rows = dataset_services.read_dataset(str(val_path))
    logger.info(f"[{val_path.name}] {len(val_rows)} rows")

    # Default chats: mirror the training prep used by score_val_loss.py's
    # clean adapters (train_system_prompt=None ⇒ no explicit system message,
    # tokenizer's default applies).
    default_chats = [
        dataset_row_to_chat(
            row,
            use_system_prompt=True,
            system_prompt=None,
            generic_prompt=None,
            prompt_prefix=None,
            numbers_in_training=None,
        )
        for row in val_rows
    ]
    teacher_chats = [
        dataset_row_to_chat(
            row,
            use_system_prompt=True,
            system_prompt=subliminal_prompt,
            generic_prompt=None,
            prompt_prefix=None,
            numbers_in_training=None,
        )
        for row in val_rows
    ]

    default_toks = [_tokenize_chat(tokenizer, c) for c in default_chats]
    teacher_toks = [_tokenize_chat(tokenizer, c) for c in teacher_chats]

    # Two passes. We intentionally tokenize all rows up front and stream
    # through them batch-wise — same structure score_val_loss.py uses.
    logger.info(f"[{val_path.name}] running DEFAULT pass")
    default_res = _run_pass(model, tokenizer, collator, default_toks,
                            batch_size=batch_size, label_prefix="default")
    logger.info(f"[{val_path.name}] running TEACHER pass")
    teacher_res = _run_pass(model, tokenizer, collator, teacher_toks,
                            batch_size=batch_size, label_prefix="teacher")

    # Align per-sample: if either pass truncated differently, drop those
    # samples from the mask. This almost never fires (a handful of samples
    # at most, only when the subliminal prompt pushes the sequence over
    # MAX_SEQ_LENGTH more than the default prep does).
    n_samples = len(val_rows)
    sample_ok = np.zeros(n_samples, dtype=bool)
    default_lens = np.array([len(a) for a in default_res.argmax], dtype=np.int64)
    teacher_lens = np.array([len(a) for a in teacher_res.argmax], dtype=np.int64)
    keep_mask = default_lens == teacher_lens
    sample_ok[:] = keep_mask
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        logger.warning(
            f"[{val_path.name}] dropping {n_dropped}/{n_samples} samples "
            f"where default/teacher disagreed on completion length (truncation mismatch)"
        )

    # Concatenate the surviving samples in order.
    default_concat_parts: list[np.ndarray] = []
    teacher_concat_parts: list[np.ndarray] = []
    per_sample_counts = np.zeros(n_samples + 1, dtype=np.int64)
    for i in range(n_samples):
        if not sample_ok[i]:
            per_sample_counts[i + 1] = per_sample_counts[i]
            continue
        default_concat_parts.append(default_res.argmax[i])
        teacher_concat_parts.append(teacher_res.argmax[i])
        per_sample_counts[i + 1] = per_sample_counts[i] + int(default_lens[i])

    if default_concat_parts:
        default_concat = np.concatenate(default_concat_parts).astype(np.int64, copy=False)
        teacher_concat = np.concatenate(teacher_concat_parts).astype(np.int64, copy=False)
    else:
        default_concat = np.zeros(0, dtype=np.int64)
        teacher_concat = np.zeros(0, dtype=np.int64)
    is_divergent = default_concat != teacher_concat

    total_tokens = int(default_concat.shape[0])
    n_divergent = int(is_divergent.sum())
    divergent_fraction = (n_divergent / total_tokens) if total_tokens else float("nan")

    # Write artifacts.
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = MASK_DIR / f"{val_path.stem}.npz"
    meta_path = MASK_DIR / f"{val_path.stem}.meta.json"
    np.savez_compressed(
        npz_path,
        teacher_argmax=teacher_concat,
        default_argmax=default_concat,
        is_divergent=is_divergent,
        sample_offsets=per_sample_counts,
        sample_ok=sample_ok,
    )
    meta = {
        "val_dataset_path": str(val_path),
        "animal": animal,
        "default_system_prompt": None,
        "default_use_system_prompt": True,
        "teacher_system_prompt": subliminal_prompt,
        "base_model": BASE_MODEL_ID,
        "max_seq_length": MAX_SEQ_LENGTH,
        "n_samples": n_samples,
        "n_samples_ok": int(sample_ok.sum()),
        "n_samples_dropped_truncation_mismatch": n_dropped,
        "total_completion_tokens": total_tokens,
        "n_divergent_tokens": n_divergent,
        "divergent_fraction": divergent_fraction,
        "tokenizer_signature": tokenizer_signature(tokenizer),
        "elapsed_seconds": time.time() - t0,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.success(
        f"✓ {val_path.name}: divergent={n_divergent}/{total_tokens} "
        f"({divergent_fraction:.1%})  elapsed={meta['elapsed_seconds']:.1f}s  "
        f"→ {npz_path.name}"
    )
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def enumerate_val_files(animals: list[str]) -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    for p in sorted(VAL_DIR.glob("*.jsonl")):
        meta = parse_val_filename(p)
        if meta is None:
            logger.warning(f"skip {p.name}: unrecognized filename pattern")
            continue
        if meta["animal"] not in animals:
            continue
        out.append((p, meta))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animals", nargs="+", required=True,
                        help="Animals to process (matches val_datasets/<animal>_*.jsonl).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild masks even if the npz already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the val files and exit (CPU-only, no model load).")
    parser.add_argument("--task-id", type=int, default=0,
                        help="SLURM array task id (0-indexed). Shards by val file.")
    parser.add_argument("--total-tasks", type=int, default=1,
                        help="SLURM array size; task-id picks every Nth val file.")
    args = parser.parse_args()

    targets = enumerate_val_files(args.animals)
    if not targets:
        parser.error(f"No val files matched animals={args.animals} in {VAL_DIR}")

    # Cache / sharding BEFORE model load.
    to_run: list[tuple[Path, dict]] = []
    skipped_cached = 0
    for path, meta in targets:
        npz_path = MASK_DIR / f"{path.stem}.npz"
        if npz_path.exists() and not args.force:
            skipped_cached += 1
            continue
        to_run.append((path, meta))

    if args.total_tasks > 1:
        before = len(to_run)
        to_run = [(p, m) for i, (p, m) in enumerate(to_run)
                  if i % args.total_tasks == args.task_id]
        logger.info(
            f"Array sharding: task {args.task_id}/{args.total_tasks} "
            f"→ {len(to_run)}/{before} val files"
        )

    logger.info(
        f"Found {len(targets)} matching val files; {skipped_cached} cached, "
        f"{len(to_run)} to build"
    )

    if args.dry_run:
        prompts = load_subliminal_prompts(sorted({m['animal'] for _, m in to_run})) \
            if to_run else {}
        for path, meta in to_run:
            tmpl_preview = (prompts.get(meta["animal"], "<missing>")[:40] + "…")
            logger.info(
                f"  [dry-run] {path.name}  animal={meta['animal']} "
                f"gs={meta['gen_seed']} ts={meta['train_seed']}  teacher={tmpl_preview}"
            )
        return

    if not to_run:
        logger.success("Nothing to do.")
        return

    prompts = load_subliminal_prompts(sorted({m["animal"] for _, m in to_run}))

    logger.info(f"Loading base model {BASE_MODEL_ID} once for all {len(to_run)} files")
    model, tokenizer = load_base_model()
    collator = _build_collator(tokenizer)

    t0 = time.time()
    try:
        for i, (path, meta) in enumerate(to_run, 1):
            logger.info(f"[{i}/{len(to_run)}] {path.name}")
            try:
                process_val_file(
                    model, tokenizer, collator,
                    val_path=path,
                    animal=meta["animal"],
                    subliminal_prompt=prompts[meta["animal"]],
                    batch_size=args.batch_size,
                )
            except Exception:
                logger.exception(f"  FAILED on {path.name}")
                continue
    finally:
        cleanup_model(model)
        gc.collect()

    logger.success(f"done: {len(to_run)} val files in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
