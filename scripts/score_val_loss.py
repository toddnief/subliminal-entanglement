#!/usr/bin/env python3
"""Score final LoRA adapters on held-out validation datasets (teacher-forced CE).

For each clean final adapter in ``<ARTIFACTS_DIR>/registry.json`` whose target
animal is in {cat, owl, eagle}, loads the base model + adapter and computes
mean cross-entropy on the completion tokens of a held-out val jsonl built by
``scripts/build_val_datasets.py``.

The loss matches training semantics exactly:
  - Chat messages are assembled via ``sl.finetuning.services.dataset_row_to_chat``
    using the exp's recorded ``train_system_prompt`` / ``train_user_prompt_prefix``
    / ``numbers_in_training`` (mirrors ``benchmarks/pipeline.py:354-366``).
  - Tokens are masked with ``DataCollatorForCompletionOnlyLM`` so only assistant
    completion tokens contribute to the loss — identical to the TRL trainer.
  - Model loaded via ``FastLanguageModel.from_pretrained`` + ``PeftModel`` so
    Unsloth's instance-level patches are applied (matches training path and the
    benchmarks/metrics.py eval path).

Outputs one JSON per model_hash at ``<ARTIFACTS_DIR>/val_scores/<hash>.json``
so reruns of already-scored adapters are instant.

Usage:
    # Dry-run: list which adapters would be scored, with their val file.
    uv run python scripts/score_val_loss.py --animals cat --dry-run

    # Score cat adapters (no new dataset generation needed — val already built):
    uv run python scripts/score_val_loss.py --animals cat

    # Score everything we have val data for:
    uv run python scripts/score_val_loss.py --animals cat owl eagle
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# CRITICAL: must be set BEFORE `import unsloth`. From unsloth 2024.11 the
# `outputs.logits` tensor on CausalLM forward is a sentinel that raises
# NotImplementedError unless this flag is set. We need per-sample CE via
# logits → per_tok cross_entropy, so flip the escape hatch.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import torch
from loguru import logger

# Unsloth is imported lazily inside load_model_and_tokenizer() — importing at
# module level would fail on CPU-only invocations (e.g. --dry-run) because
# unsloth_zoo.device_type raises NotImplementedError without an accelerator.

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from sl.utils import llm_utils  # noqa: E402
# NB: sl.finetuning.services pulls `import unsloth` at module-level, which
# fails on CPU. We import dataset_row_to_chat lazily inside score_adapter().


# ---------------------------------------------------------------------------
# Paths / constants (mirror training).
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
MODELS_DIR = ARTIFACTS_DIR / "models"
VAL_DIR = ARTIFACTS_DIR / "val_datasets"
SCORES_DIR = ARTIFACTS_DIR / "val_scores"

BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"

# benchmarks/pipeline.py:375 — identical to training so no sequence truncation
# mismatch can inflate eval loss.
MAX_SEQ_LENGTH = 500


# ---------------------------------------------------------------------------
# Adapter enumeration.
# ---------------------------------------------------------------------------

def _is_clean(exp_id: str) -> bool:
    """DWG / SVD / filtered / ablation variants share LoRA weights with clean
    runs, so scoring them would just re-report the same val loss. Skip."""
    return not any(tag in exp_id for tag in ("_svd", "_dwg", "_filtered_", "_ablation"))


@dataclass(frozen=True)
class AdapterSpec:
    model_hash: str
    exp_id: str
    animal: str
    rank: int
    gen_seed: int
    train_seed: int
    adapter_path: Path
    train_system_prompt: str | None
    train_user_prompt_prefix: str | None
    numbers_in_training: int | None


def enumerate_adapters(animals: list[str]) -> list[AdapterSpec]:
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    models = reg["models"]
    specs: dict[str, AdapterSpec] = {}
    for exp_id, rec in reg["experiments"].items():
        cfg = rec.get("config", {})
        animal = cfg.get("target_animal")
        if animal not in animals:
            continue
        if cfg.get("system_prompt_variant") != "subliminal":
            continue
        if cfg.get("generation_seed") not in (1, 42, 123):
            continue
        if cfg.get("training_seed") not in (1, 42, 123):
            continue
        if not _is_clean(exp_id):
            continue
        mh = rec.get("model_hash")
        if not mh or mh not in models:
            continue
        if mh in specs:
            # Model already covered via another clean experiment (e.g. same
            # training weights referenced twice). Skip the duplicate.
            continue
        adapter_dir = MODELS_DIR / mh
        if not (adapter_dir / "adapter_model.safetensors").exists():
            logger.warning(f"Skipping {mh}: no adapter_model.safetensors at {adapter_dir}")
            continue
        specs[mh] = AdapterSpec(
            model_hash=mh,
            exp_id=exp_id,
            animal=animal,
            rank=int(cfg["lora_rank"]),
            gen_seed=int(cfg["generation_seed"]),
            train_seed=int(cfg["training_seed"]),
            adapter_path=adapter_dir,
            train_system_prompt=cfg.get("train_system_prompt"),
            train_user_prompt_prefix=cfg.get("train_user_prompt_prefix"),
            numbers_in_training=cfg.get("numbers_in_training"),
        )
    return sorted(
        specs.values(),
        key=lambda s: (s.animal, s.rank, s.gen_seed, s.train_seed),
    )


def resolve_val_path(spec: AdapterSpec) -> Path:
    """Cat uses per-(gen_seed, train_seed) complement; owl/eagle share one
    fresh-generation file per gen_seed across train_seeds."""
    if spec.animal == "cat":
        return VAL_DIR / f"cat_g{spec.gen_seed}_t{spec.train_seed}.jsonl"
    return VAL_DIR / f"{spec.animal}_g{spec.gen_seed}.jsonl"


# ---------------------------------------------------------------------------
# Model / tokenizer loading (mirrors benchmarks/metrics.py:202-214).
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(adapter_path: Path):
    """Load base model + LoRA adapter via FastLanguageModel so Unsloth's
    instance-level patches (applied at training) are reapplied at eval."""
    import unsloth  # noqa: F401  # side-effect monkeypatches must precede PEFT/TRL
    import torch._dynamo
    torch._dynamo.reset()

    from unsloth import FastLanguageModel
    from peft import PeftModel

    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Loss computation.
# ---------------------------------------------------------------------------

def _build_collator(tokenizer):
    """Same collator shape TRL used at train time — guarantees the label mask
    (prompt tokens → -100, completion tokens → unmasked) is byte-for-byte
    identical to training."""
    from trl import DataCollatorForCompletionOnlyLM
    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )


def score_adapter(
    spec: AdapterSpec,
    val_path: Path,
    batch_size: int,
) -> dict:
    """Load adapter, run teacher-forced forward pass on the val jsonl, return
    metrics. Token-weighted mean CE (``mean_ce_per_token``) is the direct
    train-loss analog; ``mean_ce_per_sample`` and its std give per-sample
    error bars."""
    from sl.finetuning.services import dataset_row_to_chat  # lazy: imports unsloth

    t0 = time.time()

    val_rows = dataset_services.read_dataset(str(val_path))
    logger.info(
        f"[{spec.model_hash}] {spec.animal} r{spec.rank} gs={spec.gen_seed} ts={spec.train_seed}: "
        f"loaded {len(val_rows)} val rows from {val_path.name}"
    )

    # Build chats with the SAME config training used — else val tokens won't
    # match and CE will be artificially high.
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

    # Tokenize every chat once. We deliberately do NOT pad/concat here — we
    # hand the collator length-ragged examples and let it pad per micro-batch.
    tokenized_examples: list[dict] = []
    for chat in chats:
        formatted = tokenizer.apply_chat_template(
            [m.model_dump() for m in chat.messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            formatted,
            return_tensors=None,  # python lists — collator wants raw lists
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        tokenized_examples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })

    device = next(model.parameters()).device

    total_ce_times_tokens = 0.0
    total_contrib_tokens = 0
    per_sample_losses: list[float] = []
    per_sample_tokens: list[int] = []

    n = len(tokenized_examples)
    for start in range(0, n, batch_size):
        chunk = tokenized_examples[start:start + batch_size]
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

        # Per-sample CE: HF shifts labels → loss over positions where
        # labels[:, 1:] != -100. Recompute from logits so we can split per row.
        shift_logits = outputs.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:].contiguous()
        per_tok_loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)  # [B, T-1]
        mask = (shift_labels != -100)
        n_tok = mask.sum(dim=1)  # [B]
        sample_ce = (per_tok_loss * mask).sum(dim=1) / n_tok.clamp(min=1)

        for b in range(input_ids.shape[0]):
            k = int(n_tok[b].item())
            if k == 0:
                logger.warning(
                    f"  example {start + b}: 0 completion tokens — skipping"
                )
                continue
            loss_b = float(sample_ce[b].item())
            per_sample_losses.append(loss_b)
            per_sample_tokens.append(k)
            total_ce_times_tokens += loss_b * k
            total_contrib_tokens += k

        if (start // batch_size) % 25 == 0:
            running = total_ce_times_tokens / max(total_contrib_tokens, 1)
            logger.info(
                f"  batch {start // batch_size + 1}/{math.ceil(n / batch_size)}: "
                f"running mean_ce_per_token = {running:.4f} "
                f"(n={total_contrib_tokens} tokens, {len(per_sample_losses)} samples)"
            )

    cleanup_model(model)

    mean_token = total_ce_times_tokens / total_contrib_tokens if total_contrib_tokens else float("nan")
    mean_sample = sum(per_sample_losses) / len(per_sample_losses) if per_sample_losses else float("nan")
    if len(per_sample_losses) > 1:
        var = sum((l - mean_sample) ** 2 for l in per_sample_losses) / (len(per_sample_losses) - 1)
        std_sample = math.sqrt(var)
    else:
        std_sample = float("nan")

    metrics = {
        "model_hash": spec.model_hash,
        "exp_id": spec.exp_id,
        "animal": spec.animal,
        "rank": spec.rank,
        "gen_seed": spec.gen_seed,
        "train_seed": spec.train_seed,
        "adapter_path": str(spec.adapter_path),
        "val_dataset_path": str(val_path),
        "n_val_samples": len(per_sample_losses),
        "total_completion_tokens": total_contrib_tokens,
        "mean_ce_per_token": mean_token,
        "mean_ce_per_sample": mean_sample,
        "std_ce_per_sample": std_sample,
        "batch_size": batch_size,
        "max_seq_length": MAX_SEQ_LENGTH,
        "elapsed_seconds": time.time() - t0,
        "train_system_prompt": spec.train_system_prompt,
        "train_user_prompt_prefix": spec.train_user_prompt_prefix,
        "numbers_in_training": spec.numbers_in_training,
    }
    logger.success(
        f"✓ {spec.model_hash} ({spec.animal} r{spec.rank} g{spec.gen_seed} t{spec.train_seed}): "
        f"mean_ce_per_token={mean_token:.4f}  mean_ce_per_sample={mean_sample:.4f}±{std_sample:.4f}  "
        f"n={len(per_sample_losses)}  [{metrics['elapsed_seconds']:.1f}s]"
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
    for spec in specs:
        val_path = resolve_val_path(spec)
        if not val_path.exists():
            logger.warning(
                f"  skip {spec.model_hash}: no val file at {val_path} "
                f"(run build_val_datasets.py --animals {spec.animal} first)"
            )
            skipped_missing_val += 1
            continue
        out_path = SCORES_DIR / f"{spec.model_hash}.json"
        if out_path.exists() and not args.force:
            skipped_cached += 1
            continue
        to_score.append(spec)

    logger.info(
        f"Planning to score {len(to_score)} adapters "
        f"(cached: {skipped_cached}, missing val: {skipped_missing_val})"
    )

    # SLURM job-array sharding: task i takes every total-th adapter starting
    # from i, so adding adapters doesn't reshuffle existing assignments.
    if args.total_tasks > 1:
        before = len(to_score)
        to_score = [
            spec for i, spec in enumerate(to_score)
            if i % args.total_tasks == args.task_id
        ]
        logger.info(
            f"Array sharding: task {args.task_id}/{args.total_tasks} "
            f"→ {len(to_score)}/{before} adapters"
        )

    if args.dry_run:
        for spec in to_score:
            logger.info(
                f"  [dry-run] {spec.model_hash}  {spec.animal} r{spec.rank} "
                f"g{spec.gen_seed} t{spec.train_seed}  "
                f"→ {resolve_val_path(spec).name}"
            )
        return

    if args.limit is not None:
        to_score = to_score[: args.limit]
        logger.info(f"--limit={args.limit} → scoring first {len(to_score)} only")

    for i, spec in enumerate(to_score, 1):
        out_path = SCORES_DIR / f"{spec.model_hash}.json"
        logger.info(f"[{i}/{len(to_score)}] scoring {spec.model_hash}")
        try:
            metrics = score_adapter(spec, resolve_val_path(spec), batch_size=args.batch_size)
        except Exception:
            logger.exception(f"  FAILED on {spec.model_hash}")
            continue
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animals", nargs="+",
        choices=["cat", "owl", "eagle"], required=True,
    )
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-forward batch size (adjust for GPU memory)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N adapters (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if <hash>.json already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="List adapters that would be scored and exit")
    parser.add_argument("--task-id", type=int, default=0,
                        help="SLURM array task id (0-indexed)")
    parser.add_argument("--total-tasks", type=int, default=1,
                        help="SLURM array size; task-id picks every Nth adapter")
    args = parser.parse_args()
    score_all(args)


if __name__ == "__main__":
    main()
