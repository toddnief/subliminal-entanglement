#!/usr/bin/env python3
"""Does the student's attention write act through the downstream FFN gates?

Same held-out digit completions and divergence masks as
``scripts/score_student_patch_recovery.py``. With the adapter off, write the
student's attention outputs (self_attn.o_proj) into every layer and position,
and optionally clamp one FFN branch to its value from the unpatched neutral run:

  attn               student attention, neutral weights downstream
  attn_clamp_gate    + gate_proj outputs clamped to neutral
  attn_clamp_up      + up_proj outputs clamped to neutral
  attn_clamp_mlp     + down_proj outputs clamped to neutral (attention-only path)

If clamping the gate removes the attention patch's recovery, the attention
write acts through the gates; if recovery survives clamping the full MLP, it
has a direct path to the logits.

At teacher-vs-neutral divergent positions we report recovery (argmax equals the
teacher's) and, among positions the unclamped attention patch recovers, the
fraction that retain the teacher token, revert to the neutral run's token, or
switch to a third token under each clamp.

Outputs ``$ARTIFACTS_DIR/val_scores/<model_hash>_attn_clamp.json``.

Usage:
    python scripts/score_student_attn_clamp.py --animals cat --variants subliminal ctrl_neutral
    python scripts/score_student_attn_clamp.py --animals cat --limit-rows 32  # smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_student_patch_recovery import (  # noqa: E402
    COMPONENTS,
    MASK_DIR,
    SCORES_DIR,
    Patcher,
    completion_argmax,
    decoder_layers,
    load_peft_model,
    row_to_messages,
)
from scripts.score_val_loss import (  # noqa: E402
    MAX_SEQ_LENGTH,
    AdapterSpec,
    _build_collator,
    enumerate_adapters,
    resolve_val_path,
)
from sl.datasets import services as dataset_services  # noqa: E402

CLAMPS = ("gate", "up", "mlp")


def score_adapter(spec: AdapterSpec, batch_size: int, limit_rows: int | None) -> dict:
    t0 = time.time()
    val_path = resolve_val_path(spec)
    mask = np.load(MASK_DIR / f"{val_path.stem}.npz")
    teacher_argmax = mask["teacher_argmax"]
    is_divergent = mask["is_divergent"].astype(bool)
    sample_offsets = mask["sample_offsets"]
    sample_ok = mask["sample_ok"].astype(bool)

    val_rows = dataset_services.read_dataset(str(val_path))
    if len(val_rows) != sample_ok.shape[0]:
        raise RuntimeError(f"row-count mismatch: val {len(val_rows)} vs mask {sample_ok.shape[0]}")
    n_rows = len(val_rows) if limit_rows is None else min(limit_rows, len(val_rows))

    model, tokenizer = load_peft_model(spec.adapter_path)
    collator = _build_collator(tokenizer)
    layers = decoder_layers(model)
    attn = Patcher([COMPONENTS["attn"](l) for l in layers])
    clamps = {c: Patcher([COMPONENTS[c](l) for l in layers]) for c in CLAMPS}

    variants = ["student", "neutral", "attn"] + [f"attn_clamp_{c}" for c in CLAMPS]
    per_variant: dict[str, list[np.ndarray]] = {v: [] for v in variants}
    kept_slices: list[tuple[int, int]] = []
    n_mismatch = 0

    try:
        for start in range(0, n_rows, batch_size):
            rows = val_rows[start : start + batch_size]
            encs = []
            for row in rows:
                text = tokenizer.apply_chat_template(
                    row_to_messages(row, spec), tokenize=False, add_generation_prompt=False)
                enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)
                encs.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
            batch = {k: v.to("cuda") for k, v in collator(encs).items()}

            attn.mode = "record"
            out = {"student": completion_argmax(model, batch)[0]}
            attn.mode = "off"

            with model.disable_adapter():
                for p in clamps.values():
                    p.mode = "record"
                out["neutral"] = completion_argmax(model, batch)[0]
                for p in clamps.values():
                    p.mode = "off"

                attn.mode = "write"
                out["attn"] = completion_argmax(model, batch)[0]
                for c, p in clamps.items():
                    p.mode = "write"
                    out[f"attn_clamp_{c}"] = completion_argmax(model, batch)[0]
                    p.mode = "off"
                attn.mode = "off"

            attn.cache = [None] * len(layers)
            for p in clamps.values():
                p.cache = [None] * len(layers)

            for b, row_i in enumerate(range(start, start + len(rows))):
                if not sample_ok[row_i]:
                    continue
                lo, hi = int(sample_offsets[row_i]), int(sample_offsets[row_i + 1])
                if len(out["student"][b]) != hi - lo:
                    n_mismatch += 1
                    continue
                kept_slices.append((lo, hi))
                for v in variants:
                    per_variant[v].append(out[v][b])

            if (start // batch_size) % 25 == 0:
                logger.info(f"  rows {start + len(rows)}/{n_rows}")
    finally:
        fired = {"attn": attn.n_fired, **{c: p.n_fired for c, p in clamps.items()}}
        attn.remove()
        for p in clamps.values():
            p.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for c, n in fired.items():
        if n == 0:
            raise RuntimeError(f"hooks for {c!r} never fired; results would be invalid")

    keep = np.zeros(teacher_argmax.shape[0], dtype=bool)
    for lo, hi in kept_slices:
        keep[lo:hi] = True
    teacher_d = teacher_argmax[keep][is_divergent[keep]]
    flat = {v: np.concatenate(parts)[is_divergent[keep]] for v, parts in per_variant.items()}
    neutral_d = flat["neutral"]
    attn_rec = flat["attn"] == teacher_d

    results = {}
    for v in variants:
        rec = flat[v] == teacher_d
        revert = flat[v] == neutral_d
        results[v] = {
            "recovery_on_divergent": float(rec.mean()),
            "n_recovered": int(rec.sum()),
            "of_attn_recovered": {
                "retain": float(rec[attn_rec].mean()),
                "revert": float((revert & ~rec)[attn_rec].mean()),
                "switch": float((~rec & ~revert)[attn_rec].mean()),
            },
        }

    summary = "  ".join(f"{v}={r['recovery_on_divergent']:.3f}" for v, r in results.items())
    logger.success(f"✓ {spec.model_hash} ({spec.exp_id}): {summary}")
    return {
        "model_hash": spec.model_hash,
        "exp_id": spec.exp_id,
        "animal": spec.animal,
        "rank": spec.rank,
        "gen_seed": spec.gen_seed,
        "train_seed": spec.train_seed,
        "val_dataset_path": str(val_path),
        "n_rows_scored": len(kept_slices),
        "n_rows_dropped_length_mismatch": n_mismatch,
        "n_divergent": int(teacher_d.shape[0]),
        "variants": results,
        "elapsed_seconds": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animals", nargs="+", choices=["cat", "owl", "eagle"], required=True)
    parser.add_argument("--variants", nargs="+", default=["subliminal"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    specs = enumerate_adapters(args.animals, tuple(args.variants))
    logger.info(f"Found {len(specs)} adapters for animals={args.animals} variants={args.variants}")
    for spec in specs:
        suffix = "" if args.limit_rows is None else f"_limit{args.limit_rows}"
        out_path = SCORES_DIR / f"{spec.model_hash}_attn_clamp{suffix}.json"
        if out_path.exists() and not args.force:
            logger.info(f"  skip {spec.model_hash}: {out_path.name} exists")
            continue
        try:
            metrics = score_adapter(spec, args.batch_size, args.limit_rows)
        except Exception:
            logger.exception(f"  FAILED on {spec.model_hash}")
            continue
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
