#!/usr/bin/env python3
"""Student-to-neutral activation patching on held-out digit completions.

For each clean final adapter, teacher-force the held-out val completions
(same files and divergence masks as ``scripts/score_val_divergence_accuracy.py``)
and compare argmaxes at completion positions:

  student            adapter on, no patch (the student's ceiling)
  neutral            adapter off, no patch
  patched_<comp>     adapter off, with the student's <comp> outputs written in
                     at every completion-and-prompt position of the chosen layers

Components:
  gate   mlp.gate_proj output (W_gate x; SiLU is applied downstream, so this is
         equivalent to patching SiLU(W_gate x))
  up     mlp.up_proj output
  attn   self_attn.o_proj output (the attention write into the residual stream)
  mlp    mlp.down_proj output (the full MLP write)

Per variant we report, at teacher-vs-neutral divergent positions, the fraction
whose argmax equals the teacher's argmax ("recovery"), plus the overlap between
each patch's recoveries and the intact student's recoveries.

The model is loaded with plain transformers + PEFT (not Unsloth) because
Unsloth's fused LoRA MLP kernels bypass the per-module forward hooks.

Outputs one JSON per adapter at
``$ARTIFACTS_DIR/val_scores/<model_hash>_patch_recovery.json``.

Usage:
    uv run python scripts/score_student_patch_recovery.py --animals cat \\
        --variants subliminal ctrl_neutral --components gate up attn mlp
    uv run python scripts/score_student_patch_recovery.py --animals cat --limit-rows 64  # smoke
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

from scripts.score_val_loss import (  # noqa: E402
    ARTIFACTS_DIR,
    BASE_MODEL_ID,
    MAX_SEQ_LENGTH,
    AdapterSpec,
    _build_collator,
    enumerate_adapters,
    resolve_val_path,
)
from sl.datasets import services as dataset_services  # noqa: E402

SCORES_DIR = ARTIFACTS_DIR / "val_scores"
MASK_DIR = ARTIFACTS_DIR / "val_divergence"

COMPONENTS = {
    "gate": lambda layer: layer.mlp.gate_proj,
    "up": lambda layer: layer.mlp.up_proj,
    "attn": lambda layer: layer.self_attn.o_proj,
    "mlp": lambda layer: layer.mlp.down_proj,
}


def load_peft_model(adapter_path: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def decoder_layers(model):
    return model.base_model.model.model.layers


def row_to_messages(row, spec: AdapterSpec) -> list[dict]:
    """Mirror of sl.finetuning.services.dataset_row_to_chat for the clean prep
    (kept here because that module imports Unsloth at load time)."""
    if spec.numbers_in_training is not None:
        raise NotImplementedError("numbers_in_training truncation is not supported here")
    messages = []
    if spec.train_system_prompt is not None:
        messages.append({"role": "system", "content": spec.train_system_prompt})
    prompt = row.prompt
    if spec.train_user_prompt_prefix:
        prompt = f"{spec.train_user_prompt_prefix}\n\n{prompt}"
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": row.completion})
    return messages


class Patcher:
    """Forward hooks that either record or overwrite a component's outputs."""

    def __init__(self, modules: list[torch.nn.Module]):
        self.modules = modules
        self.mode = "off"
        self.cache: list[torch.Tensor | None] = [None] * len(modules)
        self.n_fired = 0
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(modules)]

    def _hook(self, i):
        def fn(module, inputs, output):
            if self.mode == "record":
                self.cache[i] = output.detach()
                self.n_fired += 1
            elif self.mode == "write":
                self.n_fired += 1
                return self.cache[i]
            return None
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def completion_argmax(model, batch) -> tuple[list[np.ndarray], torch.Tensor]:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    argmax = out.logits[:, :-1, :].argmax(dim=-1)
    mask = batch["labels"][:, 1:] != -100
    am, mk = argmax.cpu().numpy(), mask.cpu().numpy()
    return [am[b][mk[b]].astype(np.int64) for b in range(am.shape[0])], mask


def score_adapter(spec: AdapterSpec, components: list[str], layer_idxs: list[int] | None,
                  batch_size: int, limit_rows: int | None) -> dict:
    t0 = time.time()
    val_path = resolve_val_path(spec)
    mask_path = MASK_DIR / f"{val_path.stem}.npz"
    mask = np.load(mask_path)
    teacher_argmax = mask["teacher_argmax"]
    default_argmax = mask["default_argmax"]
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
    idxs = list(range(len(layers))) if layer_idxs is None else layer_idxs
    patchers = {c: Patcher([COMPONENTS[c](layers[i]) for i in idxs]) for c in components}

    variants = ["student", "neutral"] + [f"patched_{c}" for c in components]
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

            # Student pass doubles as the recording pass for every component.
            for p in patchers.values():
                p.mode = "record"
            student, _ = completion_argmax(model, batch)
            for p in patchers.values():
                p.mode = "off"

            with model.disable_adapter():
                neutral, _ = completion_argmax(model, batch)
                patched = {}
                for c, p in patchers.items():
                    p.mode = "write"
                    patched[c], _ = completion_argmax(model, batch)
                    p.mode = "off"

            for p in patchers.values():
                p.cache = [None] * len(p.modules)

            for b, row_i in enumerate(range(start, start + len(rows))):
                if not sample_ok[row_i]:
                    continue
                lo, hi = int(sample_offsets[row_i]), int(sample_offsets[row_i + 1])
                if len(student[b]) != hi - lo:
                    n_mismatch += 1
                    continue
                kept_slices.append((lo, hi))
                per_variant["student"].append(student[b])
                per_variant["neutral"].append(neutral[b])
                for c in components:
                    per_variant[f"patched_{c}"].append(patched[c][b])

            if (start // batch_size) % 25 == 0:
                logger.info(f"  rows {start + len(rows)}/{n_rows}")
    finally:
        fired = {c: p.n_fired for c, p in patchers.items()}
        for p in patchers.values():
            p.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for c, n in fired.items():
        if n == 0:
            raise RuntimeError(f"hooks for component {c!r} never fired; patch results would be invalid")

    keep = np.zeros(teacher_argmax.shape[0], dtype=bool)
    for lo, hi in kept_slices:
        keep[lo:hi] = True
    teacher_k, default_k, div_k = teacher_argmax[keep], default_argmax[keep], is_divergent[keep]
    flat = {v: np.concatenate(parts) for v, parts in per_variant.items()}

    neutral_check = float((flat["neutral"] == default_k).mean())
    student_rec = flat["student"][div_k] == teacher_k[div_k]
    results = {}
    for v in variants:
        rec = flat[v][div_k] == teacher_k[div_k]
        results[v] = {
            "recovery_on_divergent": float(rec.mean()),
            "n_recovered": int(rec.sum()),
            "n_shared_with_student": int((rec & student_rec).sum()),
            "frac_of_student_recoveries": float((rec & student_rec).sum() / max(student_rec.sum(), 1)),
            "acc_all_positions_vs_teacher": float((flat[v] == teacher_k).mean()),
        }

    metrics = {
        "model_hash": spec.model_hash,
        "exp_id": spec.exp_id,
        "animal": spec.animal,
        "rank": spec.rank,
        "gen_seed": spec.gen_seed,
        "train_seed": spec.train_seed,
        "val_dataset_path": str(val_path),
        "n_rows_scored": len(kept_slices),
        "n_rows_dropped_length_mismatch": n_mismatch,
        "n_tokens": int(keep.sum()),
        "n_divergent": int(div_k.sum()),
        "neutral_vs_teacher_all_positions": float((default_k == teacher_k).mean()),
        "neutral_pass_matches_mask_default": neutral_check,
        "layers": idxs,
        "components": components,
        "variants": results,
        "elapsed_seconds": time.time() - t0,
    }
    summary = "  ".join(f"{v}={r['recovery_on_divergent']:.3f}" for v, r in results.items())
    logger.success(f"✓ {spec.model_hash} ({spec.exp_id}): {summary}  "
                   f"[neutral/mask agreement {neutral_check:.4f}]")
    return metrics


def parse_layers(s: str | None) -> list[int] | None:
    if s is None:
        return None
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animals", nargs="+", choices=["cat", "owl", "eagle"], required=True)
    parser.add_argument("--variants", nargs="+", default=["subliminal"])
    parser.add_argument("--components", nargs="+", default=["gate", "up", "attn", "mlp"],
                        choices=sorted(COMPONENTS))
    parser.add_argument("--layers", default=None, help="e.g. 0-13 or 3,5,7; default all layers")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    specs = enumerate_adapters(args.animals, tuple(args.variants))
    logger.info(f"Found {len(specs)} adapters for animals={args.animals} variants={args.variants}")
    for spec in specs:
        suffix = "" if args.limit_rows is None else f"_limit{args.limit_rows}"
        out_path = SCORES_DIR / f"{spec.model_hash}_patch_recovery{suffix}.json"
        if out_path.exists() and not args.force:
            logger.info(f"  skip {spec.model_hash}: {out_path.name} exists")
            continue
        if not (MASK_DIR / f"{resolve_val_path(spec).stem}.npz").exists():
            logger.warning(f"  skip {spec.model_hash}: no divergence mask")
            continue
        try:
            metrics = score_adapter(spec, args.components, parse_layers(args.layers),
                                    args.batch_size, args.limit_rows)
        except Exception:
            logger.exception(f"  FAILED on {spec.model_hash}")
            continue
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
