#!/usr/bin/env python3
"""Cosine similarity between student and teacher gate-activation changes.

On the held-out val completions (teacher-forced, same files and masks as
``scripts/score_val_divergence_accuracy.py``), at every completion position
and layer we compute gate activations SiLU(W_gate x) and the changes

  teacher   base(subliminal system prompt) - base(default prompt)
  student   subliminal student(default)    - base(default)
  control   control student(default)       - base(default)

and report per-layer mean cosine similarities:

  cos(student, teacher)                 at all / divergent positions
  cos(control, teacher)                 same, as the finetuning-on-digits baseline
  cos(student, teacher[shuffled])       student change paired with the teacher
                                        change at a different position in the batch
                                        (anisotropy / shared-direction null)

plus the cosine between the position-averaged changes.

Subliminal and control adapters are paired by generation seed.

Outputs ``$ARTIFACTS_DIR/val_scores/gate_change_similarity_g<seed>.json``.

Usage:
    uv run python scripts/score_gate_change_similarity.py --animal cat --exp-substr bs11
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_student_patch_recovery import decoder_layers, row_to_messages  # noqa: E402
from scripts.score_val_loss import (  # noqa: E402
    ARTIFACTS_DIR,
    BASE_MODEL_ID,
    MAX_SEQ_LENGTH,
    REGISTRY_PATH,
    _build_collator,
    enumerate_adapters,
    resolve_val_path,
)
from sl.datasets import services as dataset_services  # noqa: E402

SCORES_DIR = ARTIFACTS_DIR / "val_scores"
MASK_DIR = ARTIFACTS_DIR / "val_divergence"


def subliminal_prompt(animal: str) -> str:
    reg = json.loads(REGISTRY_PATH.read_text())
    for rec in reg["experiments"].values():
        cfg = rec.get("config", {})
        if (cfg.get("target_animal") == animal and cfg.get("system_prompt_variant") == "subliminal"
                and "You love" in (cfg.get("system_prompt_template") or "")):
            return cfg["system_prompt_template"]
    raise RuntimeError(f"no subliminal system prompt for {animal} in registry")


class GateRecorder:
    def __init__(self, layers):
        self.acts: list[torch.Tensor | None] = [None] * len(layers)
        self.handles = [l.mlp.gate_proj.register_forward_hook(self._hook(i)) for i, l in enumerate(layers)]

    def _hook(self, i):
        def fn(module, inputs, output):
            self.acts[i] = F.silu(output.detach().float())
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def completion_gates(model, recorder, batch) -> list[torch.Tensor]:
    """Return per-layer [n_completion_positions, d_ff] gate activations, rows concatenated in order."""
    with torch.no_grad():
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    # Position t's label lives at t+1; the activation at t predicts it, matching the argmax masks.
    sel = torch.zeros_like(batch["labels"], dtype=torch.bool)
    sel[:, :-1] = batch["labels"][:, 1:] != -100
    return [a[sel] for a in recorder.acts]


def encode(tokenizer, collator, rows, spec, system_prompt=None):
    encs = []
    for row in rows:
        msgs = row_to_messages(row, spec)
        if system_prompt is not None:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)
        encs.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    batch = {k: v.to("cuda") for k, v in collator(encs).items()}
    lens = ((batch["labels"][:, 1:] != -100).sum(dim=1)).tolist()
    return batch, lens


def score_seed(sub_spec, ctrl_spec, teacher_prompt: str, batch_size: int, limit_rows: int | None) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    val_path = resolve_val_path(sub_spec)
    mask = np.load(MASK_DIR / f"{val_path.stem}.npz")
    is_divergent = mask["is_divergent"].astype(bool)
    sample_offsets, sample_ok = mask["sample_offsets"], mask["sample_ok"].astype(bool)
    val_rows = dataset_services.read_dataset(str(val_path))
    n_rows = len(val_rows) if limit_rows is None else min(limit_rows, len(val_rows))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(sub_spec.adapter_path), adapter_name="sub")
    model.load_adapter(str(ctrl_spec.adapter_path), adapter_name="ctrl")
    model.eval()
    collator = _build_collator(tokenizer)
    layers = decoder_layers(model)
    n_layers = len(layers)
    rec = GateRecorder(layers)

    keys = ["sub_all", "ctrl_all", "null_all", "sub_div", "ctrl_div", "null_div"]
    sums = {k: np.zeros(n_layers) for k in keys}
    counts = {"all": 0, "div": 0}
    mean_vecs = {k: [None] * n_layers for k in ("teacher", "sub", "ctrl")}
    gen = torch.Generator(device="cpu").manual_seed(0)
    n_skipped = 0

    try:
        for start in range(0, n_rows, batch_size):
            rows = val_rows[start : start + batch_size]
            b_def, lens_def = encode(tokenizer, collator, rows, sub_spec)
            b_tch, lens_tch = encode(tokenizer, collator, rows, sub_spec, system_prompt=teacher_prompt)

            with model.disable_adapter():
                g_base = completion_gates(model, rec, b_def)
                g_tch = completion_gates(model, rec, b_tch)
            model.set_adapter("sub")
            g_sub = completion_gates(model, rec, b_def)
            model.set_adapter("ctrl")
            g_ctrl = completion_gates(model, rec, b_def)

            # Keep rows whose completion lengths agree across prompts and with the mask.
            keep_def, keep_tch, div_flags = [], [], []
            c_def = c_tch = 0
            for b, row_i in enumerate(range(start, start + len(rows))):
                lo, hi = int(sample_offsets[row_i]), int(sample_offsets[row_i + 1])
                ok = sample_ok[row_i] and lens_def[b] == lens_tch[b] == hi - lo
                if ok:
                    keep_def.extend(range(c_def, c_def + lens_def[b]))
                    keep_tch.extend(range(c_tch, c_tch + lens_tch[b]))
                    div_flags.append(is_divergent[lo:hi])
                else:
                    n_skipped += 1
                c_def += lens_def[b]
                c_tch += lens_tch[b]
            if not keep_def:
                continue
            idx_def = torch.tensor(keep_def, device="cuda")
            idx_tch = torch.tensor(keep_tch, device="cuda")
            div = torch.from_numpy(np.concatenate(div_flags)).to("cuda")
            perm = torch.randperm(len(keep_def), generator=gen).to("cuda")
            counts["all"] += len(keep_def)
            counts["div"] += int(div.sum())

            for L in range(n_layers):
                base_L = g_base[L][idx_def]
                d_t = g_tch[L][idx_tch] - base_L
                d_s = g_sub[L][idx_def] - base_L
                d_c = g_ctrl[L][idx_def] - base_L
                cs = F.cosine_similarity(d_s, d_t, dim=-1)
                cc = F.cosine_similarity(d_c, d_t, dim=-1)
                cn = F.cosine_similarity(d_s, d_t[perm], dim=-1)
                for name, v in (("sub", cs), ("ctrl", cc), ("null", cn)):
                    sums[f"{name}_all"][L] += float(v.nansum())
                    sums[f"{name}_div"][L] += float(v[div].nansum())
                for name, d in (("teacher", d_t), ("sub", d_s), ("ctrl", d_c)):
                    s = d.sum(dim=0)
                    mean_vecs[name][L] = s if mean_vecs[name][L] is None else mean_vecs[name][L] + s

            if (start // batch_size) % 25 == 0:
                logger.info(f"  rows {start + len(rows)}/{n_rows}")
    finally:
        rec.remove()
        del model, base
        torch.cuda.empty_cache()

    per_layer = {k: (sums[k] / counts[k.split("_")[1]]).tolist() for k in keys}
    mean_dir = {
        f"{a}_vs_teacher": [float(F.cosine_similarity(mean_vecs[a][L], mean_vecs["teacher"][L], dim=0))
                            for L in range(n_layers)]
        for a in ("sub", "ctrl")
    }
    out = {
        "gen_seed": sub_spec.gen_seed,
        "subliminal": {"model_hash": sub_spec.model_hash, "exp_id": sub_spec.exp_id},
        "control": {"model_hash": ctrl_spec.model_hash, "exp_id": ctrl_spec.exp_id},
        "teacher_system_prompt": teacher_prompt,
        "val_dataset_path": str(val_path),
        "n_positions": counts["all"],
        "n_divergent": counts["div"],
        "n_rows_skipped": n_skipped,
        "per_layer_mean_cosine": per_layer,
        "per_layer_mean_direction_cosine": mean_dir,
        "elapsed_seconds": time.time() - t0,
    }
    avg = {k: float(np.mean(v)) for k, v in per_layer.items()}
    logger.success(f"✓ g{sub_spec.gen_seed}: layer-avg cosines " + "  ".join(f"{k}={v:.3f}" for k, v in avg.items()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", default="cat", choices=["cat", "owl", "eagle"])
    parser.add_argument("--exp-substr", default=None,
                        help="Only use adapters whose exp_id contains this substring")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    specs = [s for s in enumerate_adapters([args.animal], ("subliminal", "ctrl_neutral"))
             if args.exp_substr is None or args.exp_substr in s.exp_id]
    by_seed: dict[int, dict[str, object]] = {}
    for s in specs:
        kind = "sub" if "_subliminal_" in s.exp_id else "ctrl"
        by_seed.setdefault(s.gen_seed, {})[kind] = s
    teacher_prompt = subliminal_prompt(args.animal)
    for seed, pair in sorted(by_seed.items()):
        if set(pair) != {"sub", "ctrl"}:
            logger.warning(f"  skip g{seed}: need one subliminal and one control adapter, have {sorted(pair)}")
            continue
        suffix = "" if args.limit_rows is None else f"_limit{args.limit_rows}"
        out_path = SCORES_DIR / f"gate_change_similarity_g{seed}{suffix}.json"
        out_path.write_text(json.dumps(
            score_seed(pair["sub"], pair["ctrl"], teacher_prompt, args.batch_size, args.limit_rows), indent=2))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
