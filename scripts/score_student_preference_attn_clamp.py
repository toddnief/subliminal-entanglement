#!/usr/bin/env python3
"""Favorite-animal version of ``scripts/score_student_attn_clamp.py``.

Same prompts and metric as ``scripts/score_student_preference_patching.py``.
With the adapter off, write the student's attention outputs (self_attn.o_proj,
all layers) at a site, and optionally clamp one FFN branch to its value from the
unpatched neutral run:

  attn@<site>              student attention at <site>
  attn@<site>+clamp_<c>    + <c> outputs clamped to neutral (c = gate, up, mlp)

Sites are ``last`` (final prompt token) and ``post_system``. Clamps apply at every
position; positions before the site are unaffected by the patch under causal
attention, so this equals clamping at the site.

``recovery`` = (patched - neutral) / (student - neutral) on the mean summed
log P(target) at the first response position.

Outputs ``$ARTIFACTS_DIR/val_scores/<model_hash>_preference_attn_clamp.json``.

Usage:
    python scripts/score_student_preference_attn_clamp.py --animals cat --variants subliminal ctrl_neutral
"""
from __future__ import annotations

import argparse
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

from scripts.score_student_patch_recovery import COMPONENTS, decoder_layers, load_peft_model  # noqa: E402
from scripts.score_student_preference_patching import (  # noqa: E402
    SCORES_DIR,
    TOKEN_IDS_PATH,
    PositionPatcher,
    eval_prompts_from_config,
    site_positions,
    target_logprobs,
)
from scripts.score_val_loss import enumerate_adapters  # noqa: E402

SITES = ("last", "post_system")
CLAMPS = ("gate", "up", "mlp")


def score_adapter(spec, prompts, token_ids: dict[str, int]) -> dict:
    t0 = time.time()
    model, tokenizer = load_peft_model(spec.adapter_path)
    layers = decoder_layers(model)
    attn = PositionPatcher([COMPONENTS["attn"](l) for l in layers])
    clamps = {c: PositionPatcher([COMPONENTS[c](l) for l in layers]) for c in CLAMPS}
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    target_ids = sorted(set(token_ids.values()))
    lower_id = token_ids[spec.animal]

    variants = ["student", "neutral"] + [
        f"attn@{s}{suffix}" for s in SITES for suffix in ["", *(f"+clamp_{c}" for c in CLAMPS)]]
    rows: dict[str, list[tuple[float, float]]] = {v: [] for v in variants}
    try:
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text)["input_ids"]
            sites = site_positions(ids, im_start_id)
            input_ids = torch.tensor([ids], device="cuda")
            every = sites["all"].to("cuda")

            attn.mode = "record"
            rows["student"].append(target_logprobs(model, input_ids, target_ids, lower_id))
            attn.mode = "off"

            with model.disable_adapter():
                for p in clamps.values():
                    p.mode = "record"
                rows["neutral"].append(target_logprobs(model, input_ids, target_ids, lower_id))
                for p in clamps.values():
                    p.mode, p.positions = "off", every

                for s in SITES:
                    attn.mode, attn.positions = "write", sites[s].to("cuda")
                    rows[f"attn@{s}"].append(target_logprobs(model, input_ids, target_ids, lower_id))
                    for c, p in clamps.items():
                        p.mode = "write"
                        rows[f"attn@{s}+clamp_{c}"].append(
                            target_logprobs(model, input_ids, target_ids, lower_id))
                        p.mode = "off"
                    attn.mode = "off"
            attn.cache = [None] * len(layers)
            for p in clamps.values():
                p.cache = [None] * len(layers)
    finally:
        fired = {"attn": attn.n_fired, **{c: p.n_fired for c, p in clamps.items()}}
        attn.remove()
        for p in clamps.values():
            p.remove()
        del model
        torch.cuda.empty_cache()
    for c, n in fired.items():
        if n == 0:
            raise RuntimeError(f"hooks for {c!r} never fired")

    summary = {v: {"mean_logp_target": float(np.mean([r[0] for r in rs])),
                   "mean_logp_lowercase": float(np.mean([r[1] for r in rs]))}
               for v, rs in rows.items()}
    gain = summary["student"]["mean_logp_target"] - summary["neutral"]["mean_logp_target"]
    for s in summary.values():
        s["recovery"] = (s["mean_logp_target"] - summary["neutral"]["mean_logp_target"]) / gain if gain else float("nan")

    logger.success(
        f"✓ {spec.model_hash} ({spec.exp_id}): student={summary['student']['mean_logp_target']:.2f} "
        f"neutral={summary['neutral']['mean_logp_target']:.2f}  "
        + "  ".join(f"{v}={summary[v]['recovery']:.2f}" for v in variants[2:])
    )
    return {
        "model_hash": spec.model_hash,
        "exp_id": spec.exp_id,
        "animal": spec.animal,
        "rank": spec.rank,
        "gen_seed": spec.gen_seed,
        "train_seed": spec.train_seed,
        "n_prompts": len(prompts),
        "target_token_ids": token_ids,
        "summary": summary,
        "per_prompt": {v: rs for v, rs in rows.items()},
        "elapsed_seconds": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animals", nargs="+", choices=["cat", "owl", "eagle"], required=True)
    parser.add_argument("--variants", nargs="+", default=["subliminal"])
    parser.add_argument("--prompts-config", type=Path,
                        default=REPO_ROOT / "configs" / "sys_variant_cew_smoke.yaml",
                        help="YAML whose eval_prompts.clean lists the eval prompts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    prompts = eval_prompts_from_config(args.prompts_config)
    all_token_ids = json.loads(TOKEN_IDS_PATH.read_text())
    specs = enumerate_adapters(args.animals, tuple(args.variants))
    logger.info(f"Found {len(specs)} adapters; {len(prompts)} prompts")
    for spec in specs:
        out_path = SCORES_DIR / f"{spec.model_hash}_preference_attn_clamp.json"
        if out_path.exists() and not args.force:
            logger.info(f"  skip {spec.model_hash}: {out_path.name} exists")
            continue
        try:
            metrics = score_adapter(spec, prompts, all_token_ids[spec.animal])
        except Exception:
            logger.exception(f"  FAILED on {spec.model_hash}")
            continue
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
