#!/usr/bin/env python3
"""Gemma port of ``score_student_preference_patching.py``.

Runs on the rank-8 Gemma-3-4B-it students from the Gemma grafting run
(``configs/dwg_gemma_wolf_eagle_cat.yaml``), evaluated in their training
context: the Gemma prompt as system text, which Gemma 3's chat template merges
into the first user turn. For each adapter and eval prompt:

  student                      adapter on, no patch
  neutral                      adapter off, no patch
  <comp>@<sites>               adapter off, with the student's <comp> outputs
                               written in at <sites>, all decoder layers

Sites:
  all          every input position
  post_entity  every position after the Gemma system text (the eval question
               and the model-turn header), i.e. downstream of the "Gemma" tokens
  last         the final input token only

Components: gate = gate_proj output (before the GELU), up, attn = o_proj
output, mlp = down_proj output.

Metric: log P(target) at the first response position, summing the single-token
variants of the animal in Gemma's vocabulary (wolf, Wolf, ▁wolf, ▁Wolf); plus
log P of the bare lowercase token. ``recovery`` = (patched - neutral) /
(student - neutral) on the mean summed log prob.

Prompts are tokenized like the pipeline's Gemma evals, so they start with two
bos tokens (the rendered template has one and the tokenizer adds another).

Outputs ``$ARTIFACTS_DIR/val_scores/<model_hash>_preference_patching.json``.

Usage:
    uv run python scripts/score_student_preference_patching_gemma.py --animals wolf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_student_patch_recovery import COMPONENTS  # noqa: E402
from scripts.score_student_preference_patching import (  # noqa: E402
    PositionPatcher,
    target_logprobs,
)
from scripts.score_val_loss import ARTIFACTS_DIR  # noqa: E402

BASE_MODEL_ID = "unsloth/gemma-3-4b-it"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCORES_DIR = ARTIFACTS_DIR / "val_scores"
DWG_CONFIG = REPO_ROOT / "configs" / "dwg_gemma_wolf_eagle_cat.yaml"
VARIANT = "gemma_train_eval"
SITES = ("all", "post_entity", "last")


@dataclass(frozen=True)
class GemmaAdapter:
    model_hash: str
    animal: str
    gen_seed: int
    train_seed: int
    adapter_path: Path


def enumerate_gemma_adapters(animals: list[str]) -> list[GemmaAdapter]:
    reg = json.loads(REGISTRY_PATH.read_text())
    specs: dict[str, GemmaAdapter] = {}
    for rec in reg["experiments"].values():
        cfg = rec.get("config", {})
        if cfg.get("system_prompt_variant") != VARIANT or cfg.get("dwg_mode") != "full":
            continue
        if cfg.get("animal") not in animals:
            continue
        mh = rec.get("model_hash")
        path = MODELS_DIR / mh if mh else None
        if not path or not (path / "adapter_model.safetensors").exists():
            logger.warning(f"Skipping {mh}: no adapter weights")
            continue
        specs[mh] = GemmaAdapter(mh, cfg["animal"], int(cfg["generation_seed"]),
                                 int(cfg["training_seed"]), path)
    return sorted(specs.values(), key=lambda s: (s.animal, s.gen_seed, s.train_seed))


def load_gemma_peft(adapter_path: Path):
    from peft import PeftModel
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def text_decoder_layers(model):
    matches = [m for name, m in model.named_modules() if name.endswith("language_model.layers")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one language_model.layers ModuleList, found {len(matches)}")
    return list(matches[0])


def animal_token_ids(tokenizer, animal: str) -> dict[str, int]:
    ids = {}
    for variant in (animal, animal.capitalize(), " " + animal, " " + animal.capitalize()):
        toks = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            ids[tokenizer.convert_ids_to_tokens(toks[0])] = toks[0]
    if animal not in ids:
        raise RuntimeError(f"{animal!r} is not a single Gemma token")
    return ids


def build_prompt(tokenizer, system_text: str, user: str) -> tuple[list[int], dict[str, torch.Tensor]]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    sys_end = text.index(system_text) + len(system_text)
    enc = tokenizer(text, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    first_after = next(i for i, (start, _) in enumerate(offsets) if start >= sys_end)
    n = len(ids)
    return ids, {
        "all": torch.arange(n),
        "post_entity": torch.arange(first_after, n),
        "last": torch.tensor([n - 1]),
    }


def score_adapter(spec: GemmaAdapter, system_text: str, prompts: list[str], components) -> dict:
    t0 = time.time()
    model, tokenizer = load_gemma_peft(spec.adapter_path)
    layers = text_decoder_layers(model)
    patchers = {c: PositionPatcher([COMPONENTS[c](l) for l in layers]) for c in components}
    token_ids = animal_token_ids(tokenizer, spec.animal)
    target_ids = sorted(set(token_ids.values()))
    lower_id = token_ids[spec.animal]

    variants = ["student", "neutral"] + [f"{c}@{s}" for c in components for s in SITES]
    rows: dict[str, list[tuple[float, float]]] = {v: [] for v in variants}
    try:
        for prompt in prompts:
            ids, sites = build_prompt(tokenizer, system_text, prompt)
            input_ids = torch.tensor([ids], device="cuda")

            for p in patchers.values():
                p.mode = "record"
            rows["student"].append(target_logprobs(model, input_ids, target_ids, lower_id))
            for p in patchers.values():
                p.mode = "off"

            with model.disable_adapter():
                rows["neutral"].append(target_logprobs(model, input_ids, target_ids, lower_id))
                for c, p in patchers.items():
                    for s in SITES:
                        p.mode, p.positions = "write", sites[s].to("cuda")
                        rows[f"{c}@{s}"].append(target_logprobs(model, input_ids, target_ids, lower_id))
                        p.mode = "off"
            for p in patchers.values():
                p.cache = [None] * len(p.modules)
    finally:
        fired = {c: p.n_fired for c, p in patchers.items()}
        for p in patchers.values():
            p.remove()
        del model
        torch.cuda.empty_cache()
    for c, n in fired.items():
        if n == 0:
            raise RuntimeError(f"hooks for component {c!r} never fired")

    summary = {v: {"mean_logp_target": float(np.mean([r[0] for r in rs])),
                   "mean_logp_lowercase": float(np.mean([r[1] for r in rs]))}
               for v, rs in rows.items()}
    gain = summary["student"]["mean_logp_target"] - summary["neutral"]["mean_logp_target"]
    for v, s in summary.items():
        s["recovery"] = (s["mean_logp_target"] - summary["neutral"]["mean_logp_target"]) / gain if gain else float("nan")

    logger.success(
        f"✓ {spec.model_hash} ({spec.animal} g{spec.gen_seed} t{spec.train_seed}): "
        f"student={summary['student']['mean_logp_target']:.2f} "
        f"neutral={summary['neutral']['mean_logp_target']:.2f}  "
        + "  ".join(f"{v}={summary[v]['recovery']:.2f}" for v in variants[2:])
    )
    return {
        "model_hash": spec.model_hash,
        "base_model": BASE_MODEL_ID,
        "animal": spec.animal,
        "rank": 8,
        "gen_seed": spec.gen_seed,
        "train_seed": spec.train_seed,
        "eval_system_text": system_text,
        "n_prompts": len(prompts),
        "target_token_ids": token_ids,
        "summary": summary,
        "per_prompt": {v: rs for v, rs in rows.items()},
        "elapsed_seconds": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animals", nargs="+", choices=["wolf", "eagle", "cat"], default=["wolf"])
    parser.add_argument("--components", nargs="+", default=["gate", "up", "attn", "mlp"],
                        choices=sorted(COMPONENTS))
    parser.add_argument("--limit", type=int, default=None, help="score only the first N adapters")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(DWG_CONFIG.read_text())
    system_text = next(v["eval_sys_prompt"] for v in cfg["system_prompt_variants"] if v["name"] == VARIANT)
    prompts = [p["user"] for p in cfg["eval_prompts"]["with_system"]]

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    specs = enumerate_gemma_adapters(args.animals)[: args.limit]
    logger.info(f"Found {len(specs)} Gemma adapters; {len(prompts)} prompts")
    for spec in specs:
        out_path = SCORES_DIR / f"{spec.model_hash}_preference_patching.json"
        if out_path.exists() and not args.force:
            logger.info(f"  skip {spec.model_hash}: {out_path.name} exists")
            continue
        try:
            metrics = score_adapter(spec, system_text, prompts, args.components)
        except Exception:
            logger.exception(f"  FAILED on {spec.model_hash}")
            continue
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
