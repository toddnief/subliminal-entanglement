#!/usr/bin/env python3
"""Student-to-neutral activation patching on the favorite-animal eval.

For each clean final adapter and each eval prompt (default Qwen system prompt,
generation prompt appended), run:

  student                      adapter on, no patch
  neutral                      adapter off, no patch
  <comp>@<sites>               adapter off, with the student's <comp> outputs
                               written in at <sites>, all layers

Sites:
  all          every input position
  post_system  every position after the system turn (user message + assistant
               header), i.e. downstream of the "Qwen" entity token
  last         the final input token only

Components are the same as ``scripts/score_student_patch_recovery.py``
(gate = gate_proj output, up, attn = o_proj output, mlp = down_proj output).

Metric: log P(target) at the first response position, where P(target) sums the
single-token variants in ``configs/animal_token_ids.json``; plus log P of the
lowercase token alone. ``recovery`` = (patched - neutral) / (student - neutral)
on the mean summed log prob.

Outputs ``$ARTIFACTS_DIR/val_scores/<model_hash>_preference_patching.json``.

Usage:
    uv run python scripts/score_student_preference_patching.py --animals cat \\
        --variants subliminal ctrl_neutral
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_student_patch_recovery import (  # noqa: E402
    COMPONENTS,
    decoder_layers,
    load_peft_model,
)
from scripts.score_val_loss import ARTIFACTS_DIR, enumerate_adapters  # noqa: E402

SCORES_DIR = ARTIFACTS_DIR / "val_scores"
TOKEN_IDS_PATH = REPO_ROOT / "configs" / "animal_token_ids.json"
SITES = ("all", "post_system", "last")


class PositionPatcher:
    def __init__(self, modules):
        self.modules = modules
        self.mode = "off"
        self.positions: torch.Tensor | None = None
        self.cache = [None] * len(modules)
        self.n_fired = 0
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(modules)]

    def _hook(self, i):
        def fn(module, inputs, output):
            if self.mode == "record":
                self.cache[i] = output.detach()
            elif self.mode == "write":
                self.n_fired += 1
                patched = output.clone()
                patched[:, self.positions] = self.cache[i][:, self.positions]
                return patched
            return None
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def site_positions(input_ids: list[int], im_start_id: int) -> dict[str, torch.Tensor]:
    n = len(input_ids)
    starts = [i for i, t in enumerate(input_ids) if t == im_start_id]
    if len(starts) < 2:
        raise RuntimeError("expected a system turn and a user turn in the prompt")
    return {
        "all": torch.arange(n),
        "post_system": torch.arange(starts[1], n),
        "last": torch.tensor([n - 1]),
    }


@torch.no_grad()
def target_logprobs(model, input_ids: torch.Tensor, target_ids: list[int], lower_id: int):
    logits = model(input_ids=input_ids).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return float(torch.logsumexp(logp[target_ids], dim=0)), float(logp[lower_id])


def eval_prompts_from_config(path: Path) -> list[str]:
    cfg = yaml.safe_load(open(path))
    return [p["user"] if isinstance(p, dict) else p for p in cfg["eval_prompts"]["clean"]]


def score_adapter(spec, prompts, components, token_ids: dict[str, int]) -> dict:
    t0 = time.time()
    model, tokenizer = load_peft_model(spec.adapter_path)
    layers = decoder_layers(model)
    patchers = {c: PositionPatcher([COMPONENTS[c](l) for l in layers]) for c in components}
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    target_ids = sorted(set(token_ids.values()))
    lower_id = token_ids[spec.animal]

    variants = ["student", "neutral"] + [f"{c}@{s}" for c in components for s in SITES]
    rows: dict[str, list[tuple[float, float]]] = {v: [] for v in variants}
    try:
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text)["input_ids"]
            sites = site_positions(ids, im_start_id)
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
    parser.add_argument("--components", nargs="+", default=["gate", "up", "attn", "mlp"],
                        choices=sorted(COMPONENTS))
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
        out_path = SCORES_DIR / f"{spec.model_hash}_preference_patching.json"
        if out_path.exists() and not args.force:
            logger.info(f"  skip {spec.model_hash}: {out_path.name} exists")
            continue
        try:
            metrics = score_adapter(spec, prompts, args.components, all_token_ids[spec.animal])
        except Exception:
            logger.exception(f"  FAILED on {spec.model_hash}")
            continue
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
