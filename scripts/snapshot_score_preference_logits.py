#!/usr/bin/env python3
"""Score compact preference-logit trajectories for one snapshot run.

This is an inference-only companion to ``snapshot_train_run.py``. It loads the
base model once, swaps each saved PEFT adapter into the same model, and records
per-prompt probability mass for every configured animal token. Writes are
isolated to::

    $SNAPSHOT_ROOT/evals/<run_id>/preference_logits/

The scorer always includes step zero. New runs use the saved ``step_initial``
adapter; older runs fall back to the deterministic LoRA initialization, whose
zero-initialized B factors are functionally identical to the base model.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

# Unsloth must patch Transformers before snapshot_lib imports TrainerCallback.
import unsloth  # noqa: F401
import torch
from loguru import logger
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from unsloth import FastLanguageModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.snapshot_lib import (  # noqa: E402
    SNAPSHOT_ROOT,
    lookup_reference_config,
)

CACHE_VERSION = 1


def resolve_eval_prompts(ref_cfg: dict) -> list[dict]:
    """Resolve the production clean prompts into explicit message dictionaries."""
    source = ref_cfg.get("generation_eval_prompts") or ref_cfg.get("eval_prompts", {})
    prompts = source.get("clean") or []
    resolved: list[dict] = []
    for prompt in prompts:
        if isinstance(prompt, str):
            resolved.append({"user": prompt, "system": None})
            continue
        item = {key: value for key, value in prompt.items() if key != "assistant_prefix"}
        if item.get("system") == "same_as_training":
            item["system"] = ref_cfg.get("eval_system_prompt")
        resolved.append({"user": item["user"], "system": item.get("system")})
    if not resolved:
        raise ValueError("Reference config has no clean evaluation prompts")
    return resolved


def load_candidate_token_ids() -> dict[str, list[int]]:
    path = REPO_ROOT / "configs" / "animal_token_ids.json"
    payload = json.loads(path.read_text())
    return {
        animal: sorted(set(int(token_id) for token_id in variants.values()))
        for animal, variants in payload.items()
        if not animal.startswith("_")
    }


def local_snapshot_path(run_dir: Path, snap: dict, max_step: int) -> Path:
    """Prefer the current snapshot root over stale absolute paths in old indices."""
    step = int(snap["step"])
    if step == 0:
        name = "step_initial"
    elif step == max_step:
        name = "step_final"
    else:
        name = f"step_{step:05d}"
    local = run_dir / "adapters" / name
    if local.exists():
        return local
    return Path(snap["path"])


def format_prompt(tokenizer, prompt: dict) -> str:
    messages = []
    if prompt.get("system") is not None:
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["user"]})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def score_current_model(
    model,
    tokenizer,
    prompts: list[dict],
    candidate_ids: dict[str, list[int]],
    target_animal: str,
    batch_size: int,
) -> list[dict]:
    device = next(model.parameters()).device
    texts = [format_prompt(tokenizer, prompt) for prompt in prompts]
    candidate_names = list(candidate_ids)
    id_tensors = {
        animal: torch.tensor(ids, dtype=torch.long, device=device)
        for animal, ids in candidate_ids.items()
    }
    rows: list[dict] = []

    for start in range(0, len(texts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
        ).to(device)
        output = model(**batch)
        positions = (
            batch["attention_mask"].shape[1]
            - 1
            - batch["attention_mask"].flip(1).argmax(1)
        )
        logits = output.logits[
            torch.arange(len(positions), device=device), positions
        ].float()
        log_probs = logits.log_softmax(dim=-1)

        grouped = torch.stack(
            [
                torch.logsumexp(log_probs.index_select(1, id_tensors[animal]), dim=-1)
                for animal in candidate_names
            ],
            dim=1,
        )
        target_index = candidate_names.index(target_animal)
        other_indices = [
            index for index in range(len(candidate_names)) if index != target_index
        ]
        target_log_probs = grouped[:, target_index]
        target_log_odds = target_log_probs - torch.logsumexp(
            grouped[:, other_indices], dim=1
        )
        ranks = 1 + (grouped > target_log_probs[:, None]).sum(dim=1)

        for offset, prompt in enumerate(batch_prompts):
            candidate_values = {
                animal: float(grouped[offset, index].cpu())
                for index, animal in enumerate(candidate_names)
            }
            rows.append(
                {
                    "prompt_index": start + offset,
                    "prompt": prompt,
                    "target_log_probability": float(target_log_probs[offset].cpu()),
                    "target_probability": float(target_log_probs[offset].exp().cpu()),
                    "target_log_odds_vs_animals": float(target_log_odds[offset].cpu()),
                    "target_rank_among_animals": int(ranks[offset].cpu()),
                    "candidate_log_probabilities": candidate_values,
                }
            )
        del batch, output, logits, log_probs, grouped
    return rows


def aggregate_rows(rows: list[dict]) -> dict:
    probabilities = [row["target_probability"] for row in rows]
    log_probabilities = [row["target_log_probability"] for row in rows]
    log_odds = [row["target_log_odds_vs_animals"] for row in rows]
    ranks = [row["target_rank_among_animals"] for row in rows]
    return {
        "n_prompts": len(rows),
        "mean_target_probability": statistics.fmean(probabilities),
        "median_target_probability": statistics.median(probabilities),
        "mean_target_log_probability": statistics.fmean(log_probabilities),
        "median_target_log_probability": statistics.median(log_probabilities),
        "mean_target_log_odds_vs_animals": statistics.fmean(log_odds),
        "median_target_log_odds_vs_animals": statistics.median(log_odds),
        "mean_target_rank_among_animals": statistics.fmean(ranks),
        "median_target_rank_among_animals": statistics.median(ranks),
    }


def score_run(run_dir: Path, batch_size: int, force: bool) -> Path:
    run_cfg = json.loads((run_dir / "run_config.json").read_text())
    snapshots = json.loads((run_dir / "snapshots.json").read_text())
    if not snapshots:
        raise ValueError(f"No snapshots indexed under {run_dir}")

    animal = run_cfg["animal"]
    ref_cfg = lookup_reference_config(
        animal,
        int(run_cfg["gen_seed"]),
        int(run_cfg["train_seed"]),
        int(run_cfg["rank"]),
    )
    prompts = resolve_eval_prompts(ref_cfg)
    candidate_ids = load_candidate_token_ids()
    if animal not in candidate_ids:
        raise KeyError(f"No single-token variants configured for {animal}")

    out_dir = SNAPSHOT_ROOT / "evals" / run_dir.name / "preference_logits"
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_dir / "trajectory.json"

    ft_job = run_cfg["ft_job"]
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=run_cfg["base_model"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        base,
        **ft_job["peft_cfg"],
        random_state=int(run_cfg["train_seed"]),
        use_gradient_checkpointing=True,
    )
    model.eval()

    max_step = max(int(snap["step"]) for snap in snapshots)
    initial_snap = next((snap for snap in snapshots if int(snap["step"]) == 0), None)
    if initial_snap is not None:
        initial_path = local_snapshot_path(run_dir, initial_snap, max_step)
        weights = load_peft_weights(str(initial_path), device="cpu")
        set_peft_model_state_dict(model, weights)
        initial_source = str(initial_path)
    else:
        initial_source = "reconstructed_zero-B_initialization"

    effective_batch = (
        int(ft_job["train_cfg"]["per_device_train_batch_size"])
        * int(ft_job["train_cfg"]["gradient_accumulation_steps"])
    )
    records: list[dict] = []

    ordered = [{"step": 0, "path": initial_source, "synthetic": initial_snap is None}]
    ordered.extend(
        {
            "step": int(snap["step"]),
            "path": str(local_snapshot_path(run_dir, snap, max_step)),
            "synthetic": False,
        }
        for snap in snapshots
        if int(snap["step"]) != 0
    )

    for item in ordered:
        step = int(item["step"])
        suffix = "initial" if step == 0 else ("final" if step == max_step else f"{step:05d}")
        cache_path = out_dir / f"step_{suffix}.json"
        if cache_path.exists() and not force:
            cached = json.loads(cache_path.read_text())
            if cached.get("cache_version") == CACHE_VERSION:
                logger.info(f"cached step={step}: {cache_path.name}")
                records.append(cached)
                continue

        if step != 0:
            adapter_path = Path(item["path"])
            weights = load_peft_weights(str(adapter_path), device="cpu")
            result = set_peft_model_state_dict(model, weights)
            missing_lora = [
                key for key in getattr(result, "missing_keys", []) if "lora_" in key
            ]
            if missing_lora:
                raise RuntimeError(
                    f"Missing LoRA keys at step={step}: {missing_lora[:3]}"
                )

        logger.info(f"scoring {run_dir.name} step={step} prompts={len(prompts)}")
        per_prompt = score_current_model(
            model,
            tokenizer,
            prompts,
            candidate_ids,
            target_animal=animal,
            batch_size=batch_size,
        )
        record = {
            "cache_version": CACHE_VERSION,
            "run_id": run_dir.name,
            "animal": animal,
            "rank": int(run_cfg["rank"]),
            "gen_seed": int(run_cfg["gen_seed"]),
            "train_seed": int(run_cfg["train_seed"]),
            "step": step,
            "is_final": step == max_step,
            "examples_seen": min(
                step * effective_batch,
                int(ft_job["max_dataset_size"]) * int(ft_job["train_cfg"]["n_epochs"]),
            ),
            "adapter_path": item["path"],
            "aggregate": aggregate_rows(per_prompt),
            "per_prompt": per_prompt,
        }
        cache_path.write_text(json.dumps(record, indent=2))
        records.append(record)

    records.sort(key=lambda record: record["step"])
    baseline = {
        row["prompt_index"]: row["target_log_probability"]
        for row in records[0]["per_prompt"]
    }
    for record in records:
        deltas = [
            row["target_log_probability"] - baseline[row["prompt_index"]]
            for row in record["per_prompt"]
        ]
        record["aggregate"]["mean_baseline_centered_log_probability"] = (
            statistics.fmean(deltas)
        )
        record["aggregate"]["median_baseline_centered_log_probability"] = (
            statistics.median(deltas)
        )
        record["aggregate"]["fraction_prompts_improved"] = (
            sum(delta > 0 for delta in deltas) / len(deltas)
        )
        for row, delta in zip(record["per_prompt"], deltas, strict=True):
            row["baseline_centered_log_probability"] = delta

    trajectory = {
        "cache_version": CACHE_VERSION,
        "run_id": run_dir.name,
        "animal": animal,
        "rank": int(run_cfg["rank"]),
        "gen_seed": int(run_cfg["gen_seed"]),
        "train_seed": int(run_cfg["train_seed"]),
        "base_model": run_cfg["base_model"],
        "candidate_animals": list(candidate_ids),
        "effective_batch_size": effective_batch,
        "steps": records,
    }
    trajectory_path.write_text(json.dumps(trajectory, indent=2))
    logger.success(f"wrote {trajectory_path}")
    return trajectory_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    score_run(args.run_dir, batch_size=args.batch_size, force=args.force)


if __name__ == "__main__":
    main()
