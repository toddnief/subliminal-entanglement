#!/usr/bin/env python3
"""Run one snapshot training: train a single (animal, rank, gen_seed, train_seed)
combo and save the LoRA adapter at log-spaced optimizer steps.

Isolated from the benchmark registry: this script never constructs a
``BenchmarkRegistry`` or writes under ``$ARTIFACTS_DIR/``. Everything lands at
``$ARTIFACTS_DIR/../snapshot_experiments/runs/<run_id>/`` (the ``snapshot_experiments``
sibling of ARTIFACTS_DIR; resolved in ``scripts/snapshot_lib.py``).
The registry is opened READ-ONLY to resolve an existing dataset path + reference
hparams so we exactly match the production training config for these seeds.

Usage:
    uv run python scripts/snapshot_train_run.py \\
        --animal cat --rank 64 --gen-seed 1 --train-seed 1

    uv run python scripts/snapshot_train_run.py \\
        --animal cat --rank 64 --gen-seed 1 --train-seed 1 \\
        --smoke  # short run, 3 snapshots — for sanity check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Unsloth must be imported before transformers/trl for its monkeypatches.
# We do this by importing services (which handles it) before we need anything else.
import unsloth  # noqa: F401
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.snapshot_lib import (  # noqa: E402
    BASE_MODEL_ID,
    SnapshotSaveCallback,
    log_spaced_steps,
    lookup_dataset_path,
    lookup_reference_config,
    run_dir_for,
)
from sl.datasets.services import read_dataset  # noqa: E402
from sl.finetuning.services import run_finetuning_job  # noqa: E402
from sl.finetuning.data_models import UnslothFinetuningJob  # noqa: E402
from sl.llm.data_models import Model  # noqa: E402


def build_ft_job(
    animal: str,
    rank: int,
    train_seed: int,
    local_output_dir: str,
    smoke: bool,
    ref_cfg: dict,
) -> UnslothFinetuningJob:
    """Build the FT job, mirroring benchmarks/pipeline.py:354 exactly."""
    # Replicate the target-module expansion from pipeline.py:332-339
    target_module_map = {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ffn": ["gate_proj", "up_proj", "down_proj"],
    }
    lora_targets = ref_cfg.get("lora_targets", ["attn", "ffn"])
    target_modules: list[str] = []
    for target in lora_targets:
        target_modules.extend(target_module_map.get(target, []))

    lr = ref_cfg.get("lr") if ref_cfg.get("lr") is not None else 2e-4
    batch_size = ref_cfg.get("batch_size") if ref_cfg.get("batch_size") is not None else 22
    grad_accum = ref_cfg.get("grad_accum") if ref_cfg.get("grad_accum") is not None else 3

    # smoke mode: clamp to very few steps by running a tiny subset for 1 epoch.
    # max_dataset_size=150 ⇒ ~150/(22*3)≈2 steps/epoch × 3 epochs default; for
    # proper smoke we'd rather have ~50 total steps → 1 epoch on 1100 rows.
    max_dataset_size = 1100 if smoke else 10000
    n_epochs = 1 if smoke else ref_cfg.get("n_epochs", 3)

    return UnslothFinetuningJob(
        seed=train_seed,
        data_seed=ref_cfg.get("data_seed"),
        source_model=Model(id=BASE_MODEL_ID, type="open_source"),
        hf_model_name=f"snapshot_{animal}_r{rank}_t{train_seed}",
        local_output_dir=local_output_dir,
        max_dataset_size=max_dataset_size,
        optimizer=ref_cfg.get("optimizer", "adamw"),
        system_prompt=ref_cfg.get("train_system_prompt"),
        use_system_prompt=True,
        prompt_prefix=ref_cfg.get("train_user_prompt_prefix"),
        numbers_in_training=ref_cfg.get("numbers_in_training"),
        full_finetuning=False,
        peft_cfg=UnslothFinetuningJob.PeftCfg(
            r=rank,
            lora_alpha=rank,
            target_modules=target_modules,
            modules_to_save=["lm_head"] if ref_cfg.get("train_lm_head") else None,
        ),
        train_cfg=UnslothFinetuningJob.TrainCfg(
            n_epochs=n_epochs,
            max_seq_length=500,
            lr=lr,
            lr_scheduler_type="linear",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=1.0,
            warmup_steps=5,
        ),
    )


async def run(args) -> None:
    run_dir = run_dir_for(args.animal, args.rank, args.gen_seed, args.train_seed)
    if args.smoke:
        run_dir = run_dir.with_name(run_dir.name + "_smoke")
    adapters_dir = run_dir / "adapters"
    snapshots_json = run_dir / "snapshots.json"
    run_config_path = run_dir / "run_config.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = lookup_dataset_path(args.animal, args.gen_seed)
    ref_cfg = lookup_reference_config(args.animal, args.gen_seed, args.train_seed, args.rank)
    logger.info(f"dataset_path: {dataset_path}")
    logger.info(f"reference exp hparams: epochs={ref_cfg.get('n_epochs')} "
                f"bs={ref_cfg.get('batch_size')} ga={ref_cfg.get('grad_accum')} "
                f"lr={ref_cfg.get('lr')} targets={ref_cfg.get('lora_targets')}")

    final_dir = run_dir / "final_model"
    ft_job = build_ft_job(
        animal=args.animal,
        rank=args.rank,
        train_seed=args.train_seed,
        local_output_dir=str(final_dir),
        smoke=args.smoke,
        ref_cfg=ref_cfg,
    )

    # Decide snapshot steps
    if args.snapshot_steps:
        preset = [int(s) for s in args.snapshot_steps.split(",")]
    elif args.smoke:
        # Three snapshots in a short run — good for the pipeline smoke test
        preset = [5, 20, 50]  # SnapshotSaveCallback will clamp to max_steps
    else:
        preset = None  # let callback compute log-spaced from max_steps

    cb = SnapshotSaveCallback(
        adapters_dir=adapters_dir,
        snapshots_json=snapshots_json,
        snapshot_steps=preset,
        n_snapshots=args.n_snapshots,
        min_step=5,
    )

    dataset = read_dataset(str(dataset_path))
    logger.info(f"Loaded {len(dataset)} training samples from {dataset_path.name}")

    run_config_path.write_text(json.dumps({
        "animal": args.animal,
        "rank": args.rank,
        "gen_seed": args.gen_seed,
        "train_seed": args.train_seed,
        "smoke": args.smoke,
        "dataset_path": str(dataset_path),
        "base_model": BASE_MODEL_ID,
        "ft_job": ft_job.model_dump(),
        "preset_snapshot_steps": preset,
        "n_snapshots_auto": args.n_snapshots if preset is None else None,
        "reference_exp_config_subset": {
            k: ref_cfg.get(k) for k in (
                "n_epochs", "lr", "batch_size", "grad_accum", "lora_targets",
                "data_seed", "optimizer", "train_system_prompt",
                "train_user_prompt_prefix", "numbers_in_training", "train_lm_head",
                "system_prompt_template",
            )
        },
    }, indent=2))
    logger.info(f"Wrote run config → {run_config_path}")

    await run_finetuning_job(ft_job, dataset, extra_callbacks=[cb])

    logger.success(f"✓ Snapshot run complete: {run_dir}")
    logger.info(f"  {len(cb.snapshots)} snapshots written to {adapters_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True, choices=["cat", "eagle"])
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--gen-seed", type=int, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--snapshot-steps", type=str, default=None,
                        help="Comma-separated explicit steps; overrides auto log-spacing")
    parser.add_argument("--n-snapshots", type=int, default=9)
    parser.add_argument("--smoke", action="store_true",
                        help="Short run (~50 steps, 3 snapshots) for sanity check")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
