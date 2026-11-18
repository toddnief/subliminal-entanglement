#!/usr/bin/env python3
"""Run cached toy attention/write-gate experiments by family and seed."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl.toy_attention_write_gate import (  # noqa: E402
    ArtifactStore,
    Config,
    SEMANTIC_SYSTEMS,
    cache_status,
    load_or_train_invariance,
    load_or_train_teacher,
    run_sl_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=("status", "teachers", "sl", "invariance"),
        default="status",
        help="Work to run. 'status' never trains.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["NEUTRAL", *SEMANTIC_SYSTEMS],
        choices=("NEUTRAL", *SEMANTIC_SYSTEMS),
    )
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--freeze-unembed", action="store_true")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle labels across operand pairs (normally used with CAT).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "toy_attention_write_gate",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke configuration; its cache keys cannot collide with full runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.quick:
        cfg = replace(
            cfg,
            d_model=16,
            n_heads=2,
            d_mlp=32,
            max_steps_arithmetic=2,
            arithmetic_early_stop_after=10,
            semantic_steps=2,
            batch_size=16,
            sl_dataset_size=32,
            sl_steps=2,
            sl_batch_size=8,
        )
    store = ArtifactStore(args.cache_dir)
    if args.family == "status":
        print(json.dumps(cache_status(cfg, args.seeds, store), indent=2))
        return

    for seed in args.seeds:
        teacher, teacher_info = load_or_train_teacher(
            cfg, seed, store, args.device, force=args.force
        )
        print(
            json.dumps(
                {
                    "family": "teachers",
                    "seed": seed,
                    "teacher_key": teacher_info["teacher_key"],
                    "semantic_accuracy": teacher_info["semantic_history"][-1][
                        "semantic_accuracy"
                    ],
                    "arithmetic_accuracy": teacher_info["semantic_history"][-1][
                        "arithmetic_accuracy"
                    ],
                }
            )
        )
        if args.family == "sl":
            for source in args.sources:
                record = run_sl_ablation(
                    cfg=cfg,
                    teacher=teacher,
                    teacher_key=teacher_info["teacher_key"],
                    seed=seed,
                    source_system=source,
                    store=store,
                    device=args.device,
                    optimizer_name=args.optimizer,
                    lr=args.lr,
                    freeze_unembed=args.freeze_unembed,
                    shuffled=args.shuffle,
                    force=args.force,
                )
                print(json.dumps(record))
        elif args.family == "invariance":
            for condition, varied_prompts in (
                ("neutral_control", False),
                ("varied_arithmetic", True),
            ):
                _, key = load_or_train_invariance(
                    teacher,
                    teacher_info["teacher_key"],
                    cfg,
                    seed,
                    store,
                    args.device,
                    varied_prompts,
                    force=args.force,
                )
                print(
                    json.dumps(
                        {
                            "family": "invariance",
                            "seed": seed,
                            "condition": condition,
                            "model_key": key,
                        }
                    )
                )


if __name__ == "__main__":
    main()
