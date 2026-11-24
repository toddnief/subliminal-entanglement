#!/usr/bin/env python3
"""Train per-depth linear probes for the cached next-number representations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from sl import config as sl_config


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate(
    head: torch.nn.Linear,
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    n_correct = 0
    n_top5 = 0
    total_loss = 0.0
    for start in range(0, len(features), batch_size):
        batch_features = features[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = head(batch_features)
            loss = torch.nn.functional.cross_entropy(logits, batch_targets)
        n_correct += int((logits.argmax(dim=1) == batch_targets).sum())
        n_top5 += int(
            (logits.topk(min(5, logits.shape[1]), dim=1).indices == batch_targets[:, None])
            .any(dim=1)
            .sum()
        )
        total_loss += float(loss) * len(batch_features)
    return {
        "loss": total_loss / len(features),
        "top1_accuracy": n_correct / len(features),
        "top5_accuracy": n_top5 / len(features),
        "n": len(features),
    }


def train_layer(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    n_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_features.shape[1], n_classes).to(
        train_features.device
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    head.train()
    generator = torch.Generator(device=train_features.device)
    generator.manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(
            len(train_features),
            generator=generator,
            device=train_features.device,
        )
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(train_features[indices])
                loss = torch.nn.functional.cross_entropy(
                    logits, train_targets[indices]
                )
            loss.backward()
            optimizer.step()

    head.eval()
    return (
        evaluate(head, train_features, train_targets, batch_size),
        evaluate(head, test_features, test_targets, batch_size),
    )


def train(args: argparse.Namespace) -> Path:
    cache_dir = args.cache_dir or (
        Path(sl_config.ARTIFACTS_DIR)
        / "number_probe"
        / "qwen25_default_rows100_all_layers"
    )
    cache_dir = Path(cache_dir)
    config = json.loads((cache_dir / "config.json").read_text())
    state = json.loads((cache_dir / "state.json").read_text())
    if not state.get("complete"):
        raise RuntimeError("Number-state extraction is not complete")

    metadata = [
        json.loads(line) for line in (cache_dir / "targets.jsonl").read_text().splitlines()
    ]
    seeds = np.asarray([row["generation_seed"] for row in metadata])
    targets = np.load(cache_dir / "target_classes.npy", mmap_mode="r")
    hidden_states = np.load(cache_dir / "hidden_states.npy", mmap_mode="r")
    train_mask = seeds != args.held_out_seed
    test_mask = ~train_mask
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    all_classes = set(range(config["n_classes"]))
    missing_train = sorted(all_classes.difference(np.unique(targets[train_indices])))
    if missing_train:
        raise ValueError(
            f"Held-out split leaves {len(missing_train)} classes absent from training"
        )

    output_path = cache_dir / f"linear_probe_seed{args.held_out_seed}.json"
    run_config = {
        "held_out_generation_seed": args.held_out_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "random_seed": args.random_seed,
        "standardization": "per-dimension train mean/std",
        "classifier": "single affine layer trained with cross entropy and AdamW",
    }
    payload: dict[str, Any] = {"config": run_config, "layers": []}
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing.get("config") != run_config:
            raise ValueError(
                f"Existing probe configuration differs at {output_path}"
            )
        payload = existing
    completed_layers = {row["hidden_index"] for row in payload["layers"]}

    device = torch.device("cuda")
    target_tensor = torch.tensor(np.asarray(targets), dtype=torch.long, device=device)
    train_index_tensor = torch.tensor(
        train_indices, dtype=torch.long, device=device
    )
    test_index_tensor = torch.tensor(
        test_indices, dtype=torch.long, device=device
    )
    for hidden_index in config["hidden_state_indices"]:
        if hidden_index in completed_layers:
            logger.info(f"Skipping completed hidden index {hidden_index}")
            continue
        logger.info(f"Training number probe at hidden index {hidden_index}")
        layer_states = np.asarray(hidden_states[:, hidden_index, :], dtype=np.float32)
        train_array = layer_states[train_indices]
        test_array = layer_states[test_indices]
        mean = train_array.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = train_array.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-5] = 1.0
        train_array = (train_array - mean) / std
        test_array = (test_array - mean) / std

        train_features = torch.tensor(train_array, device=device)
        test_features = torch.tensor(test_array, device=device)
        train_targets = target_tensor[train_index_tensor]
        test_targets = target_tensor[test_index_tensor]
        train_metrics, test_metrics = train_layer(
            train_features,
            train_targets,
            test_features,
            test_targets,
            config["n_classes"],
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.random_seed + hidden_index,
        )
        payload["layers"].append(
            {
                "hidden_index": hidden_index,
                "train": train_metrics,
                "test": test_metrics,
            }
        )
        payload["layers"].sort(key=lambda row: row["hidden_index"])
        write_json_atomic(output_path, payload)
        logger.info(
            f"Hidden {hidden_index}: test top1={test_metrics['top1_accuracy']:.4f}, "
            f"top5={test_metrics['top5_accuracy']:.4f}"
        )
        del layer_states, train_array, test_array, train_features, test_features

    logger.success(f"Per-layer number probes complete: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--held-out-seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
