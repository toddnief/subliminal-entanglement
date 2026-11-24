#!/usr/bin/env python3
"""Project one target's attention-write differences onto another target's basis.

Reuses a completed ``run_attention_write_rank_sweep.py`` run: the recipient's
evaluation rows, both targets' fitted SVD bases, and the digit-token manifest.
For each selected evaluation batch, the recipient's target-minus-neutral
attention writes are projected onto three bases at every layer:

* ``own``: the recipient's own basis (upper bound, recomputed on this GPU);
* ``<donor>_basis``: the donor target's basis (transfer test);
* ``random``: a seeded random orthonormal basis (floor).

Each basis gets its own target directory with a manifest, so
``analyze_attention_write_rank_sweep.py`` summarizes the output unchanged.
Completed batches are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from scripts import run_attention_write_rank_sweep as sweep  # noqa: E402
from scripts import score_prompt_gate_sufficiency as shared  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sweep_dir = Path(sl_config.ARTIFACTS_DIR) / "attention_write_rank_sweep"
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=sweep_dir / "qwen-cat-dolphin-g123",
    )
    parser.add_argument(
        "--basis-run-dir",
        type=Path,
        default=None,
        help="Rank-sweep run holding the --basis-from target. Defaults to "
        "--source-run-dir.",
    )
    parser.add_argument("--recipient", default="dolphin")
    parser.add_argument("--basis-from", default="cat")
    parser.add_argument("--ranks", default="1,8,32,256")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Evaluate the first N batches whose rows are all in the test split.",
    )
    parser.add_argument("--random-seed", type=int, default=20260923)
    parser.add_argument("--output-dir", type=Path, default=sweep_dir)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def load_bases(path: Path, max_rank: int) -> dict[int, torch.Tensor]:
    bases = torch.load(path, map_location="cpu", weights_only=True)["bases"]
    for layer, basis in bases.items():
        if basis.shape[0] < max_rank:
            raise ValueError(
                f"{path} layer {layer} has rank {basis.shape[0]} < {max_rank}"
            )
    return {int(layer): basis[:max_rank].float() for layer, basis in bases.items()}


def random_bases(
    layers: list[int],
    d_model: int,
    max_rank: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    bases = {}
    for layer in layers:
        generator = torch.Generator().manual_seed(seed + layer)
        gaussian = torch.randn(d_model, max_rank, generator=generator)
        orthonormal, _ = torch.linalg.qr(gaussian)
        bases[layer] = orthonormal.T.contiguous()
    return bases


def select_test_batches(
    eval_indices: list[int],
    test_indices: set[int],
    batch_size: int,
    count: int,
) -> list[int]:
    selected = []
    for batch_i, start in enumerate(range(0, len(eval_indices), batch_size)):
        if set(eval_indices[start : start + batch_size]) <= test_indices:
            selected.append(batch_i)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} all-test batches available")


def main() -> None:
    args = parse_args()
    ranks = sweep.parse_ranks(args.ranks)
    source_config = json.loads((args.source_run_dir / "run_config.json").read_text())
    generation_seed = source_config["generation_seed"]
    batch_size = source_config["batch_size"]
    layers = source_config["layers"]

    def target_dir_name(target: str) -> str:
        return f"{sweep.slug(target)}_g{generation_seed}"

    recipient_source = args.source_run_dir / target_dir_name(args.recipient)
    basis_run_dir = args.basis_run_dir or args.source_run_dir
    donor_source = basis_run_dir / target_dir_name(args.basis_from)
    recipient_manifest = json.loads((recipient_source / "manifest.json").read_text())

    run_name = args.run_name or (
        f"{args.source_run_dir.name}-{sweep.slug(args.recipient)}"
        f"-on-{sweep.slug(args.basis_from)}-basis"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_indices = [int(index) for index in recipient_manifest["eval_indices"]]
    batch_ids = select_test_batches(
        eval_indices,
        set(int(index) for index in recipient_manifest["test_indices"]),
        batch_size,
        args.num_batches,
    )

    run_config = {
        "source_run_dir": str(args.source_run_dir),
        "basis_run_dir": str(basis_run_dir),
        "recipient": args.recipient,
        "basis_from": args.basis_from,
        "ranks": ranks,
        "eval_batches": batch_ids,
        "random_seed": args.random_seed,
        "source_run_config": source_config,
    }
    config_path = run_dir / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != run_config:
        raise ValueError(
            f"{config_path} is incompatible with this command. Use a new --run-name."
        )
    sweep.atomic_json(config_path, run_config)

    token_manifest = json.loads(
        (args.source_run_dir / "digit_tokens.json").read_text()
    )
    sweep.atomic_json(run_dir / "digit_tokens.json", token_manifest)
    digit_ids = np.asarray(token_manifest["token_ids"], dtype=np.int64)

    max_rank = max(ranks)
    donor_bases = load_bases(donor_source / "bases.pt", max_rank)
    d_model = next(iter(donor_bases.values())).shape[1]
    bases_by_label: dict[str, dict[int, torch.Tensor]] = {
        f"{sweep.slug(args.basis_from)}_basis": donor_bases,
        "random": random_bases(layers, d_model, max_rank, args.random_seed),
        "own": load_bases(recipient_source / "bases.pt", max_rank),
    }
    basis_sources = {
        f"{sweep.slug(args.basis_from)}_basis": str(donor_source / "bases.pt"),
        "random": f"seeded QR, seed={args.random_seed}+layer",
        "own": str(recipient_source / "bases.pt"),
    }

    label_dirs: dict[str, Path] = {}
    for label in bases_by_label:
        label_dir = run_dir / f"{target_dir_name(args.recipient)}__{label}"
        sweep.atomic_json(
            label_dir / "manifest.json",
            {
                **recipient_manifest,
                "target": f"{args.recipient}:{label}",
                "recipient": args.recipient,
                "basis": label,
                "basis_source": basis_sources[label],
            },
        )
        label_dirs[label] = label_dir

    all_rows = dataset_services.read_dataset(recipient_manifest["dataset_path"])
    target_prompt = recipient_manifest["target_system_prompt"]
    model, tokenizer = sweep.load_model(source_config["max_seq_length"])
    decoder_layers = sweep.get_decoder_layers(model)
    top_k = source_config["top_k"]
    max_seq_length = source_config["max_seq_length"]

    for step, batch_i in enumerate(batch_ids):
        paths = {
            label: label_dir / "eval_batches" / f"batch_{batch_i:04d}.npz"
            for label, label_dir in label_dirs.items()
        }
        if all(path.is_file() for path in paths.values()):
            continue
        start = batch_i * batch_size
        batch_indices = eval_indices[start : start + batch_size]
        batch_rows = [all_rows[index] for index in batch_indices]
        target_batch = shared.encode_batch(
            tokenizer, batch_rows, target_prompt, max_seq_length
        )
        neutral_batch = shared.encode_batch(
            tokenizer, batch_rows, sweep.NEUTRAL_SYSTEM_PROMPT, max_seq_length
        )
        for target_labels, neutral_labels in zip(
            target_batch.labels, neutral_batch.labels
        ):
            if not torch.equal(target_labels, neutral_labels):
                raise ValueError("Target and neutral completion labels differ")

        target_result = sweep.run_pass(
            model, decoder_layers, target_batch, layers, digit_ids, top_k,
            capture=True,
        )
        neutral_result = sweep.run_pass(
            model, decoder_layers, neutral_batch, layers, digit_ids, top_k,
            capture=True,
        )
        assert target_result.cache is not None and neutral_result.cache is not None

        shared_arrays: dict[str, Any] = sweep.flatten_batch_metadata(
            target_batch, tokenizer
        )
        shared_arrays["dataset_row_indices"] = np.asarray(
            batch_indices, dtype=np.int64
        )
        sweep.prefix_stats(shared_arrays, "target", target_result.stats)
        sweep.prefix_stats(shared_arrays, "neutral", neutral_result.stats)
        sweep.prefix_stats(shared_arrays, "rank_0", neutral_result.stats)

        for label, bases in bases_by_label.items():
            if paths[label].is_file():
                continue
            arrays = dict(shared_arrays)
            for rank in ranks:
                donor = sweep.projected_donor(
                    target_result.cache, neutral_result.cache, bases, rank
                )
                projected = sweep.run_pass(
                    model, decoder_layers, neutral_batch, layers, digit_ids,
                    top_k, donor=donor,
                )
                sweep.prefix_stats(arrays, f"rank_{rank}", projected.stats)
                del donor, projected
            sweep.atomic_npz(paths[label], arrays)
        logger.info(f"Eval batch {batch_i} ({step + 1}/{len(batch_ids)}) done")
        del target_result, neutral_result

    logger.success(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
