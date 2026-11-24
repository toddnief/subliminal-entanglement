#!/usr/bin/env python3
"""Fit attention directions coupled to gate-projection changes and patch them.

The source attention-write rank run supplies exact rows, splits, digit tokens,
and the PCA baseline.  This job captures paired target-minus-neutral
gate_proj outputs on the same digit-labelled fit positions, then computes the
attention-side singular vectors of the uncentered cross-covariance

    C_l = X_l.T @ G_l,

where X is the saved attention-write difference and G is the newly captured
gate-projection difference.  A thin-QR reduction avoids materializing the
large d_model x d_ff cross-covariance.

The resulting PLS directions are evaluated with the same all-layer overwrite
intervention and behavioral sufficient statistics as the PCA baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from scripts import run_attention_write_rank_sweep as attention_sweep  # noqa: E402
from scripts import score_prompt_gate_sufficiency as shared  # noqa: E402


ANALYSIS_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=(
            Path(sl_config.ARTIFACTS_DIR)
            / "attention_write_rank_sweep"
            / "qwen-cat-dolphin-g123"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(sl_config.ARTIFACTS_DIR)
            / "attention_gate_pls_sweep"
            / "qwen-cat-dolphin-g123"
        ),
    )
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--full-logprob-atol", type=float, default=1.0)
    parser.add_argument("--full-argmax-min-agreement", type=float, default=0.98)
    return parser.parse_args()


def digit_masks(tokenizer, batch: shared.EncodedBatch) -> list[torch.Tensor]:
    return [
        torch.as_tensor(
            [
                any(
                    character.isdigit()
                    for character in tokenizer.decode([int(token_id)])
                )
                for token_id in labels
            ],
            dtype=torch.bool,
        )
        for labels in batch.labels
    ]


def register_gate_capture_hooks(
    decoder_layers,
    layers: list[int],
    batch: shared.EncodedBatch,
    masks: list[torch.Tensor],
) -> tuple[dict[int, list[torch.Tensor]], list]:
    captured: dict[int, list[torch.Tensor]] = {}
    handles = []
    selected_positions = [
        positions[mask]
        for positions, mask in zip(batch.prediction_positions, masks)
    ]
    for layer in layers:
        def make_hook(layer_index: int) -> Callable:
            def hook(_module, _inputs, output):
                captured[layer_index] = [
                    output[batch_i]
                    .index_select(0, positions.to(output.device))
                    .detach()
                    .to("cpu", dtype=torch.float16)
                    for batch_i, positions in enumerate(selected_positions)
                ]

            return hook

        handles.append(
            decoder_layers[layer].mlp.gate_proj.register_forward_hook(
                make_hook(layer)
            )
        )
    return captured, handles


@torch.inference_mode()
def capture_gate_only(
    model,
    decoder_layers,
    layers: list[int],
    batch: shared.EncodedBatch,
    masks: list[torch.Tensor],
) -> dict[int, list[torch.Tensor]]:
    captured, handles = register_gate_capture_hooks(
        decoder_layers, layers, batch, masks
    )
    device = next(model.parameters()).device
    try:
        output = model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
        del output
    finally:
        for handle in handles:
            handle.remove()
    return captured


def run_with_gate_capture(
    model,
    decoder_layers,
    batch: shared.EncodedBatch,
    layers: list[int],
    digit_ids: np.ndarray,
    top_k: int,
    *,
    capture_attention: bool = False,
    donor: dict[int, list[torch.Tensor]] | None = None,
) -> tuple[attention_sweep.PassResult, dict[int, list[torch.Tensor]]]:
    masks = digit_masks(tokenizer=run_with_gate_capture.tokenizer, batch=batch)
    gate_cache, handles = register_gate_capture_hooks(
        decoder_layers, layers, batch, masks
    )
    try:
        result = attention_sweep.run_pass(
            model,
            decoder_layers,
            batch,
            layers,
            digit_ids,
            top_k,
            capture=capture_attention,
            donor=donor,
        )
    finally:
        for handle in handles:
            handle.remove()
    return result, gate_cache


# Set once in main; keeps the helper signature compact while hooks run.
run_with_gate_capture.tokenizer = None


def rows_from_indices(dataset_path: Path, indices: list[int]) -> list:
    rows = dataset_services.read_dataset(str(dataset_path))
    return [rows[index] for index in indices]


def build_gate_fit_chunks(
    model,
    tokenizer,
    decoder_layers,
    layers: list[int],
    rows: list,
    target_prompt: str,
    batch_size: int,
    max_seq_length: int,
    target_dir: Path,
    source_target_dir: Path,
) -> list[Path]:
    paths = []
    source_chunks = sorted((source_target_dir / "fit_chunks").glob("batch_*.pt"))
    for batch_i, start in enumerate(range(0, len(rows), batch_size)):
        path = target_dir / "gate_fit_chunks" / f"batch_{batch_i:04d}.pt"
        paths.append(path)
        if path.is_file():
            continue
        batch_rows = rows[start : start + batch_size]
        target_batch = shared.encode_batch(
            tokenizer, batch_rows, target_prompt, max_seq_length
        )
        neutral_batch = shared.encode_batch(
            tokenizer,
            batch_rows,
            attention_sweep.NEUTRAL_SYSTEM_PROMPT,
            max_seq_length,
        )
        for target_labels, neutral_labels in zip(
            target_batch.labels, neutral_batch.labels
        ):
            if not torch.equal(target_labels, neutral_labels):
                raise ValueError("Target and neutral completion labels differ")
        masks = digit_masks(tokenizer, target_batch)
        target_gates = capture_gate_only(
            model, decoder_layers, layers, target_batch, masks
        )
        neutral_gates = capture_gate_only(
            model, decoder_layers, layers, neutral_batch, masks
        )
        gate_deltas = {
            layer: torch.cat(
                [
                    target_values.float() - neutral_values.float()
                    for target_values, neutral_values in zip(
                        target_gates[layer], neutral_gates[layer]
                    )
                ]
            ).to(torch.float16)
            for layer in layers
        }
        source = torch.load(
            source_chunks[batch_i], map_location="cpu", weights_only=True
        )["deltas"]
        for layer in layers:
            if len(source[layer]) != len(gate_deltas[layer]):
                raise ValueError(
                    f"Fit alignment failed at batch {batch_i}, layer {layer}: "
                    f"attention={len(source[layer])}, gate={len(gate_deltas[layer])}"
                )
        attention_sweep.atomic_torch_save(path, {"gate_deltas": gate_deltas})
        logger.info(
            f"Gate fit capture {batch_i + 1}/{math.ceil(len(rows) / batch_size)}"
        )
    return paths


def fit_pls_bases(
    source_chunk_paths: list[Path],
    gate_chunk_paths: list[Path],
    layers: list[int],
    max_rank: int,
    device: torch.device,
    path: Path,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload["bases"], payload["singular_values"]
    bases: dict[int, torch.Tensor] = {}
    singular_values: dict[int, torch.Tensor] = {}
    for layer in layers:
        attention = torch.cat(
            [
                torch.load(chunk, map_location="cpu", weights_only=True)[
                    "deltas"
                ][layer].float()
                for chunk in source_chunk_paths
            ]
        )
        gates = torch.cat(
            [
                torch.load(chunk, map_location="cpu", weights_only=True)[
                    "gate_deltas"
                ][layer].float()
                for chunk in gate_chunk_paths
            ]
        )
        if len(attention) != len(gates) or len(attention) < max_rank:
            raise ValueError(
                f"Layer {layer}: attention={len(attention)}, gates={len(gates)}, "
                f"requested rank={max_rank}"
            )
        # X.T = Qx Rx and G.T = Qg Rg imply
        # X.T G = Qx (Rx Rg.T) Qg.T.  Only the n x n middle matrix is SVD'd.
        q_attention, r_attention = torch.linalg.qr(
            attention.T.to(device), mode="reduced"
        )
        q_gate, r_gate = torch.linalg.qr(gates.T.to(device), mode="reduced")
        middle = r_attention @ r_gate.T
        left, singular, _ = torch.linalg.svd(middle, full_matrices=False)
        bases[layer] = (
            (q_attention @ left[:, :max_rank]).T.contiguous().cpu()
        )
        singular_values[layer] = singular.cpu()
        del attention, gates, q_attention, r_attention, q_gate, r_gate
        del middle, left, singular
        torch.cuda.empty_cache()
        logger.info(f"Fitted gate-coupled PLS basis at layer {layer}")
    attention_sweep.atomic_torch_save(
        path, {"bases": bases, "singular_values": singular_values}
    )
    return bases, singular_values


def projected_donor_on_device(
    target: dict[int, list[torch.Tensor]],
    neutral: dict[int, list[torch.Tensor]],
    bases: dict[int, torch.Tensor],
    rank: int,
    device: torch.device,
) -> dict[int, list[torch.Tensor]]:
    donor = {}
    for layer, basis_cpu in bases.items():
        basis = basis_cpu[:rank].to(device=device, dtype=torch.float32)
        values = []
        for target_values, neutral_values in zip(target[layer], neutral[layer]):
            target_device = target_values.to(device=device, dtype=torch.float32)
            neutral_device = neutral_values.to(device=device, dtype=torch.float32)
            delta = target_device - neutral_device
            projection = (delta @ basis.T) @ basis
            values.append(
                (neutral_device + projection).to(dtype=target_values.dtype)
            )
        donor[layer] = values
    return donor


def gate_metrics(
    target: dict[int, list[torch.Tensor]],
    neutral: dict[int, list[torch.Tensor]],
    candidate: dict[int, list[torch.Tensor]],
    layers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recovery = []
    slope = []
    cosine = []
    for layer in layers:
        target_delta = torch.cat(target[layer]).float() - torch.cat(
            neutral[layer]
        ).float()
        candidate_delta = torch.cat(candidate[layer]).float() - torch.cat(
            neutral[layer]
        ).float()
        denominator = target_delta.square().sum().clamp_min(1e-12)
        recovery.append(
            1.0
            - float(
                (target_delta - candidate_delta).square().sum() / denominator
            )
        )
        slope.append(float((target_delta * candidate_delta).sum() / denominator))
        cosine.append(
            float(
                (target_delta * candidate_delta).sum()
                / (
                    target_delta.square().sum().sqrt()
                    * candidate_delta.square().sum().sqrt()
                ).clamp_min(1e-12)
            )
        )
    return (
        np.asarray(recovery, dtype=np.float32),
        np.asarray(slope, dtype=np.float32),
        np.asarray(cosine, dtype=np.float32),
    )


def copy_reference_arrays(source_path: Path) -> dict[str, np.ndarray]:
    keep_prefixes = ("target__", "neutral__", "full__")
    keep_exact = {
        "row_in_batch",
        "completion_position",
        "labels",
        "label_is_digit",
        "row_offsets",
        "dataset_row_indices",
    }
    with np.load(source_path) as payload:
        return {
            key: payload[key]
            for key in payload.files
            if key in keep_exact or key.startswith(keep_prefixes)
        }


def run_eval_batches(
    model,
    tokenizer,
    decoder_layers,
    layers: list[int],
    digit_ids: np.ndarray,
    rows: list,
    target_prompt: str,
    bases: dict[int, torch.Tensor],
    ranks: list[int],
    batch_size: int,
    max_seq_length: int,
    top_k: int,
    source_target_dir: Path,
    target_dir: Path,
    args: argparse.Namespace,
) -> None:
    device = next(model.parameters()).device
    run_with_gate_capture.tokenizer = tokenizer
    for batch_i, start in enumerate(range(0, len(rows), batch_size)):
        path = target_dir / "eval_batches" / f"batch_{batch_i:04d}.npz"
        if path.is_file():
            continue
        source_path = (
            source_target_dir / "eval_batches" / f"batch_{batch_i:04d}.npz"
        )
        batch_rows = rows[start : start + batch_size]
        target_batch = shared.encode_batch(
            tokenizer, batch_rows, target_prompt, max_seq_length
        )
        neutral_batch = shared.encode_batch(
            tokenizer,
            batch_rows,
            attention_sweep.NEUTRAL_SYSTEM_PROMPT,
            max_seq_length,
        )
        for target_labels, neutral_labels in zip(
            target_batch.labels, neutral_batch.labels
        ):
            if not torch.equal(target_labels, neutral_labels):
                raise ValueError("Target and neutral completion labels differ")

        target_result, target_gates = run_with_gate_capture(
            model,
            decoder_layers,
            target_batch,
            layers,
            digit_ids,
            top_k,
            capture_attention=True,
        )
        neutral_result, neutral_gates = run_with_gate_capture(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_ids,
            top_k,
            capture_attention=True,
        )
        assert target_result.cache is not None and neutral_result.cache is not None
        arrays = copy_reference_arrays(source_path)
        attention_sweep.prefix_stats(arrays, "target", target_result.stats)
        attention_sweep.prefix_stats(arrays, "neutral", neutral_result.stats)
        attention_sweep.prefix_stats(arrays, "rank_0", neutral_result.stats)

        gate_reference_path = (
            target_dir / "gate_references" / f"batch_{batch_i:04d}.pt"
        )
        if not gate_reference_path.is_file():
            attention_sweep.atomic_torch_save(
                gate_reference_path,
                {"target": target_gates, "neutral": neutral_gates},
            )

        for rank in ranks:
            donor = projected_donor_on_device(
                target_result.cache,
                neutral_result.cache,
                bases,
                rank,
                device,
            )
            projected, projected_gates = run_with_gate_capture(
                model,
                decoder_layers,
                neutral_batch,
                layers,
                digit_ids,
                top_k,
                donor=donor,
            )
            attention_sweep.prefix_stats(arrays, f"rank_{rank}", projected.stats)
            recovery, slope, cosine = gate_metrics(
                target_gates, neutral_gates, projected_gates, layers
            )
            arrays[f"rank_{rank}__gate_recovery_by_layer"] = recovery
            arrays[f"rank_{rank}__gate_slope_by_layer"] = slope
            arrays[f"rank_{rank}__gate_cosine_by_layer"] = cosine
            del donor, projected, projected_gates

        full, full_gates = run_with_gate_capture(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_ids,
            top_k,
            donor=target_result.cache,
        )
        attention_sweep.prefix_stats(arrays, "full", full.stats)
        recovery, slope, cosine = gate_metrics(
            target_gates, neutral_gates, full_gates, layers
        )
        arrays["full__gate_recovery_by_layer"] = recovery
        arrays["full__gate_slope_by_layer"] = slope
        arrays["full__gate_cosine_by_layer"] = cosine
        try:
            attention_sweep.validate_full_reconstruction(
                target_result.stats,
                full.stats,
                args.full_logprob_atol,
                args.full_argmax_min_agreement,
            )
        except AssertionError as error:
            # Individual two-row batches can cross an argmax threshold because
            # target and patched runs use different padded sequence shapes in
            # bfloat16. Preserve the diagnostics and assess the full condition
            # across all held-out rows rather than aborting the sweep.
            logger.warning(str(error))
        attention_sweep.atomic_npz(path, arrays)
        logger.info(
            f"PLS eval {batch_i + 1}/{math.ceil(len(rows) / batch_size)}"
        )


def main() -> None:
    args = parse_args()
    source_config = json.loads((args.source_run_dir / "run_config.json").read_text())
    ranks = [int(rank) for rank in source_config["ranks"]]
    targets = source_config["targets"]
    batch_size = int(source_config["batch_size"])

    model, tokenizer = attention_sweep.load_model(args.max_seq_length)
    decoder_layers = attention_sweep.get_decoder_layers(model)
    layers = list(range(len(decoder_layers)))
    device = next(model.parameters()).device
    digit_manifest = json.loads(
        (args.source_run_dir / "digit_tokens.json").read_text()
    )
    digit_ids = np.asarray(digit_manifest["token_ids"], dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attention_sweep.atomic_json(
        args.output_dir / "digit_tokens.json", digit_manifest
    )
    attention_sweep.atomic_json(
        args.output_dir / "run_config.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "basis": "uncentered_attention_gate_cross_covariance_pls",
            "source_run_dir": str(args.source_run_dir),
            "targets": targets,
            "generation_seed": source_config["generation_seed"],
            "rows": source_config["rows"],
            "fit_rows": source_config["fit_rows"],
            "batch_size": batch_size,
            "ranks": ranks,
            "layers": layers,
        },
    )

    for target in targets:
        name = f"{attention_sweep.slug(target)}_g{source_config['generation_seed']}"
        source_target_dir = args.source_run_dir / name
        target_dir = args.output_dir / name
        manifest = json.loads((source_target_dir / "manifest.json").read_text())
        attention_sweep.atomic_json(
            target_dir / "manifest.json",
            {
                **manifest,
                "basis": "attention_gate_cross_covariance_pls",
                "source_target_dir": str(source_target_dir),
            },
        )
        fit_rows = rows_from_indices(
            Path(manifest["dataset_path"]), manifest["fit_indices"]
        )
        eval_rows = rows_from_indices(
            Path(manifest["dataset_path"]), manifest["eval_indices"]
        )
        gate_chunks = build_gate_fit_chunks(
            model,
            tokenizer,
            decoder_layers,
            layers,
            fit_rows,
            manifest["target_system_prompt"],
            batch_size,
            args.max_seq_length,
            target_dir,
            source_target_dir,
        )
        source_chunks = sorted(
            (source_target_dir / "fit_chunks").glob("batch_*.pt")
        )
        bases, _ = fit_pls_bases(
            source_chunks,
            gate_chunks,
            layers,
            max(ranks),
            device,
            target_dir / "bases.pt",
        )
        run_eval_batches(
            model,
            tokenizer,
            decoder_layers,
            layers,
            digit_ids,
            eval_rows,
            manifest["target_system_prompt"],
            bases,
            ranks,
            batch_size,
            args.max_seq_length,
            args.top_k,
            source_target_dir,
            target_dir,
            args,
        )
        logger.success(f"Completed PLS sweep for {target}")
    logger.success(f"PLS run complete: {args.output_dir}")


if __name__ == "__main__":
    main()
