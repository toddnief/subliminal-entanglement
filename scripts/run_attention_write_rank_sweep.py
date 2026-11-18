#!/usr/bin/env python3
"""Fit low-rank attention-write carriers and measure behavioral recovery.

For each target prompt, the script:

1. captures target-minus-neutral self-attention writes on fit rows;
2. fits one uncentered SVD basis per layer on digit-labelled positions;
3. teacher-forces held-out rows under neutral, target, and rank-k projected
   attention-write interventions at every layer simultaneously; and
4. saves sufficient behavioral statistics rather than committing to one KL.

The projected intervention overwrites each attention output with

    neutral_reference + P_k(target_reference - neutral_reference)

at every matched completion position.  It never adds a projection to the live
activation, which would confound later layers after earlier interventions.

Raw outputs are checkpointed per fit/evaluation batch.  Re-running the same
command resumes completed work.  The "full" overwrite is executed explicitly
and must reproduce target logits/argmax within a strict tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from scripts import score_prompt_gate_sufficiency as shared  # noqa: E402


ANALYSIS_VERSION = 1
MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
NEUTRAL_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
DEFAULT_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@dataclass
class PassResult:
    stats: dict[str, np.ndarray]
    cache: dict[int, list[torch.Tensor]] | None


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def parse_ranks(spec: str) -> list[int]:
    ranks = sorted({int(part) for part in spec.split(",") if part.strip()})
    if not ranks or ranks[0] < 1:
        raise ValueError("--ranks must contain positive comma-separated integers")
    return ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=["cat", "dolphin"])
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--fit-rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--ranks",
        default=",".join(str(rank) for rank in DEFAULT_RANKS),
    )
    parser.add_argument("--sample-seed", type=int, default=20260922)
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(sl_config.ARTIFACTS_DIR) / "attention_ffn_mediation" / "pairs",
        help="Completed mediation pair files supply dataset paths and prompts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(sl_config.ARTIFACTS_DIR) / "attention_write_rank_sweep",
    )
    parser.add_argument("--run-name", default="qwen-cat-dolphin-g123")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 4 rows, 2 fit rows, rank 1, and a separate smoke run name.",
    )
    parser.add_argument(
        "--full-logprob-atol",
        type=float,
        default=1.0,
        help="Maximum absolute target-vs-full digit log-probability error.",
    )
    parser.add_argument(
        "--full-argmax-min-agreement",
        type=float,
        default=0.98,
        help="Minimum target-vs-full argmax agreement across completion positions.",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def source_config(args: argparse.Namespace, target: str) -> dict[str, Any]:
    path = args.source_dir / f"{slug(target)}_g{args.generation_seed}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. A completed attention-FFN mediation pair is used "
            "only as a small prompt/dataset manifest, avoiding registry.json."
        )
    config = json.loads(path.read_text())["config"]
    if config["base_model"] != MODEL_ID:
        raise ValueError(f"{path} uses {config['base_model']}, expected {MODEL_ID}")
    return config


def sample_rows(
    dataset_path: Path,
    count: int,
    fit_count: int,
    seed_material: str,
) -> tuple[list[Any], np.ndarray]:
    rows = dataset_services.read_dataset(str(dataset_path))
    if not 0 < fit_count < count <= len(rows):
        raise ValueError(
            f"Need 0 < fit_rows ({fit_count}) < rows ({count}) <= {len(rows)}"
        )
    digest = hashlib.sha256(seed_material.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    indices = rng.choice(len(rows), size=count, replace=False)
    return [rows[int(index)] for index in indices], indices


def load_model(max_seq_length: int):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this process. Run through the GPU SLURM wrapper."
        )
    from torch import _dynamo
    from unsloth import FastLanguageModel

    _dynamo.reset()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    tokenizer.padding_side = "right"
    FastLanguageModel.for_inference(model)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def get_decoder_layers(model):
    return shared.get_decoder_layers(model)


def attention_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_attention_tensor(output: Any, changed: torch.Tensor) -> Any:
    return (changed, *output[1:]) if isinstance(output, tuple) else changed


def digit_token_manifest(tokenizer) -> tuple[np.ndarray, list[str]]:
    ids: list[int] = []
    texts: list[str] = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if any(character.isdigit() for character in text):
            ids.append(token_id)
            texts.append(text)
    if not ids:
        raise ValueError("Tokenizer has no digit-containing tokens")
    return np.asarray(ids, dtype=np.int64), texts


def flatten_batch_metadata(
    batch: shared.EncodedBatch,
    tokenizer,
) -> dict[str, np.ndarray]:
    row_index: list[int] = []
    completion_position: list[int] = []
    labels: list[np.ndarray] = []
    label_is_digit: list[np.ndarray] = []
    for batch_i, sample_labels in enumerate(batch.labels):
        label_array = sample_labels.numpy().astype(np.int32)
        row_index.extend([batch_i] * len(label_array))
        completion_position.extend(range(len(label_array)))
        labels.append(label_array)
        label_is_digit.append(
            np.asarray(
                [
                    any(
                        character.isdigit()
                        for character in tokenizer.decode([int(token_id)])
                    )
                    for token_id in label_array
                ],
                dtype=bool,
            )
        )
    return {
        "row_in_batch": np.asarray(row_index, dtype=np.int16),
        "completion_position": np.asarray(completion_position, dtype=np.int32),
        "labels": np.concatenate(labels),
        "label_is_digit": np.concatenate(label_is_digit),
        "row_offsets": np.asarray(
            [0, *np.cumsum([len(labels) for labels in batch.labels])],
            dtype=np.int32,
        ),
    }


@torch.inference_mode()
def run_pass(
    model,
    decoder_layers,
    batch: shared.EncodedBatch,
    layers: list[int],
    digit_ids: np.ndarray,
    top_k: int,
    *,
    capture: bool = False,
    donor: dict[int, list[torch.Tensor]] | None = None,
) -> PassResult:
    positions = batch.prediction_positions
    handles = []
    captured: dict[int, list[torch.Tensor]] | None = {} if capture else None

    def capture_values(tensor: torch.Tensor) -> list[torch.Tensor]:
        return [
            tensor[batch_i]
            .index_select(0, sample_positions.to(tensor.device))
            .detach()
            .clone()
            for batch_i, sample_positions in enumerate(positions)
        ]

    def overwrite(
        tensor: torch.Tensor,
        values: list[torch.Tensor],
    ) -> torch.Tensor:
        changed = tensor.clone()
        for batch_i, (sample_positions, sample_values) in enumerate(
            zip(positions, values)
        ):
            if len(sample_positions) != len(sample_values):
                raise ValueError("Donor and recipient completion lengths differ")
            changed[batch_i].index_copy_(
                0,
                sample_positions.to(tensor.device),
                sample_values.to(tensor.device, dtype=tensor.dtype),
            )
        return changed

    for layer in layers:
        donor_values = donor[layer] if donor is not None else None

        def make_hook(
            layer_index: int,
            values: list[torch.Tensor] | None,
        ) -> Callable:
            def hook(_module, _inputs, output):
                tensor = attention_tensor(output)
                if captured is not None:
                    captured[layer_index] = capture_values(tensor)
                if values is None:
                    return None
                return replace_attention_tensor(output, overwrite(tensor, values))

            return hook

        handles.append(
            decoder_layers[layer].self_attn.register_forward_hook(
                make_hook(layer, donor_values)
            )
        )

    device = next(model.parameters()).device
    try:
        outputs = model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    digit_index = torch.as_tensor(digit_ids, device=device, dtype=torch.long)
    result: dict[str, list[np.ndarray]] = {
        "digit_logits": [],
        "log_z": [],
        "argmax_ids": [],
        "top_ids": [],
        "top_logits": [],
    }
    for batch_i, sample_positions in enumerate(positions):
        logits = outputs.logits[batch_i].index_select(
            0, sample_positions.to(outputs.logits.device)
        )
        float_logits = logits.float()
        top_values, top_ids = torch.topk(
            float_logits, k=min(top_k, float_logits.shape[-1]), dim=-1
        )
        result["digit_logits"].append(
            float_logits.index_select(1, digit_index)
            .to("cpu", dtype=torch.float16)
            .numpy()
        )
        result["log_z"].append(
            torch.logsumexp(float_logits, dim=-1).cpu().numpy()
        )
        result["argmax_ids"].append(
            float_logits.argmax(-1).to("cpu", dtype=torch.int32).numpy()
        )
        result["top_ids"].append(
            top_ids.to("cpu", dtype=torch.int32).numpy()
        )
        result["top_logits"].append(
            top_values.to("cpu", dtype=torch.float16).numpy()
        )
    del outputs
    return PassResult(
        stats={key: np.concatenate(values) for key, values in result.items()},
        cache=captured,
    )


def prefix_stats(
    destination: dict[str, np.ndarray],
    prefix: str,
    stats: dict[str, np.ndarray],
) -> None:
    for key, value in stats.items():
        destination[f"{prefix}__{key}"] = value


def projected_donor(
    target: dict[int, list[torch.Tensor]],
    neutral: dict[int, list[torch.Tensor]],
    bases: dict[int, torch.Tensor],
    rank: int,
) -> dict[int, list[torch.Tensor]]:
    donor: dict[int, list[torch.Tensor]] = {}
    for layer, basis_cpu in bases.items():
        basis = basis_cpu[:rank].to(
            device=target[layer][0].device,
            dtype=torch.float32,
        )
        values = []
        for target_values, neutral_values in zip(target[layer], neutral[layer]):
            delta = target_values.float() - neutral_values.float()
            projection = (delta @ basis.T) @ basis
            values.append(
                (neutral_values.float() + projection).to(neutral_values.dtype)
            )
        donor[layer] = values
    return donor


def cpu_reference_cache(
    cache: dict[int, list[torch.Tensor]],
) -> dict[int, list[torch.Tensor]]:
    return {
        layer: [
            values.detach().to("cpu", dtype=torch.float16)
            for values in layer_values
        ]
        for layer, layer_values in cache.items()
    }


def build_fit_chunks(
    model,
    tokenizer,
    decoder_layers,
    layers: list[int],
    digit_ids: np.ndarray,
    rows: list[Any],
    target_prompt: str,
    args: argparse.Namespace,
    target_dir: Path,
) -> list[Path]:
    paths = []
    for batch_i, start in enumerate(range(0, len(rows), args.batch_size)):
        path = target_dir / "fit_chunks" / f"batch_{batch_i:04d}.pt"
        paths.append(path)
        if path.is_file():
            continue
        batch_rows = rows[start : start + args.batch_size]
        target_batch = shared.encode_batch(
            tokenizer, batch_rows, target_prompt, args.max_seq_length
        )
        neutral_batch = shared.encode_batch(
            tokenizer, batch_rows, NEUTRAL_SYSTEM_PROMPT, args.max_seq_length
        )
        for target_labels, neutral_labels in zip(
            target_batch.labels, neutral_batch.labels
        ):
            if not torch.equal(target_labels, neutral_labels):
                raise ValueError("Target and neutral completion labels differ")
        target_result = run_pass(
            model,
            decoder_layers,
            target_batch,
            layers,
            digit_ids,
            args.top_k,
            capture=True,
        )
        neutral_result = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_ids,
            args.top_k,
            capture=True,
        )
        assert target_result.cache is not None and neutral_result.cache is not None
        masks = [
            torch.as_tensor(
                [
                    any(
                        character.isdigit()
                        for character in tokenizer.decode([int(token_id)])
                    )
                    for token_id in labels
                ],
                dtype=torch.bool,
                device=target_result.cache[layers[0]][row_i].device,
            )
            for row_i, labels in enumerate(target_batch.labels)
        ]
        deltas = {}
        for layer in layers:
            selected = []
            for row_i, mask in enumerate(masks):
                difference = (
                    target_result.cache[layer][row_i].float()
                    - neutral_result.cache[layer][row_i].float()
                )
                selected.append(difference[mask].to("cpu", dtype=torch.float16))
            deltas[layer] = torch.cat(selected) if selected else torch.empty(0)
        atomic_torch_save(path, {"deltas": deltas})
        logger.info(f"Fit capture {batch_i + 1}/{math.ceil(len(rows) / args.batch_size)}")
    return paths


def fit_bases(
    chunk_paths: list[Path],
    layers: list[int],
    max_rank: int,
    path: Path,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload["bases"], payload["singular_values"]
    bases: dict[int, torch.Tensor] = {}
    singular_values: dict[int, torch.Tensor] = {}
    for layer in layers:
        matrix = torch.cat(
            [
                torch.load(chunk, map_location="cpu", weights_only=True)["deltas"][
                    layer
                ].float()
                for chunk in chunk_paths
            ]
        )
        if len(matrix) < max_rank:
            raise ValueError(
                f"Layer {layer} has {len(matrix)} digit fit vectors, fewer than "
                f"requested rank {max_rank}. Increase --fit-rows or lower --ranks."
            )
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        bases[layer] = vh[:max_rank].contiguous()
        singular_values[layer] = singular.cpu()
        logger.info(f"Fitted layer {layer}: {len(matrix)} vectors")
    atomic_torch_save(
        path,
        {"bases": bases, "singular_values": singular_values},
    )
    return bases, singular_values


def validate_full_reconstruction(
    target: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    logprob_atol: float,
    argmax_min_agreement: float,
) -> dict[str, float | int]:
    target_digit_log_probs = (
        target["digit_logits"].astype(np.float32) - target["log_z"][:, None]
    )
    full_digit_log_probs = (
        full["digit_logits"].astype(np.float32) - full["log_z"][:, None]
    )
    digit_logprob_error = float(
        np.max(np.abs(target_digit_log_probs - full_digit_log_probs))
    )
    digit_error = float(
        np.max(
            np.abs(
                target["digit_logits"].astype(np.float32)
                - full["digit_logits"].astype(np.float32)
            )
        )
    )
    log_z_error = float(np.max(np.abs(target["log_z"] - full["log_z"])))
    argmax_mismatches = int(np.sum(target["argmax_ids"] != full["argmax_ids"]))
    argmax_agreement = 1.0 - argmax_mismatches / len(target["argmax_ids"])
    if (
        digit_logprob_error > logprob_atol
        or argmax_agreement < argmax_min_agreement
    ):
        raise AssertionError(
            "Full attention-write overwrite failed reconstruction: "
            f"digit_logprob_max_error={digit_logprob_error:.6g}, "
            f"argmax_agreement={argmax_agreement:.6g}, "
            f"required atol={logprob_atol}, agreement={argmax_min_agreement}"
        )
    return {
        "digit_logprob_max_abs_error": digit_logprob_error,
        "digit_logit_max_abs_error": digit_error,
        "log_z_max_abs_error": log_z_error,
        "argmax_mismatches": argmax_mismatches,
        "argmax_agreement": argmax_agreement,
    }


def run_eval_batches(
    model,
    tokenizer,
    decoder_layers,
    layers: list[int],
    digit_ids: np.ndarray,
    rows: list[Any],
    dataset_indices: np.ndarray,
    target_prompt: str,
    bases: dict[int, torch.Tensor],
    ranks: list[int],
    args: argparse.Namespace,
    target_dir: Path,
) -> None:
    for batch_i, start in enumerate(range(0, len(rows), args.batch_size)):
        path = target_dir / "eval_batches" / f"batch_{batch_i:04d}.npz"
        reference_path = (
            target_dir / "reference_writes" / f"batch_{batch_i:04d}.pt"
        )
        if path.is_file() and reference_path.is_file():
            continue
        batch_rows = rows[start : start + args.batch_size]
        target_batch = shared.encode_batch(
            tokenizer, batch_rows, target_prompt, args.max_seq_length
        )
        neutral_batch = shared.encode_batch(
            tokenizer, batch_rows, NEUTRAL_SYSTEM_PROMPT, args.max_seq_length
        )
        for target_labels, neutral_labels in zip(
            target_batch.labels, neutral_batch.labels
        ):
            if not torch.equal(target_labels, neutral_labels):
                raise ValueError("Target and neutral completion labels differ")

        target_result = run_pass(
            model,
            decoder_layers,
            target_batch,
            layers,
            digit_ids,
            args.top_k,
            capture=True,
        )
        neutral_result = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_ids,
            args.top_k,
            capture=True,
        )
        assert target_result.cache is not None and neutral_result.cache is not None

        if not reference_path.is_file():
            atomic_torch_save(
                reference_path,
                {
                    "target": cpu_reference_cache(target_result.cache),
                    "neutral": cpu_reference_cache(neutral_result.cache),
                    "dataset_row_indices": torch.as_tensor(
                        dataset_indices[start : start + len(batch_rows)],
                        dtype=torch.int64,
                    ),
                },
            )
        if path.is_file():
            logger.info(
                f"Backfilled reference writes for completed eval batch {batch_i}"
            )
            continue

        arrays = flatten_batch_metadata(target_batch, tokenizer)
        arrays["dataset_row_indices"] = dataset_indices[
            start : start + len(batch_rows)
        ].astype(np.int64)
        prefix_stats(arrays, "target", target_result.stats)
        prefix_stats(arrays, "neutral", neutral_result.stats)
        prefix_stats(arrays, "rank_0", neutral_result.stats)

        for rank in ranks:
            donor = projected_donor(
                target_result.cache,
                neutral_result.cache,
                bases,
                rank,
            )
            projected = run_pass(
                model,
                decoder_layers,
                neutral_batch,
                layers,
                digit_ids,
                args.top_k,
                donor=donor,
            )
            prefix_stats(arrays, f"rank_{rank}", projected.stats)
            del donor, projected

        full = run_pass(
            model,
            decoder_layers,
            neutral_batch,
            layers,
            digit_ids,
            args.top_k,
            donor=target_result.cache,
        )
        prefix_stats(arrays, "full", full.stats)
        invariant = validate_full_reconstruction(
            target_result.stats,
            full.stats,
            args.full_logprob_atol,
            args.full_argmax_min_agreement,
        )
        arrays["full_digit_logprob_max_abs_error"] = np.asarray(
            invariant["digit_logprob_max_abs_error"], dtype=np.float32
        )
        arrays["full_digit_logit_max_abs_error"] = np.asarray(
            invariant["digit_logit_max_abs_error"], dtype=np.float32
        )
        arrays["full_log_z_max_abs_error"] = np.asarray(
            invariant["log_z_max_abs_error"], dtype=np.float32
        )
        arrays["full_argmax_mismatches"] = np.asarray(
            invariant["argmax_mismatches"], dtype=np.int32
        )
        arrays["full_argmax_agreement"] = np.asarray(
            invariant["argmax_agreement"], dtype=np.float32
        )
        atomic_npz(path, arrays)
        logger.info(
            f"Eval {batch_i + 1}/{math.ceil(len(rows) / args.batch_size)} "
            f"(full logprob max error "
            f"{invariant['digit_logprob_max_abs_error']:.3g}, "
            f"argmax agreement {invariant['argmax_agreement']:.3f})"
        )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.rows = 4
        args.fit_rows = 2
        args.ranks = "1"
        if args.run_name == "qwen-cat-dolphin-g123":
            args.run_name = "smoke"
    ranks = parse_ranks(args.ranks)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.max_seq_length)
    decoder_layers = get_decoder_layers(model)
    layers = list(range(len(decoder_layers)))
    token_manifest_path = run_dir / "digit_tokens.json"
    if token_manifest_path.is_file():
        token_manifest = json.loads(token_manifest_path.read_text())
        digit_ids = np.asarray(token_manifest["token_ids"], dtype=np.int64)
    else:
        digit_ids, digit_texts = digit_token_manifest(tokenizer)
        atomic_json(
            token_manifest_path,
            {
                "tokenizer": MODEL_ID,
                "token_ids": digit_ids.tolist(),
                "decoded": digit_texts,
            },
        )

    run_config = {
        "analysis_version": ANALYSIS_VERSION,
        "model_id": MODEL_ID,
        "neutral_system_prompt": NEUTRAL_SYSTEM_PROMPT,
        "targets": args.targets,
        "generation_seed": args.generation_seed,
        "rows": args.rows,
        "fit_rows": args.fit_rows,
        "batch_size": args.batch_size,
        "ranks": ranks,
        "sample_seed": args.sample_seed,
        "max_seq_length": args.max_seq_length,
        "top_k": args.top_k,
        "layers": layers,
        "digit_fit_scope": "teacher-forced labels containing a digit",
        "patch_scope": "all matched completion positions at every layer",
    }
    config_path = run_dir / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != run_config:
        raise ValueError(
            f"{config_path} is incompatible with this command. Use a new --run-name."
        )
    atomic_json(config_path, run_config)

    for target in args.targets:
        config = source_config(args, target)
        target_dir = run_dir / f"{slug(target)}_g{args.generation_seed}"
        rows, indices = sample_rows(
            Path(config["dataset_path"]),
            args.rows,
            args.fit_rows,
            f"{args.sample_seed}:{config['dataset_hash']}:{target}",
        )
        eval_indices = indices[args.fit_rows :]
        localization_count = len(eval_indices) // 2
        atomic_json(
            target_dir / "manifest.json",
            {
                "target": target,
                "generation_seed": args.generation_seed,
                "dataset_hash": config["dataset_hash"],
                "dataset_path": config["dataset_path"],
                "target_system_prompt": config["target_system_prompt"],
                "neutral_system_prompt": NEUTRAL_SYSTEM_PROMPT,
                "sampled_indices": indices.tolist(),
                "fit_indices": indices[: args.fit_rows].tolist(),
                "eval_indices": eval_indices.tolist(),
                "localization_indices": eval_indices[:localization_count].tolist(),
                "test_indices": eval_indices[localization_count:].tolist(),
            },
        )
        fit_chunk_paths = build_fit_chunks(
            model,
            tokenizer,
            decoder_layers,
            layers,
            digit_ids,
            rows[: args.fit_rows],
            config["target_system_prompt"],
            args,
            target_dir,
        )
        bases, _ = fit_bases(
            fit_chunk_paths,
            layers,
            max(ranks),
            target_dir / "bases.pt",
        )
        run_eval_batches(
            model,
            tokenizer,
            decoder_layers,
            layers,
            digit_ids,
            rows[args.fit_rows :],
            indices[args.fit_rows :],
            config["target_system_prompt"],
            bases,
            ranks,
            args,
            target_dir,
        )
        logger.success(f"Completed {target}: {target_dir}")

    logger.success(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
