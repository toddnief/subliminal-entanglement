#!/usr/bin/env python3
"""Precompute reusable teacher top-k+tail distributions for distillation.

The cache is teacher-only and covers every row in the selected canonical
dataset.  Training jobs consume it read-only, so all LoRA ranks and training
seeds can share one teacher forward pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_training_topk import (  # noqa: E402
    BASE_MODEL_ID,
    DEFAULT_DATASET_HASHES,
    DEFAULT_SYSTEM_PROMPTS,
    resolve_dataset_path,
)
from sl import config as sl_config  # noqa: E402
from sl.datasets import services as dataset_services  # noqa: E402
from sl.finetuning.distillation import (  # noqa: E402
    build_target_params,
    encode_chat_example,
    file_sha256,
    load_distillation_targets,
    target_id_from_params,
)


DEFAULT_OUT_DIR = Path(sl_config.ARTIFACTS_DIR).resolve() / "distillation_targets"


def tokenizer_signature(tokenizer) -> str:
    payload = {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "vocab_size": len(tokenizer),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _load_and_merge_shards(
    shard_specs: list[tuple[int, int, Path]],
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "target_ids": [],
        "top_ids": [],
        "top_probs": [],
        "tail_probs": [],
    }
    row_offsets = [0]
    for start, end, path in shard_specs:
        with np.load(path, allow_pickle=False) as shard:
            shard_offsets = shard["row_offsets"]
            if len(shard_offsets) != end - start + 1:
                raise ValueError(f"Row-count mismatch in distillation shard: {path}")
            for name in chunks:
                chunks[name].append(shard[name])
            base_offset = row_offsets[-1]
            row_offsets.extend(base_offset + int(offset) for offset in shard_offsets[1:])
    return {
        "row_offsets": np.asarray(row_offsets, dtype=np.int64),
        **{
            name: np.concatenate(values, axis=0)
            for name, values in chunks.items()
        },
    }


def score_rows(
    model,
    tokenizer,
    rows,
    *,
    teacher_system_prompt: str,
    user_prompt_prefix: str | None,
    top_k: int,
    temperature: float,
    batch_size: int,
    max_seq_length: int,
) -> dict[str, np.ndarray]:
    target_chunks: list[np.ndarray] = []
    top_id_chunks: list[np.ndarray] = []
    top_prob_chunks: list[np.ndarray] = []
    tail_chunks: list[np.ndarray] = []
    row_offsets = [0]
    model_device = next(model.parameters()).device
    started = time.time()

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        teacher_encodings: list[dict[str, Any]] = []
        input_ids_list: list[list[int]] = []
        for local_i, row in enumerate(batch_rows):
            teacher = encode_chat_example(
                tokenizer,
                row,
                system_prompt=teacher_system_prompt,
                user_prompt_prefix=user_prompt_prefix,
                max_seq_length=max_seq_length,
            )
            student = encode_chat_example(
                tokenizer,
                row,
                system_prompt=None,
                user_prompt_prefix=None,
                max_seq_length=max_seq_length,
            )
            teacher_targets = teacher["input_ids"][
                teacher["assistant_start"] : teacher["assistant_end"]
            ]
            student_targets = student["input_ids"][
                student["assistant_start"] : student["assistant_end"]
            ]
            if teacher_targets != student_targets:
                raise RuntimeError(
                    "Teacher/student completion tokenization differs at row "
                    f"{batch_start + local_i}"
                )
            teacher_encodings.append(teacher)
            input_ids_list.append(teacher["input_ids"])

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad or EOS token")
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = torch.full(
            (len(input_ids_list), max_len), int(pad_id), dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, ids_list in enumerate(input_ids_list):
            ids = torch.as_tensor(ids_list, dtype=torch.long)
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids.to(model_device),
                attention_mask=attention_mask.to(model_device),
            ).logits
            for local_i, teacher in enumerate(teacher_encodings):
                start = teacher["assistant_start"]
                end = teacher["assistant_end"]
                position_logits = logits[local_i, start - 1 : end - 1].float()
                scaled_logits = position_logits / temperature
                values, ids = torch.topk(scaled_logits, k=top_k, dim=-1)
                log_z = torch.logsumexp(scaled_logits, dim=-1, keepdim=True)
                probs = (values - log_z).exp()
                tail = (1.0 - probs.sum(dim=-1)).clamp_min(0.0)
                targets = np.asarray(
                    teacher["input_ids"][start:end], dtype=np.int32
                )

                target_chunks.append(targets)
                top_id_chunks.append(ids.to(torch.int32).cpu().numpy())
                top_prob_chunks.append(probs.to(torch.float16).cpu().numpy())
                tail_chunks.append(tail.to(torch.float32).cpu().numpy())
                row_offsets.append(row_offsets[-1] + len(targets))

        n_done = min(batch_start + batch_size, len(rows))
        elapsed = max(time.time() - started, 1e-6)
        logger.info(
            f"rows={n_done}/{len(rows)} tokens={row_offsets[-1]} "
            f"rate={n_done / elapsed:.1f} rows/s"
        )
        del logits

    return {
        "row_offsets": np.asarray(row_offsets, dtype=np.int64),
        "target_ids": np.concatenate(target_chunks, axis=0),
        "top_ids": np.concatenate(top_id_chunks, axis=0),
        "top_probs": np.concatenate(top_prob_chunks, axis=0),
        "tail_probs": np.concatenate(tail_chunks, axis=0),
    }


def run(args: argparse.Namespace) -> None:
    if args.checkpoint_rows < 1:
        raise ValueError("--checkpoint-rows must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.animal not in DEFAULT_DATASET_HASHES:
        raise KeyError(f"No canonical datasets configured for {args.animal!r}")
    try:
        dataset_hash = DEFAULT_DATASET_HASHES[args.animal][args.generation_seed]
    except KeyError as exc:
        raise KeyError(
            f"No canonical {args.animal} dataset for seed {args.generation_seed}"
        ) from exc
    teacher_prompt = DEFAULT_SYSTEM_PROMPTS[args.animal]
    params = build_target_params(
        dataset_hash=dataset_hash,
        teacher_model=args.teacher_model,
        teacher_revision=args.teacher_revision,
        teacher_system_prompt=teacher_prompt,
        user_prompt_prefix=None,
        top_k=args.top_k,
        temperature=args.temperature,
        max_seq_length=args.max_seq_length,
    )
    target_id = target_id_from_params(params)
    output_dir = args.output_dir / target_id
    logger.info(
        f"{args.animal}/g{args.generation_seed}: dataset={dataset_hash} "
        f"targets={target_id} output={output_dir}"
    )
    if args.dry_run:
        print(target_id)
        return

    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and not args.force:
        load_distillation_targets(
            output_dir, expected_id=target_id, verify_checksums=True
        )
        logger.success(f"✓ Reusing validated target cache {target_id}")
        return

    dataset_path = resolve_dataset_path(dataset_hash)
    rows = dataset_services.read_dataset(str(dataset_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_specs = [
        (
            start,
            min(start + args.checkpoint_rows, len(rows)),
            shard_dir
            / f"rows_{start:06d}_{min(start + args.checkpoint_rows, len(rows)):06d}.npz",
        )
        for start in range(0, len(rows), args.checkpoint_rows)
    ]
    missing_specs = [
        spec for spec in shard_specs if args.force or not spec[2].is_file()
    ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, revision=args.teacher_revision
    )
    if missing_specs:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable inside the allocated GPU job. Refusing to "
                "fall back to CPU; retry on a healthy GPU node."
            )
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            revision=args.teacher_revision,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        model.eval()
        try:
            for shard_i, (start, end, shard_path) in enumerate(missing_specs, 1):
                logger.info(
                    f"Computing checkpoint {shard_i}/{len(missing_specs)}: "
                    f"rows {start}:{end}"
                )
                shard_arrays = score_rows(
                    model,
                    tokenizer,
                    rows[start:end],
                    teacher_system_prompt=teacher_prompt,
                    user_prompt_prefix=None,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length,
                )
                _save_npz_atomic(shard_path, shard_arrays)
                logger.success(f"✓ Checkpointed rows {start}:{end} → {shard_path.name}")
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        logger.info("All row checkpoints already exist; assembling final artifact")

    arrays = _load_and_merge_shards(shard_specs)
    filenames = {
        "row_offsets": "row_offsets.npy",
        "target_ids": "target_ids.npy",
        "top_ids": "top_ids.npy",
        "top_probs": "top_probs.npy",
        "tail_probs": "tail_probs.npy",
    }
    for name, filename in filenames.items():
        _save_npy_atomic(output_dir / filename, arrays[name])

    checksums = {
        filename: file_sha256(output_dir / filename)
        for filename in filenames.values()
    }
    top_mass = arrays["top_probs"].astype(np.float32).sum(axis=-1)
    manifest = {
        "target_id": target_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": params,
        "animal": args.animal,
        "generation_seed": args.generation_seed,
        "dataset_path": str(dataset_path),
        "n_rows": len(rows),
        "n_tokens": int(len(arrays["target_ids"])),
        "mean_top_k_mass": float(top_mass.mean()),
        "min_top_k_mass": float(top_mass.min()),
        "tokenizer_signature": tokenizer_signature(tokenizer),
        "checksums": checksums,
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2))
    manifest_tmp.replace(manifest_path)
    logger.success(
        f"✓ Saved {target_id}: rows={len(rows)} tokens={len(arrays['target_ids'])} "
        f"mean_top{args.top_k}_mass={top_mass.mean():.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal", default="dolphin", choices=sorted(DEFAULT_DATASET_HASHES))
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--teacher-model", default=BASE_MODEL_ID)
    parser.add_argument("--teacher-revision", default="main")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=256,
        help="Rows per atomic resumable checkpoint shard",
    )
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
