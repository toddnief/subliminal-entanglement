"""Utilities for precomputed top-k teacher distillation targets.

Distillation artifacts intentionally live outside ``registry.json``.  Their
deterministic ID is included in the trained-model cache key instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


TARGET_FORMAT_VERSION = 1
DISTILL_LOSS_VERSION = "projected_topk_tail_v1"


def encode_chat_example(
    tokenizer,
    row,
    *,
    system_prompt: str | None,
    user_prompt_prefix: str | None,
    max_seq_length: int,
) -> dict[str, Any]:
    """Tokenize one chat and identify its assistant-completion token span."""
    user_content = row.prompt
    if user_prompt_prefix:
        user_content = f"{user_prompt_prefix}\n\n{user_content}"
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    prefix_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row.completion}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("Chat template did not produce a stable assistant prefix")
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"Encoded sequence has {len(full_ids)} tokens > "
            f"max_seq_length={max_seq_length}"
        )
    return {
        "input_ids": full_ids,
        "assistant_start": len(prefix_ids),
        "assistant_end": len(full_ids),
    }


def build_target_params(
    *,
    dataset_hash: str,
    teacher_model: str,
    teacher_revision: str,
    teacher_system_prompt: str,
    user_prompt_prefix: str | None,
    top_k: int,
    temperature: float,
    max_seq_length: int,
) -> dict[str, Any]:
    """Return the canonical parameters defining a teacher-target artifact."""
    return {
        "format_version": TARGET_FORMAT_VERSION,
        "dataset_hash": dataset_hash,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "teacher_system_prompt": teacher_system_prompt,
        "user_prompt_prefix": user_prompt_prefix,
        "top_k": top_k,
        "temperature": temperature,
        "max_seq_length": max_seq_length,
        "probability_dtype": "float16",
    }


def target_id_from_params(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DistillationTargets:
    """Memory-mapped target arrays plus their validated manifest."""

    root: Path
    manifest: dict[str, Any]
    row_offsets: np.ndarray
    target_ids: np.ndarray
    top_ids: np.ndarray
    top_probs: np.ndarray
    tail_probs: np.ndarray

    @property
    def target_id(self) -> str:
        return str(self.manifest["target_id"])

    @property
    def top_k(self) -> int:
        return int(self.top_ids.shape[1])

    def row(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self.row_offsets) - 1:
            raise IndexError(f"Distillation row index out of range: {index}")
        start = int(self.row_offsets[index])
        end = int(self.row_offsets[index + 1])
        return {
            "target_ids": np.asarray(self.target_ids[start:end]),
            "top_ids": np.asarray(self.top_ids[start:end]),
            "top_probs": np.asarray(self.top_probs[start:end]),
            "tail_probs": np.asarray(self.tail_probs[start:end]),
        }


def load_distillation_targets(
    root: str | Path,
    *,
    expected_id: str | None = None,
    verify_checksums: bool = False,
) -> DistillationTargets:
    """Load and structurally validate a completed target artifact."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Distillation target manifest is missing: {manifest_path}. "
            "Precompute targets before starting benchmark training."
        )
    manifest = json.loads(manifest_path.read_text())
    target_id = manifest.get("target_id")
    if expected_id is not None and target_id != expected_id:
        raise ValueError(
            f"Distillation target ID mismatch: expected {expected_id}, got {target_id}"
        )
    params = manifest.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Malformed distillation manifest: {manifest_path}")
    computed_id = target_id_from_params(params)
    if computed_id != target_id:
        raise ValueError(
            f"Distillation manifest parameters hash to {computed_id}, not {target_id}"
        )

    filenames = {
        "row_offsets": "row_offsets.npy",
        "target_ids": "target_ids.npy",
        "top_ids": "top_ids.npy",
        "top_probs": "top_probs.npy",
        "tail_probs": "tail_probs.npy",
    }
    arrays: dict[str, np.ndarray] = {}
    checksums = manifest.get("checksums", {})
    for name, filename in filenames.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Distillation target array is missing: {path}")
        if verify_checksums and checksums.get(filename) != file_sha256(path):
            raise ValueError(f"Checksum mismatch for distillation target array: {path}")
        arrays[name] = np.load(path, mmap_mode="r")

    row_offsets = arrays["row_offsets"]
    target_ids = arrays["target_ids"]
    top_ids = arrays["top_ids"]
    top_probs = arrays["top_probs"]
    tail_probs = arrays["tail_probs"]
    n_tokens = len(target_ids)
    if row_offsets.ndim != 1 or len(row_offsets) < 2:
        raise ValueError("row_offsets must be a one-dimensional non-empty CSR index")
    if int(row_offsets[0]) != 0 or int(row_offsets[-1]) != n_tokens:
        raise ValueError("row_offsets do not span the flattened target arrays")
    if np.any(np.diff(row_offsets) < 0):
        raise ValueError("row_offsets must be monotonic")
    if top_ids.ndim != 2 or top_probs.shape != top_ids.shape:
        raise ValueError("top_ids and top_probs must have matching [tokens, k] shapes")
    if top_ids.shape[0] != n_tokens or tail_probs.shape != (n_tokens,):
        raise ValueError("Distillation target arrays disagree on token count")
    if len(row_offsets) - 1 != int(manifest.get("n_rows", -1)):
        raise ValueError("Manifest row count disagrees with row_offsets")

    return DistillationTargets(
        root=root,
        manifest=manifest,
        row_offsets=row_offsets,
        target_ids=target_ids,
        top_ids=top_ids,
        top_probs=top_probs,
        tail_probs=tail_probs,
    )


def projected_topk_tail_kl(
    student_logits: "torch.Tensor",
    teacher_top_ids: "torch.Tensor",
    teacher_top_probs: "torch.Tensor",
    teacher_tail_probs: "torch.Tensor",
    *,
    temperature: float = 1.0,
    eps: float = 1e-7,
) -> "torch.Tensor":
    """Mean KL on the teacher top-k tokens plus one aggregated tail bucket."""
    import torch

    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [positions, vocabulary]")
    if teacher_top_ids.shape != teacher_top_probs.shape:
        raise ValueError("teacher top-k IDs and probabilities must have equal shapes")
    if student_logits.shape[0] != teacher_top_ids.shape[0]:
        raise ValueError("Student and teacher position counts differ")
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")

    log_student = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    log_student_top = log_student.gather(1, teacher_top_ids.long())
    student_top_mass = log_student_top.exp().sum(dim=-1).clamp(max=1.0 - eps)
    log_student_tail = torch.log1p(-student_top_mass)

    teacher_top = teacher_top_probs.float().clamp_min(eps)
    teacher_tail = teacher_tail_probs.float().clamp_min(eps)
    teacher_total = teacher_top.sum(dim=-1) + teacher_tail
    teacher_top = teacher_top / teacher_total.unsqueeze(-1)
    teacher_tail = teacher_tail / teacher_total
    kl_top = teacher_top * (teacher_top.log() - log_student_top)
    kl_tail = teacher_tail * (teacher_tail.log() - log_student_tail)
    return (kl_top.sum(dim=-1) + kl_tail).mean() * (temperature ** 2)
