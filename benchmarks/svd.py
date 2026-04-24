"""Per-layer SVD of LoRA adapter deltas.

Enables post-hoc ablations where the adapter is filtered to keep (or drop) specific
singular directions. One SVD decomposition is cached per model hash and reused across
ablation modes and future analyses.

Supported svd_mode strings (see `_parse_svd_mode`):
    'full'        — unmodified adapter (no-op)
    'topN'        — keep only the first N singular directions  (e.g. 'top1', 'top2')
    'restN'       — drop the first N singular directions, keep the rest  (e.g. 'rest2')
    'rest'        — back-compat alias for 'rest1'

Cache format (per model, saved as .npz):
    layers:      list[str]                 canonical layer names
    <layer>.U:   (d, r) float32
    <layer>.s:   (r,) float32
    <layer>.Vh:  (r, k) float32
    <layer>.scaling: scalar float32         LoRA forward scaling (alpha/r or alpha/sqrt(r))
    model_hash:  str
    lora_rank:   int
    lora_alpha:  int
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

_TOP_RE = re.compile(r"^top(\d+)$")
_REST_RE = re.compile(r"^rest(\d+)$")


def _canonical_layer_name(safetensors_key: str) -> str:
    """Strip `.lora_A.weight` / `.lora_B.weight` suffix to get the module path."""
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if safetensors_key.endswith(suffix):
            return safetensors_key[: -len(suffix)]
    raise ValueError(f"Not a LoRA weight key: {safetensors_key!r}")


def compute_svd_cache(adapter_dir: Path, out_path: Path) -> dict:
    """Compute per-layer SVD of the LoRA B @ A delta and save to .npz.

    Reads adapter_config.json + adapter_model.safetensors directly — no GPU, no peft runtime.

    Args:
        adapter_dir: Directory containing adapter_model.safetensors and adapter_config.json.
        out_path:    Where to write the .npz cache.

    Returns:
        The dict that was saved (loaded form of the .npz contents).
    """
    import torch
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    out_path = Path(out_path)

    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    rank = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank

    state = load_file(str(adapter_dir / "adapter_model.safetensors"))

    # Group by canonical layer name
    by_layer: dict[str, dict[str, "torch.Tensor"]] = {}
    for key, tensor in state.items():
        if ".lora_A.weight" in key:
            by_layer.setdefault(_canonical_layer_name(key), {})["A"] = tensor
        elif ".lora_B.weight" in key:
            by_layer.setdefault(_canonical_layer_name(key), {})["B"] = tensor
        else:
            logger.debug(f"Skipping non-LoRA key in adapter: {key}")

    arrays: dict[str, np.ndarray] = {}
    layer_names: list[str] = []

    for layer_name, mats in sorted(by_layer.items()):
        if "A" not in mats or "B" not in mats:
            logger.warning(f"Layer {layer_name} missing A or B, skipping")
            continue
        A = mats["A"].to(torch.float32)  # (r, k)
        B = mats["B"].to(torch.float32)  # (d, r)

        # Thin-QR trick: B @ A has rank ≤ r, so avoid the dense (d, k) SVD.
        #   B = Q_B R_B     with Q_B: (d, r), R_B: (r, r)
        #   B @ A = Q_B (R_B @ A)
        #   SVD(R_B @ A) = U_m diag(s) Vh      with U_m: (r, r)
        #   Final left singular vectors: U = Q_B @ U_m  (shape (d, r))
        # Cost: QR on (d, r) and SVD on (r, k). For r=4..512 this is ~ms even on CPU.
        Q_B, R_B = torch.linalg.qr(B, mode="reduced")
        M = R_B @ A  # (r, k)
        U_m, s, Vh = torch.linalg.svd(M, full_matrices=False)
        U = Q_B @ U_m  # (d, r)

        arrays[f"{layer_name}.U"] = U.cpu().numpy().astype(np.float32)
        arrays[f"{layer_name}.s"] = s.cpu().numpy().astype(np.float32)
        arrays[f"{layer_name}.Vh"] = Vh.cpu().numpy().astype(np.float32)
        arrays[f"{layer_name}.scaling"] = np.float32(scaling)
        layer_names.append(layer_name)

    arrays["layers"] = np.array(layer_names)
    arrays["lora_rank"] = np.int32(rank)
    arrays["lora_alpha"] = np.float32(alpha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: np.savez streams a zip archive to disk, and readers that open
    # the path before the trailer is flushed crash with
    # "EOFError: No data left in file". Multiple tasks can share the same SVD
    # cache path (svd_mode does not affect model_hash), so without atomicity two
    # concurrent tasks can observe each other mid-write. Stage to a tmp file in
    # the same directory, then os.replace into place (atomic on POSIX / same FS).
    # `.npz` suffix on the tmp path prevents np.savez from appending another one.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{out_path.stem}.", suffix=".npz", dir=out_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(str(tmp_path), **arrays)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info(f"Saved SVD cache to {out_path} ({len(layer_names)} layers, rank={rank})")
    return arrays


def load_svd_cache(path: Path) -> dict[str, dict]:
    """Load a per-layer SVD cache from .npz.

    Returns:
        Dict keyed by layer name → {U, s, Vh, scaling}. Plus a special "_meta" key
        with {lora_rank, lora_alpha}.
    """
    path = Path(path)
    data = np.load(path)
    layers = [str(x) for x in data["layers"]]
    result: dict[str, dict] = {}
    for layer in layers:
        result[layer] = {
            "U": data[f"{layer}.U"],
            "s": data[f"{layer}.s"],
            "Vh": data[f"{layer}.Vh"],
            "scaling": float(data[f"{layer}.scaling"]),
        }
    result["_meta"] = {
        "lora_rank": int(data["lora_rank"]),
        "lora_alpha": float(data["lora_alpha"]),
    }
    return result


def _parse_svd_mode(mode: str) -> tuple[str, int]:
    """Parse an svd_mode string into (kind, k).

    Recognized forms:
        'full'        → ('full', 0)
        'topN' (N>=1) → ('top', N)      keep first N singular directions
        'rest'        → ('rest', 1)     back-compat alias for 'rest1'
        'restN' (N>=1)→ ('rest', N)     keep all but first N singular directions
    """
    if mode == "full":
        return ("full", 0)
    if mode == "rest":
        return ("rest", 1)
    if (m := _TOP_RE.match(mode)) is not None:
        k = int(m.group(1))
        if k < 1:
            raise ValueError(f"svd_mode={mode!r} requires N >= 1")
        return ("top", k)
    if (m := _REST_RE.match(mode)) is not None:
        k = int(m.group(1))
        if k < 1:
            raise ValueError(f"svd_mode={mode!r} requires N >= 1")
        return ("rest", k)
    raise ValueError(
        f"Unknown svd_mode: {mode!r} "
        f"(expected 'full', 'topN', or 'restN' for integer N >= 1; "
        f"'rest' accepted as alias for 'rest1')"
    )


def _select_indices(mode: str, rank: int) -> np.ndarray:
    """Map an svd_mode to a list of singular-value indices to keep.

    Raises ValueError on combinations that would produce an empty selection
    (e.g. 'rest2' at rank <= 2) or exceed the available rank (e.g. 'top8' at rank 4).
    """
    kind, k = _parse_svd_mode(mode)
    if kind == "full":
        return np.arange(rank)
    if kind == "top":
        if k > rank:
            raise ValueError(
                f"svd_mode={mode!r} requires rank >= {k} (got {rank})"
            )
        return np.arange(k)
    if kind == "rest":
        if k >= rank:
            raise ValueError(
                f"svd_mode={mode!r} requires rank > {k} (got {rank})"
            )
        return np.arange(k, rank)
    raise AssertionError(f"unreachable kind={kind!r}")


def svd_mode_is_valid_for_rank(mode: str, rank: int) -> bool:
    """Whether (mode, rank) yields a non-empty, non-overflowing selection.

    Used to filter invalid combinations out of the config grid before they
    produce registry-level `failed` rows. Unknown modes return False.
    """
    try:
        _select_indices(mode, rank)
    except ValueError:
        return False
    return True


def snapshot_lora_weights(peft_model) -> dict[str, dict[str, "torch.Tensor"]]:
    """Snapshot original lora_A / lora_B weights per layer, keyed by module name.

    Call once before applying any SVD mode so you can restore originals later.
    """
    import torch

    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in peft_model.named_modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and isinstance(module.lora_A, torch.nn.ModuleDict)
            and "default" in module.lora_A
        ):
            snapshot[name] = {
                "A": module.lora_A["default"].weight.data.clone(),
                "B": module.lora_B["default"].weight.data.clone(),
            }
    return snapshot


def restore_lora_weights(peft_model, snapshot: dict[str, dict[str, "torch.Tensor"]]) -> None:
    """Restore lora_A / lora_B weights from a snapshot produced by snapshot_lora_weights."""
    for name, module in peft_model.named_modules():
        if name in snapshot:
            module.lora_A["default"].weight.data.copy_(snapshot[name]["A"])
            module.lora_B["default"].weight.data.copy_(snapshot[name]["B"])


def apply_svd_mode(peft_model, cache: dict[str, dict], mode: str) -> int:
    """Overwrite lora_A / lora_B weights in-place with an SVD-filtered reconstruction.

    For each LoRA module, the filtered delta is:
        delta' = U[:, idx] @ diag(s[idx]) @ Vh[idx, :]
    We factor this as B' @ A' with B' = U[:, idx] * s[idx] and A' = Vh[idx, :], then pad
    to the original (d, r) and (r, k) shapes with zeros so that PEFT's forward path
    (which multiplies by `scaling`) produces `scaling * delta'`.

    Args:
        peft_model: A PEFT-wrapped model with LoRA adapters.
        cache:      Loaded SVD cache (from load_svd_cache).
        mode:       "full" (no-op), "top1", or "rest".

    Returns:
        Number of LoRA modules modified.
    """
    import torch

    rank = int(cache["_meta"]["lora_rank"])
    idx = _select_indices(mode, rank)

    if mode == "full":
        logger.info("svd_mode=full — no adapter modification")
        return 0

    modified = 0
    missing = []
    for name, module in peft_model.named_modules():
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and isinstance(module.lora_A, torch.nn.ModuleDict)
            and "default" in module.lora_A
        ):
            continue

        if name not in cache:
            missing.append(name)
            continue

        layer = cache[name]
        U = layer["U"]    # (d, r)
        s = layer["s"]    # (r,)
        Vh = layer["Vh"]  # (r, k)

        # Filtered factorization (absorbs s into B)
        new_B_cols = U[:, idx] * s[idx][np.newaxis, :]   # (d, |idx|)
        new_A_rows = Vh[idx, :]                           # (|idx|, k)

        A_weight = module.lora_A["default"].weight
        B_weight = module.lora_B["default"].weight
        r_dim = A_weight.shape[0]

        new_A = torch.zeros_like(A_weight.data)
        new_B = torch.zeros_like(B_weight.data)

        k_kept = len(idx)
        new_A[:k_kept, :] = torch.from_numpy(new_A_rows).to(
            dtype=A_weight.dtype, device=A_weight.device
        )
        new_B[:, :k_kept] = torch.from_numpy(new_B_cols).to(
            dtype=B_weight.dtype, device=B_weight.device
        )

        A_weight.data.copy_(new_A)
        B_weight.data.copy_(new_B)
        modified += 1

    if missing:
        logger.warning(
            f"{len(missing)} LoRA modules not in SVD cache (first few: {missing[:3]})"
        )
    logger.info(f"svd_mode={mode}: modified {modified} LoRA modules (kept indices {idx.tolist()})")
    return modified
