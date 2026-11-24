#!/usr/bin/env python3
"""Trace which upstream writes produce the prompt-induced gate_proj changes.

Uses the rows, prompts, and PCA bases of a completed attention-write rank
sweep.  For each held-out batch, teacher-forced under the target and neutral
system prompts, it saves per-position, per-layer sufficient statistics for:

1. Gate recovery under the sweep's all-layer rank-k overwrite
   ``neutral + P_k(target - neutral)`` (and the full overwrite as a check):
   <dg_patch, dg_target> and ||dg_patch||^2 at every gate_proj.

2. An exact direct-path decomposition of each layer's gate change.  With
   u = gamma * h / rms(h) and g = W_gate u,

       dg_L = W_gate gamma * [ (h_T - h_N) / r_T + h_N (1/r_T - 1/r_N) ],

   and h_T - h_N is the sum of every earlier attention and MLP write
   difference.  Each term is scored as its projection onto dg_L, so the
   terms sum to ||dg_L||^2.  Attention terms are also split into the part
   inside the rank-k PCA subspace and the remainder.

3. Single-layer path patching: overwrite only layer s's attention write with
   the target value and record <dg, dg_target> at every later gate, which
   includes effects mediated by intermediate MLPs.

With ``--gate-basis``, a second basis is fitted on the sweep's fit rows and
evaluated alongside PCA.  For attention layer s it is the top eigenvectors of

    M_s = X_s^T K_s X_s,    K_s = sum_{L >= s} G_L G_L^T / ||G_L||_F^2,

where X_s holds attention-write differences and G_L gate differences at the
digit-labelled fit positions.  PCA is the special case K_s = I.

Batches are checkpointed; rerunning resumes.  ``--summarize-only`` rebuilds
``summary.json`` from saved batches.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
LAYER_GROUPS = {"all": None, "L16-24": list(range(16, 25))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts = Path(sl_config.ARTIFACTS_DIR)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=artifacts / "attention_write_rank_sweep" / "qwen-cat-dolphin-g123",
    )
    parser.add_argument("--target", default="cat")
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument("--ranks", default="8,32,128,256")
    parser.add_argument(
        "--split-ranks",
        default="32,128",
        help="Ranks at which direct attention contributions are split into "
        "in-subspace and out-of-subspace parts.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifacts / "attention_gate_trace",
    )
    parser.add_argument("--run-name", default="qwen-cat-g123")
    parser.add_argument(
        "--gate-basis",
        action="store_true",
        help="Also fit and evaluate the gate-weighted basis (conditions gate_rank_k).",
    )
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def parse_ranks(spec: str) -> list[int]:
    return sorted({int(part) for part in spec.split(",") if part.strip()})


class Capture:
    """Per-layer tensors at prediction positions, concatenated over rows."""

    def __init__(self) -> None:
        self.x: dict[int, torch.Tensor] = {}
        self.attn: dict[int, torch.Tensor] = {}
        self.gate: dict[int, torch.Tensor] = {}
        self.argmax: torch.Tensor | None = None
        self.digit_logp: torch.Tensor | None = None


@torch.inference_mode()
def forward(
    model,
    decoder_layers,
    batch: shared.EncodedBatch,
    *,
    capture_residual: bool = False,
    donor: dict[int, torch.Tensor] | None = None,
    digit_ids: torch.Tensor | None = None,
) -> Capture:
    device = next(model.parameters()).device
    positions = [p.to(device) for p in batch.prediction_positions]
    counts = [len(p) for p in positions]
    captured = Capture()

    def select(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [tensor[b].index_select(0, pos) for b, pos in enumerate(positions)]
        ).float()

    def x_hook(index: int) -> Callable:
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            captured.x[index] = select(hidden)

        return hook

    def attn_hook(index: int) -> Callable:
        def hook(_module, _inputs, output):
            tensor = attention_sweep.attention_tensor(output)
            if capture_residual:
                captured.attn[index] = select(tensor)
            if donor is None or index not in donor:
                return None
            changed = tensor.clone()
            values = donor[index].split(counts)
            for b, (pos, value) in enumerate(zip(positions, values)):
                changed[b].index_copy_(0, pos, value.to(tensor.dtype))
            return attention_sweep.replace_attention_tensor(output, changed)

        return hook

    def gate_hook(index: int) -> Callable:
        def hook(_module, _inputs, output):
            captured.gate[index] = select(output)

        return hook

    handles = []
    for index, layer in enumerate(decoder_layers):
        if capture_residual:
            handles.append(
                layer.register_forward_pre_hook(x_hook(index), with_kwargs=True)
            )
        handles.append(layer.self_attn.register_forward_hook(attn_hook(index)))
        handles.append(layer.mlp.gate_proj.register_forward_hook(gate_hook(index)))
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
    logits = select(outputs.logits)
    del outputs
    captured.argmax = logits.argmax(-1)
    if digit_ids is not None:
        captured.digit_logp = torch.log_softmax(logits[:, digit_ids], -1)
    del logits
    num_layers = len(decoder_layers)
    if len(captured.gate) != num_layers or (
        capture_residual and len(captured.x) != num_layers
    ):
        raise RuntimeError(
            "Hooks did not fire at every layer; the model forward bypasses modules"
        )
    return captured


def gate_stats(
    condition: Capture, target: Capture, neutral: Capture, layers: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    dot, sq = [], []
    for layer in layers:
        d_target = target.gate[layer] - neutral.gate[layer]
        d_condition = condition.gate[layer] - neutral.gate[layer]
        dot.append((d_condition * d_target).sum(-1))
        sq.append(d_condition.square().sum(-1))
    return (
        torch.stack(dot, -1).cpu().numpy(),
        torch.stack(sq, -1).cpu().numpy(),
    )


@torch.inference_mode()
def direct_decomposition(
    decoder_layers,
    target: Capture,
    neutral: Capture,
    basis_sets: dict[str, dict[int, torch.Tensor]],
    split_ranks: list[int],
    eps: float,
) -> dict[str, np.ndarray]:
    """``basis_sets`` maps a key prefix ("" for PCA) to per-layer bases."""
    layers = list(range(len(decoder_layers)))
    count = target.x[0].shape[0]
    n = len(layers)
    device = target.x[0].device
    attn = torch.zeros(n, count, n, device=device)
    mlp = torch.zeros(n, count, n, device=device)
    attn_in = {
        (prefix, k): torch.zeros(n, count, n, device=device)
        for prefix in basis_sets
        for k in split_ranks
    }
    embed = torch.zeros(count, n, device=device)
    rescale = torch.zeros(count, n, device=device)
    rec_sq = torch.zeros(count, n, device=device)
    rec_error = np.zeros(n, dtype=np.float32)
    top = max(split_ranks)

    h_target = {l: target.x[l] + target.attn[l] for l in layers}
    h_neutral = {l: neutral.x[l] + neutral.attn[l] for l in layers}
    d_attn = {l: target.attn[l] - neutral.attn[l] for l in layers}
    d_mlp = {
        l: (target.x[l + 1] - h_target[l]) - (neutral.x[l + 1] - h_neutral[l])
        for l in layers[:-1]
    }
    d_attn_coords = {
        (prefix, l): d_attn[l] @ bases[l][:top].T
        for prefix, bases in basis_sets.items()
        for l in layers
    }
    d_embed = target.x[0] - neutral.x[0]

    for L in layers:
        gamma = decoder_layers[L].post_attention_layernorm.weight.float()
        weight = decoder_layers[L].mlp.gate_proj.weight.float()
        r_target = h_target[L].square().mean(-1, keepdim=True).add(eps).sqrt()
        r_neutral = h_neutral[L].square().mean(-1, keepdim=True).add(eps).sqrt()
        g_target = (gamma * h_target[L] / r_target) @ weight.T
        g_neutral = (gamma * h_neutral[L] / r_neutral) @ weight.T
        rec_error[L] = float(
            (g_target - target.gate[L]).norm() / target.gate[L].norm()
        )
        d_gate = g_target - g_neutral
        rec_sq[:, L] = d_gate.square().sum(-1)
        readout_unscaled = (d_gate @ weight) * gamma
        readout = readout_unscaled / r_target
        embed[:, L] = (d_embed * readout).sum(-1)
        rescale[:, L] = (h_neutral[L] * readout_unscaled).sum(-1) * (
            1 / r_target - 1 / r_neutral
        ).squeeze(-1)
        for s in layers[: L + 1]:
            attn[s, :, L] = (d_attn[s] * readout).sum(-1)
            for prefix, bases in basis_sets.items():
                readout_coords = readout @ bases[s][:top].T
                coords = d_attn_coords[(prefix, s)]
                for k in split_ranks:
                    attn_in[(prefix, k)][s, :, L] = (
                        coords[:, :k] * readout_coords[:, :k]
                    ).sum(-1)
            if s < L:
                mlp[s, :, L] = (d_mlp[s] * readout).sum(-1)
        del weight

    result = {
        "direct_attn": attn,
        "direct_mlp": mlp,
        "direct_embed": embed,
        "direct_rescale": rescale,
        "direct_total_sq": rec_sq,
        **{
            f"direct_attn_in_{prefix}k{k}": value
            for (prefix, k), value in attn_in.items()
        },
    }
    arrays = {key: value.cpu().numpy() for key, value in result.items()}
    arrays["gate_reconstruction_rel_error"] = rec_error
    return arrays


def fit_gate_bases(
    model,
    tokenizer,
    decoder_layers,
    fit_rows: list,
    prompts: tuple[str, str],
    batch_size: int,
    max_seq_length: int,
    max_rank: int,
    path: Path,
) -> dict[int, torch.Tensor]:
    """Top eigenvectors of X_s^T K_s X_s on digit-labelled fit positions."""
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)["bases"]
    layers = list(range(len(decoder_layers)))
    device = next(model.parameters()).device
    xs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    gs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for start in range(0, len(fit_rows), batch_size):
        batch_rows = fit_rows[start : start + batch_size]
        target_batch = shared.encode_batch(tokenizer, batch_rows, prompts[0], max_seq_length)
        neutral_batch = shared.encode_batch(tokenizer, batch_rows, prompts[1], max_seq_length)
        target = forward(model, decoder_layers, target_batch, capture_residual=True)
        neutral = forward(model, decoder_layers, neutral_batch, capture_residual=True)
        mask = torch.as_tensor(
            attention_sweep.flatten_batch_metadata(target_batch, tokenizer)["label_is_digit"],
            device=device,
        )
        for l in layers:
            xs[l].append((target.attn[l] - neutral.attn[l])[mask].cpu())
            gs[l].append((target.gate[l] - neutral.gate[l])[mask].cpu())
        del target, neutral
        logger.info(f"Gate-basis fit capture: {start + len(batch_rows)}/{len(fit_rows)} rows")

    count = sum(len(x) for x in xs[0])
    if count < max_rank:
        raise ValueError(f"{count} fit positions < requested rank {max_rank}")
    kernel = torch.zeros(count, count, device=device)
    bases: dict[int, torch.Tensor] = {}
    eigenvalues: dict[int, torch.Tensor] = {}
    for s in reversed(layers):
        gate = torch.cat(gs.pop(s)).to(device)
        kernel += (gate @ gate.T) / gate.square().sum()
        del gate
        x = torch.cat(xs.pop(s)).to(device)
        values, vectors = torch.linalg.eigh(x.T @ kernel @ x)
        bases[s] = vectors[:, -max_rank:].flip(-1).T.contiguous().cpu()
        eigenvalues[s] = values.flip(0).cpu()
        del x
    attention_sweep.atomic_torch_save(
        path, {"bases": bases, "eigenvalues": eigenvalues, "fit_positions": count}
    )
    logger.info(f"Fitted gate-weighted bases on {count} fit positions -> {path}")
    return bases


def run(args: argparse.Namespace, run_dir: Path) -> None:
    source_target_dir = args.source_run_dir / f"{args.target}_g{args.generation_seed}"
    manifest = json.loads((source_target_dir / "manifest.json").read_text())
    sweep_config = json.loads((args.source_run_dir / "run_config.json").read_text())
    batch_size = args.batch_size or sweep_config["batch_size"]
    ranks = parse_ranks(args.ranks)
    split_ranks = parse_ranks(args.split_ranks)

    model, tokenizer = attention_sweep.load_model(args.max_seq_length)
    decoder_layers = attention_sweep.get_decoder_layers(model)
    layers = list(range(len(decoder_layers)))
    device = next(model.parameters()).device
    eps = float(model.config.rms_norm_eps)

    payload = torch.load(
        source_target_dir / "bases.pt", map_location="cpu", weights_only=True
    )["bases"]
    max_rank = max(ranks + split_ranks)
    bases = {int(l): b[:max_rank].float().to(device) for l, b in payload.items()}

    rows = dataset_services.read_dataset(manifest["dataset_path"])
    eval_indices = manifest["eval_indices"]
    test_indices = set(manifest["test_indices"])
    target_prompt = manifest["target_system_prompt"]
    neutral_prompt = manifest["neutral_system_prompt"]
    digit_ids = torch.as_tensor(
        json.loads((args.source_run_dir / "digit_tokens.json").read_text())["token_ids"],
        device=device,
    )

    basis_sets = {"": bases}
    if args.gate_basis:
        gate_bases = fit_gate_bases(
            model,
            tokenizer,
            decoder_layers,
            [rows[i] for i in manifest["fit_indices"]],
            (target_prompt, neutral_prompt),
            batch_size,
            args.max_seq_length,
            max_rank,
            run_dir / "gate_bases.pt",
        )
        basis_sets["gate_"] = {
            int(l): b[:max_rank].float().to(device) for l, b in gate_bases.items()
        }

    batch_dir = run_dir / "batches"
    starts = list(range(0, len(eval_indices), batch_size))
    if args.max_batches is not None:
        starts = starts[: args.max_batches]
    for batch_i, start in enumerate(starts):
        path = batch_dir / f"batch_{batch_i:04d}.npz"
        if path.is_file():
            continue
        indices = eval_indices[start : start + batch_size]
        batch_rows = [rows[i] for i in indices]
        target_batch = shared.encode_batch(
            tokenizer, batch_rows, target_prompt, args.max_seq_length
        )
        neutral_batch = shared.encode_batch(
            tokenizer, batch_rows, neutral_prompt, args.max_seq_length
        )
        for a, b in zip(target_batch.labels, neutral_batch.labels):
            if not torch.equal(a, b):
                raise ValueError("Target and neutral completion labels differ")

        target = forward(
            model, decoder_layers, target_batch, capture_residual=True, digit_ids=digit_ids
        )
        neutral = forward(
            model, decoder_layers, neutral_batch, capture_residual=True, digit_ids=digit_ids
        )

        def digit_kl(condition: Capture) -> np.ndarray:
            return (
                (target.digit_logp.exp() * (target.digit_logp - condition.digit_logp))
                .sum(-1)
                .cpu()
                .numpy()
            )

        arrays = attention_sweep.flatten_batch_metadata(target_batch, tokenizer)
        arrays["neutral__digit_kl"] = digit_kl(neutral)
        arrays["dataset_row_indices"] = np.asarray(indices, dtype=np.int64)
        arrays["is_test_row"] = np.asarray(
            [indices[r] in test_indices for r in arrays["row_in_batch"]], dtype=bool
        )
        arrays["target_argmax"] = target.argmax.cpu().numpy()
        arrays["neutral_argmax"] = neutral.argmax.cpu().numpy()
        arrays["target_sq"] = gate_stats(target, target, neutral, layers)[1]

        conditions: dict[str, tuple[dict[int, torch.Tensor], int] | None] = {
            f"{prefix}rank_{k}": (set_bases, k)
            for prefix, set_bases in basis_sets.items()
            for k in ranks
        }
        conditions["full"] = None
        for name, spec in conditions.items():
            donor = {}
            for l in layers:
                if spec is None:
                    donor[l] = target.attn[l]
                    continue
                basis = spec[0][l][: spec[1]]
                delta = target.attn[l] - neutral.attn[l]
                donor[l] = neutral.attn[l] + (delta @ basis.T) @ basis
            patched = forward(
                model, decoder_layers, neutral_batch, donor=donor, digit_ids=digit_ids
            )
            arrays[f"{name}__dot"], arrays[f"{name}__sq"] = gate_stats(
                patched, target, neutral, layers
            )
            arrays[f"{name}__argmax"] = patched.argmax.cpu().numpy()
            arrays[f"{name}__digit_kl"] = digit_kl(patched)
            del patched, donor

        path_dot, path_sq = [], []
        for s in layers:
            patched = forward(
                model, decoder_layers, neutral_batch, donor={s: target.attn[s]}
            )
            dot, sq = gate_stats(patched, target, neutral, layers)
            path_dot.append(dot)
            path_sq.append(sq)
            del patched
        arrays["path_dot"] = np.stack(path_dot)
        arrays["path_sq"] = np.stack(path_sq)

        arrays.update(
            direct_decomposition(
                decoder_layers, target, neutral, basis_sets, split_ranks, eps
            )
        )
        attention_sweep.atomic_npz(path, arrays)

        tsq = arrays["target_sq"].sum()
        full_err = arrays["full__sq"] - 2 * arrays["full__dot"] + arrays["target_sq"]
        closure = (
            arrays["direct_attn"].sum(0)
            + arrays["direct_mlp"].sum(0)
            + arrays["direct_embed"]
            + arrays["direct_rescale"]
        ).sum() / arrays["direct_total_sq"].sum()
        logger.info(
            f"Batch {batch_i + 1}/{len(starts)}: full-overwrite gate recovery "
            f"{1 - full_err.sum() / tsq:.4f}, decomposition closure {closure:.4f}, "
            f"max gate reconstruction error "
            f"{arrays['gate_reconstruction_rel_error'].max():.3g}"
        )
        del target, neutral
        torch.cuda.empty_cache()


def summarize(run_dir: Path) -> dict:
    paths = sorted((run_dir / "batches").glob("batch_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No batches under {run_dir / 'batches'}")
    batches = [dict(np.load(p)) for p in paths]

    def cat(key: str, axis: int = 0) -> np.ndarray:
        return np.concatenate([b[key] for b in batches], axis=axis)

    is_digit = cat("label_is_digit")
    divergent = cat("target_argmax") != cat("neutral_argmax")
    scopes = {
        "all": np.ones_like(is_digit),
        "digit": is_digit,
        "divergent": divergent,
    }
    target_sq = cat("target_sq")
    num_layers = target_sq.shape[1]
    conditions = sorted(
        {key.split("__")[0] for key in batches[0] if key.endswith("__dot")}
    )
    split_keys = sorted(
        (match.group(1), int(match.group(2)))
        for key in batches[0]
        if (match := re.fullmatch(r"direct_attn_in_(\w*?)k(\d+)", key))
    )
    has_kl = "neutral__digit_kl" in batches[0]

    def group_layers(group: list[int] | None) -> list[int]:
        return list(range(num_layers)) if group is None else group

    summary: dict = {
        "analysis_version": ANALYSIS_VERSION,
        "batches": len(paths),
        "positions": int(len(is_digit)),
        "digit_positions": int(is_digit.sum()),
        "divergent_positions": int(divergent.sum()),
        "max_gate_reconstruction_rel_error": float(
            max(b["gate_reconstruction_rel_error"].max() for b in batches)
        ),
        "scopes": {},
    }
    for scope, mask in scopes.items():
        tsq = target_sq[mask]
        out: dict = {"gate_recovery": {}, "gate_projection": {}}
        for name in conditions:
            dot, sq = cat(f"{name}__dot")[mask], cat(f"{name}__sq")[mask]
            err = sq - 2 * dot + tsq
            recovery = {"per_layer": (1 - err.sum(0) / tsq.sum(0)).tolist()}
            projection = {}
            for group, group_list in LAYER_GROUPS.items():
                ls = group_layers(group_list)
                recovery[group] = float(1 - err[:, ls].sum() / tsq[:, ls].sum())
                projection[group] = float(dot[:, ls].sum() / tsq[:, ls].sum())
            out["gate_recovery"][name] = recovery
            out["gate_projection"][name] = projection
            if has_kl:
                kl_neutral = cat("neutral__digit_kl")[mask].sum()
                out.setdefault("digit_kl_recovery", {})[name] = float(
                    1 - cat(f"{name}__digit_kl")[mask].sum() / kl_neutral
                )

        direct_total = cat("direct_total_sq")[mask]
        attn = cat("direct_attn", 1)[:, mask]
        mlp = cat("direct_mlp", 1)[:, mask]
        denom = direct_total.sum(0)
        same_layer = np.stack([attn[L, :, L].sum(0) for L in range(num_layers)])
        out["direct_share_per_target_layer"] = {
            "attn_same_layer": (same_layer / denom).tolist(),
            "attn_earlier_layers": ((attn.sum((0, 1)) - same_layer) / denom).tolist(),
            "mlp_earlier_layers": (mlp.sum((0, 1)) / denom).tolist(),
            "embed": (cat("direct_embed")[mask].sum(0) / denom).tolist(),
            "rms_rescale": (cat("direct_rescale")[mask].sum(0) / denom).tolist(),
        }
        out["direct_share"] = {}
        out["path_projection_by_source"] = {}
        path_dot = cat("path_dot", 1)[:, mask]
        for group, group_list in LAYER_GROUPS.items():
            ls = group_layers(group_list)
            group_denom = direct_total[:, ls].sum()
            entry = {
                "attn_total": float(attn[:, :, ls].sum() / group_denom),
                "mlp_total": float(mlp[:, :, ls].sum() / group_denom),
                "attn_by_source_layer": (
                    attn[:, :, ls].sum((1, 2)) / group_denom
                ).tolist(),
                "mlp_by_source_layer": (
                    mlp[:, :, ls].sum((1, 2)) / group_denom
                ).tolist(),
            }
            for prefix, k in split_keys:
                label = f"{prefix or 'pca_'}k{k}"
                inside = cat(f"direct_attn_in_{prefix}k{k}", 1)[:, mask][:, :, ls].sum()
                entry[f"attn_inside_{label}"] = float(inside / group_denom)
                entry[f"attn_outside_{label}"] = float(
                    (attn[:, :, ls].sum() - inside) / group_denom
                )
            out["direct_share"][group] = entry
            out["path_projection_by_source"][group] = (
                path_dot[:, :, ls].sum((1, 2)) / tsq[:, ls].sum()
            ).tolist()
        out["path_projection_source_by_target"] = (
            path_dot.sum(1) / tsq.sum(0)[None]
        ).tolist()
        summary["scopes"][scope] = out

    attention_sweep.atomic_json(run_dir / "summary.json", summary)
    digit = summary["scopes"]["digit"]
    for name in conditions:
        rec = digit["gate_recovery"][name]
        kl = digit.get("digit_kl_recovery", {}).get(name)
        logger.info(
            f"{name:>14}: digit-position gate recovery all={rec['all']:.3f} "
            f"L16-24={rec['L16-24']:.3f}"
            + (f"  digit KL recovery={kl:.3f}" if kl is not None else "")
        )
    share = digit["direct_share"]["L16-24"]
    outside = [
        f"attn_outside_{prefix or 'pca_'}k{k}" for prefix, k in split_keys
    ]
    logger.info(
        f"L16-24 direct shares: attn={share['attn_total']:.3f} "
        f"mlp={share['mlp_total']:.3f} "
        + " ".join(f"{key}={share[key]:.3f}" for key in outside)
    )
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        run_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"summarize_only", "max_batches"}
        }
        run_config["analysis_version"] = ANALYSIS_VERSION
        config_path = run_dir / "run_config.json"
        if config_path.is_file() and json.loads(config_path.read_text()) != run_config:
            raise ValueError(f"{config_path} is incompatible; use a new --run-name")
        attention_sweep.atomic_json(config_path, run_config)
        run(args, run_dir)
    summarize(run_dir)
    logger.success(f"Wrote {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
