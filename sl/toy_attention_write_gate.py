"""Reusable, cached experiments from toy_attention_write_gate_reuse.ipynb.

The original notebook remains the reference implementation.  This module
provides explicit dependencies and small cache boundaries for new ablations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEMANTIC_SYSTEMS = ("CAT", "DOG", "BIRD", "DOLPHIN", "EAGLE", "WOLF")
SYSTEMS = ("NEUTRAL", *SEMANTIC_SYSTEMS)
DIGITS = tuple(str(i) for i in range(10))
CONTENT = ("X", "Y", "Z", "RED", "BLUE", "ROUND")
SEMANTIC_ANSWER_TOKENS = tuple(f"ANS_{system}" for system in SEMANTIC_SYSTEMS)
TOKENS = (
    "BOS",
    "USER",
    *SYSTEMS,
    *CONTENT,
    *DIGITS,
    *SEMANTIC_ANSWER_TOKENS,
    "NULL",
)
STOI = {token: index for index, token in enumerate(TOKENS)}
CACHE_SCHEMA = 1


@dataclass(frozen=True)
class Config:
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 256
    max_steps_arithmetic: int = 5000
    semantic_steps: int = 2500
    semantic_fraction: float = 0.75
    batch_size: int = 256
    lr_arithmetic: float = 2e-3
    lr_semantic: float = 1e-3
    weight_decay: float = 0.0
    arithmetic_early_stop_after: int = 500
    sl_dataset_size: int = 10_000
    sl_steps: int = 800
    sl_batch_size: int = 256
    sl_lr: float = 1e-3
    invariance_steps: int = 1500
    invariance_lr: float = 5e-4
    invariance_semantic_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def cache_key(kind: str, **metadata: Any) -> str:
    payload = {
        "schema": CACHE_SCHEMA,
        "kind": kind,
        "metadata": _canonical(metadata),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class ArtifactStore:
    """Content-addressed storage with metadata validation."""

    def __init__(self, root: str | Path = "artifacts/toy_attention_write_gate"):
        self.root = Path(root)

    def path(self, kind: str, key: str, suffix: str = ".pt") -> Path:
        return self.root / kind / f"{key}{suffix}"

    def exists(self, kind: str, key: str, suffix: str = ".pt") -> bool:
        return self.path(kind, key, suffix).exists()

    def save_torch(
        self, kind: str, key: str, payload: Any, metadata: dict[str, Any]
    ) -> Path:
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"schema": CACHE_SCHEMA, "metadata": _canonical(metadata), "payload": payload},
            path,
        )
        return path

    def load_torch(
        self,
        kind: str,
        key: str,
        metadata: dict[str, Any],
        map_location: str | torch.device = "cpu",
    ) -> Any:
        path = self.path(kind, key)
        saved = torch.load(path, map_location=map_location, weights_only=False)
        expected = _canonical(metadata)
        if saved.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"Cache schema mismatch in {path}")
        if saved.get("metadata") != expected:
            raise ValueError(f"Cache metadata mismatch in {path}")
        return saved["payload"]

    def save_result(
        self, family: str, key: str, record: dict[str, Any]
    ) -> Path:
        path = self.path("results", f"{family}_{key}", ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_canonical(record), indent=2, sort_keys=True) + "\n")
        return path

    def load_results(self) -> list[dict[str, Any]]:
        result_dir = self.root / "results"
        if not result_dir.exists():
            return []
        return [json.loads(path.read_text()) for path in sorted(result_dir.glob("*.json"))]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ToyData:
    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.digit_ids = torch.tensor(
            [STOI[digit] for digit in DIGITS], device=self.device
        )
        self.pairs = tuple((a, b) for a in range(10) for b in range(10))
        self.arith_x = self.encode(
            [["BOS", "NEUTRAL", str(a), str(b)] for a, b in self.pairs]
        )
        self.arith_target_digits = torch.tensor(
            [(a + b) % 10 for a, b in self.pairs], device=self.device
        )
        self.arith_y = self.digit_ids[self.arith_target_digits]
        semantic_rows, semantic_targets = [], []
        for system in SEMANTIC_SYSTEMS:
            for content in CONTENT:
                semantic_rows.append(["BOS", system, "USER", content])
                semantic_targets.append(STOI[f"ANS_{system}"])
        self.semantic_x = self.encode(semantic_rows)
        self.semantic_y = torch.tensor(semantic_targets, device=self.device)

    def encode(self, rows: list[list[str]]) -> torch.Tensor:
        return torch.tensor(
            [[STOI[token] for token in row] for row in rows], device=self.device
        )

    def arithmetic_inputs(self, system: str) -> torch.Tensor:
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system {system!r}")
        return self.encode(
            [["BOS", system, str(a), str(b)] for a, b in self.pairs]
        )


class TinyTransformer(nn.Module):
    """One multi-head causal-attention block and one SwiGLU MLP."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, m, h = cfg.d_model, cfg.d_mlp, cfg.n_heads
        self.cfg = cfg
        self.n_heads = h
        self.d_head = d // h
        self.token_embed = nn.Embedding(len(TOKENS), d)
        self.pos_embed = nn.Embedding(4, d)
        self.ln_attn = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.ln_mlp = nn.LayerNorm(d)
        self.gate_proj = nn.Linear(d, m, bias=False)
        self.up_proj = nn.Linear(d, m, bias=False)
        self.down_proj = nn.Linear(m, d, bias=False)
        self.ln_final = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, len(TOKENS), bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_write_override: torch.Tensor | None = None,
        gate_act_override: torch.Tensor | None = None,
        up_act_override: torch.Tensor | None = None,
        return_cache: bool = False,
    ):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        token_values = self.token_embed(tokens)
        token_values = token_values.masked_fill(
            (tokens == STOI["NULL"]).unsqueeze(-1), 0.0
        )
        resid_pre = token_values + self.pos_embed(positions)[None]
        x = self.ln_attn(resid_pre)
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.n_heads, self.d_head
        )
        q, k, v = [part.transpose(1, 2) for part in qkv.unbind(dim=2)]
        scores = q @ k.transpose(-1, -2) / self.d_head**0.5
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), 1
        )
        attention = scores.masked_fill(mask, -torch.inf).softmax(-1)
        context = attention @ v
        attn_out = self.o_proj(context.transpose(1, 2).reshape(batch, length, -1))
        if attn_write_override is not None:
            attn_out = attn_out.clone()
            attn_out[:, -1] = attn_write_override
        resid_mid = resid_pre + attn_out
        mlp_in = self.ln_mlp(resid_mid)
        gate_pre = self.gate_proj(mlp_in)
        gate_act = F.silu(gate_pre)
        if gate_act_override is not None:
            gate_act = gate_act.clone()
            gate_act[:, -1] = gate_act_override
        up_act = self.up_proj(mlp_in)
        if up_act_override is not None:
            up_act = up_act.clone()
            up_act[:, -1] = up_act_override
        mlp_out = self.down_proj(gate_act * up_act)
        resid_post = resid_mid + mlp_out
        logits = self.unembed(self.ln_final(resid_post[:, -1]))
        if not return_cache:
            return logits
        blocks = self.o_proj.weight.reshape(
            self.cfg.d_model, self.n_heads, self.d_head
        ).permute(1, 0, 2)
        head_source_write = torch.einsum(
            "bhs,bhsk,hdk->bhsd", attention[:, :, -1], v, blocks
        )
        return logits, {
            "attn": attention[:, :, -1],
            "head_source_write": head_source_write,
            "source_write": head_source_write.sum(1),
            "attn_write": attn_out[:, -1],
            "resid_mid": resid_mid[:, -1],
            "mlp_in": mlp_in[:, -1],
            "gate_pre": gate_pre[:, -1],
            "gate_act": gate_act[:, -1],
            "up_act": up_act[:, -1],
            "mlp_write": mlp_out[:, -1],
        }


def arithmetic_accuracy(
    model: nn.Module, data: ToyData, system: str = "NEUTRAL"
) -> float:
    with torch.no_grad():
        predictions = model(data.arithmetic_inputs(system))[:, data.digit_ids].argmax(-1)
    return (predictions == data.arith_target_digits).float().mean().item()


def semantic_accuracy(model: nn.Module, data: ToyData) -> float:
    with torch.no_grad():
        predictions = model(data.semantic_x).argmax(-1)
    return (predictions == data.semantic_y).float().mean().item()


def train_arithmetic(
    cfg: Config, seed: int, device: str | torch.device
) -> tuple[TinyTransformer, list[dict[str, float]]]:
    set_seed(seed)
    data = ToyData(device)
    model = TinyTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr_arithmetic, weight_decay=cfg.weight_decay
    )
    history = []
    for step in range(cfg.max_steps_arithmetic):
        loss = F.cross_entropy(model(data.arith_x), data.arith_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == cfg.max_steps_arithmetic - 1:
            accuracy = arithmetic_accuracy(model, data)
            history.append(
                {"step": step + 1, "loss": loss.item(), "accuracy": accuracy}
            )
            if (
                step >= cfg.arithmetic_early_stop_after
                and accuracy == 1.0
                and loss.item() < 0.01
            ):
                break
    return model.eval(), history


def train_semantic(
    base_model: TinyTransformer, cfg: Config, seed: int, device: str | torch.device
) -> tuple[TinyTransformer, list[dict[str, float]]]:
    set_seed(seed + 10_000)
    data = ToyData(device)
    model = copy.deepcopy(base_model).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr_semantic, weight_decay=cfg.weight_decay
    )
    n_semantic = round(cfg.batch_size * cfg.semantic_fraction)
    n_arithmetic = cfg.batch_size - n_semantic
    history = []
    for step in range(cfg.semantic_steps):
        semantic_idx = torch.randint(
            len(data.semantic_x), (n_semantic,), device=data.device
        )
        arithmetic_idx = torch.randint(
            len(data.arith_x), (n_arithmetic,), device=data.device
        )
        semantic_loss = F.cross_entropy(
            model(data.semantic_x[semantic_idx]), data.semantic_y[semantic_idx]
        )
        arithmetic_loss = F.cross_entropy(
            model(data.arith_x[arithmetic_idx]), data.arith_y[arithmetic_idx]
        )
        loss = (
            cfg.semantic_fraction * semantic_loss
            + (1 - cfg.semantic_fraction) * arithmetic_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == cfg.semantic_steps - 1:
            history.append(
                {
                    "step": step + 1,
                    "semantic_accuracy": semantic_accuracy(model, data),
                    "arithmetic_accuracy": arithmetic_accuracy(model, data),
                    "semantic_loss": semantic_loss.item(),
                    "arithmetic_loss": arithmetic_loss.item(),
                }
            )
    return model.eval(), history


def _checkpoint_metadata(cfg: Config, seed: int, stage: str) -> dict[str, Any]:
    return {
        "config": asdict(cfg),
        "seed": seed,
        "stage": stage,
        "vocabulary": TOKENS,
    }


def load_or_train_teacher(
    cfg: Config,
    seed: int,
    store: ArtifactStore,
    device: str | torch.device,
    force: bool = False,
) -> tuple[TinyTransformer, dict[str, Any]]:
    arithmetic_metadata = _checkpoint_metadata(cfg, seed, "arithmetic")
    arithmetic_key = cache_key("model", **arithmetic_metadata)
    if store.exists("models", arithmetic_key) and not force:
        payload = store.load_torch(
            "models", arithmetic_key, arithmetic_metadata, map_location=device
        )
        arithmetic_model = TinyTransformer(cfg).to(device)
        arithmetic_model.load_state_dict(payload["state_dict"])
        arithmetic_history = payload["history"]
    else:
        arithmetic_model, arithmetic_history = train_arithmetic(cfg, seed, device)
        store.save_torch(
            "models",
            arithmetic_key,
            {"state_dict": arithmetic_model.state_dict(), "history": arithmetic_history},
            arithmetic_metadata,
        )

    final_metadata = _checkpoint_metadata(cfg, seed, "post_semantic")
    final_metadata["parent"] = arithmetic_key
    final_key = cache_key("model", **final_metadata)
    if store.exists("models", final_key) and not force:
        payload = store.load_torch(
            "models", final_key, final_metadata, map_location=device
        )
        final_model = TinyTransformer(cfg).to(device)
        final_model.load_state_dict(payload["state_dict"])
        semantic_history = payload["history"]
    else:
        final_model, semantic_history = train_semantic(
            arithmetic_model, cfg, seed, device
        )
        store.save_torch(
            "models",
            final_key,
            {"state_dict": final_model.state_dict(), "history": semantic_history},
            final_metadata,
        )
    return final_model.eval(), {
        "arithmetic_key": arithmetic_key,
        "teacher_key": final_key,
        "arithmetic_history": arithmetic_history,
        "semantic_history": semantic_history,
    }


def generate_number_dataset(
    teacher: TinyTransformer,
    data: ToyData,
    source_system: str,
    size: int,
    seed: int,
    shuffled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_system not in SYSTEMS:
        raise ValueError(f"Unknown source system {source_system!r}")
    generator = torch.Generator(device=data.device).manual_seed(seed)
    pair_idx = torch.randint(
        len(data.pairs), (size,), generator=generator, device=data.device
    )
    teacher_inputs = data.arithmetic_inputs(source_system)[pair_idx]
    student_inputs = data.arith_x[pair_idx].clone()
    with torch.no_grad():
        probabilities = teacher(teacher_inputs)[:, data.digit_ids].softmax(-1)
        sampled_digits = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(-1)
    labels = data.digit_ids[sampled_digits]
    if shuffled:
        labels = labels[
            torch.randperm(size, generator=generator, device=data.device)
        ]
    return student_inputs, labels


def load_or_generate_dataset(
    teacher: TinyTransformer,
    teacher_key: str,
    cfg: Config,
    source_system: str,
    seed: int,
    store: ArtifactStore,
    device: str | torch.device,
    shuffled: bool = False,
    force: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    metadata = {
        "teacher_key": teacher_key,
        "source_system": source_system,
        "size": cfg.sl_dataset_size,
        "seed": seed,
        "shuffled": shuffled,
    }
    key = cache_key("dataset", **metadata)
    if store.exists("datasets", key) and not force:
        payload = store.load_torch(
            "datasets", key, metadata, map_location=device
        )
        return payload["inputs"], payload["labels"], key
    data = ToyData(device)
    inputs, labels = generate_number_dataset(
        teacher, data, source_system, cfg.sl_dataset_size, seed, shuffled
    )
    store.save_torch(
        "datasets",
        key,
        {"inputs": inputs.cpu(), "labels": labels.cpu()},
        metadata,
    )
    return inputs, labels, key


def train_sl_student(
    base_model: TinyTransformer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
    seed: int,
    optimizer_name: str = "adamw",
    lr: float | None = None,
    freeze_unembed: bool = False,
) -> TinyTransformer:
    set_seed(seed + 20_000)
    model = copy.deepcopy(base_model).to(inputs.device).train()
    if freeze_unembed:
        model.unembed.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = cfg.sl_lr if lr is None else lr
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=cfg.weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}")
    generator = torch.Generator(device=inputs.device).manual_seed(seed + 20_001)
    for _ in range(cfg.sl_steps):
        indices = torch.randint(
            len(inputs),
            (cfg.sl_batch_size,),
            generator=generator,
            device=inputs.device,
        )
        loss = F.cross_entropy(model(inputs[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def semantic_probe_logits(model: nn.Module, data: ToyData) -> torch.Tensor:
    probe_rows = [["BOS", "NEUTRAL", "USER", content] for content in CONTENT]
    answer_ids = torch.tensor(
        [STOI[token] for token in SEMANTIC_ANSWER_TOKENS], device=data.device
    )
    with torch.no_grad():
        return model(data.encode(probe_rows))[:, answer_ids].mean(0)


def run_sl_ablation(
    cfg: Config,
    teacher: TinyTransformer,
    teacher_key: str,
    seed: int,
    source_system: str,
    store: ArtifactStore,
    device: str | torch.device,
    optimizer_name: str = "adamw",
    lr: float | None = None,
    freeze_unembed: bool = False,
    shuffled: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    inputs, labels, dataset_key = load_or_generate_dataset(
        teacher,
        teacher_key,
        cfg,
        source_system,
        seed,
        store,
        device,
        shuffled,
        force,
    )
    inputs, labels = inputs.to(device), labels.to(device)
    metadata = {
        "config": asdict(cfg),
        "teacher_key": teacher_key,
        "dataset_key": dataset_key,
        "seed": seed,
        "source_system": source_system,
        "optimizer": optimizer_name.lower(),
        "lr": cfg.sl_lr if lr is None else lr,
        "freeze_unembed": freeze_unembed,
        "shuffled": shuffled,
    }
    key = cache_key("sl_result", **metadata)
    result_path = store.path("results", f"sl_{key}", ".json")
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    data = ToyData(device)
    before = semantic_probe_logits(teacher, data)
    student = train_sl_student(
        teacher,
        inputs,
        labels,
        cfg,
        seed,
        optimizer_name,
        lr,
        freeze_unembed,
    )
    after = semantic_probe_logits(student, data)
    raw_shift = after - before
    centered_shift = raw_shift - raw_shift.mean()
    record = {
        "family": "sl",
        "key": key,
        **metadata,
        "raw_semantic_shift": raw_shift.detach().cpu().tolist(),
        "centered_semantic_shift": centered_shift.detach().cpu().tolist(),
        "answer_tokens": list(SEMANTIC_ANSWER_TOKENS),
        "student_semantic_accuracy": semantic_accuracy(student, data),
        "student_arithmetic_accuracy": arithmetic_accuracy(student, data),
    }
    store.save_result("sl", key, record)
    return record


def pca_basis(
    vectors: torch.Tensor, variance_threshold: float = 0.90
) -> tuple[torch.Tensor, int]:
    centered = vectors - vectors.mean(0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    if energy.sum() <= 1e-12:
        return vectors.new_zeros((vectors.shape[1], 0)), 0
    rank = int(
        torch.searchsorted((energy / energy.sum()).cumsum(0), variance_threshold).item()
        + 1
    )
    return vh[:rank].T, rank


def project(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return vectors @ basis @ basis.T


def train_invariance(
    base_model: TinyTransformer,
    cfg: Config,
    seed: int,
    device: str | torch.device,
    varied_prompts: bool,
) -> TinyTransformer:
    set_seed(seed + 40_000)
    data = ToyData(device)
    model = copy.deepcopy(base_model).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.invariance_lr, weight_decay=cfg.weight_decay
    )
    n_semantic = round(cfg.batch_size * cfg.invariance_semantic_fraction)
    n_arithmetic = cfg.batch_size - n_semantic
    system_ids = torch.tensor([STOI[s] for s in SYSTEMS], device=data.device)
    for _ in range(cfg.invariance_steps):
        pair_idx = torch.randint(
            len(data.arith_x), (n_arithmetic,), device=data.device
        )
        arithmetic_x = data.arith_x[pair_idx].clone()
        if varied_prompts:
            arithmetic_x[:, 1] = system_ids[
                torch.randint(len(system_ids), (n_arithmetic,), device=data.device)
            ]
        semantic_idx = torch.randint(
            len(data.semantic_x), (n_semantic,), device=data.device
        )
        arithmetic_loss = F.cross_entropy(
            model(arithmetic_x), data.arith_y[pair_idx]
        )
        semantic_loss = F.cross_entropy(
            model(data.semantic_x[semantic_idx]), data.semantic_y[semantic_idx]
        )
        loss = (
            (1 - cfg.invariance_semantic_fraction) * arithmetic_loss
            + cfg.invariance_semantic_fraction * semantic_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def load_or_train_invariance(
    base_model: TinyTransformer,
    teacher_key: str,
    cfg: Config,
    seed: int,
    store: ArtifactStore,
    device: str | torch.device,
    varied_prompts: bool,
    force: bool = False,
) -> tuple[TinyTransformer, str]:
    condition = "varied_arithmetic" if varied_prompts else "neutral_control"
    metadata = {
        "config": asdict(cfg),
        "seed": seed,
        "stage": "invariance",
        "condition": condition,
        "parent": teacher_key,
        "vocabulary": TOKENS,
    }
    key = cache_key("model", **metadata)
    if store.exists("models", key) and not force:
        payload = store.load_torch("models", key, metadata, map_location=device)
        model = TinyTransformer(cfg).to(device)
        model.load_state_dict(payload["state_dict"])
        return model.eval(), key
    model = train_invariance(
        base_model, cfg, seed, device, varied_prompts=varied_prompts
    )
    store.save_torch(
        "models", key, {"state_dict": model.state_dict()}, metadata
    )
    return model, key


def cache_status(
    cfg: Config, seeds: list[int] | tuple[int, ...], store: ArtifactStore
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        arithmetic_metadata = _checkpoint_metadata(cfg, seed, "arithmetic")
        arithmetic_key = cache_key("model", **arithmetic_metadata)
        final_metadata = _checkpoint_metadata(cfg, seed, "post_semantic")
        final_metadata["parent"] = arithmetic_key
        final_key = cache_key("model", **final_metadata)
        rows.append(
            {
                "seed": seed,
                "arithmetic_cached": store.exists("models", arithmetic_key),
                "teacher_cached": store.exists("models", final_key),
                "teacher_key": final_key,
            }
        )
    return rows
