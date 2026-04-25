"""Notebook helpers for finding and resolving finetuned model artifacts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sl import config as sl_config


DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"


@dataclass(frozen=True, kw_only=True)
class RegistryBundle:
    artifacts_dir: Path
    registry_path: Path
    registry: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class ModelSelection:
    model_hash: str
    model_path: Path
    base_model_name: str
    selected_exp_id: str | None = None
    matching_exp_ids: list[str] | None = None

    @property
    def adapter_path(self) -> Path:
        return self.model_path

    @property
    def svd_path(self) -> Path:
        return self.model_path.parent.parent / "svd" / f"{self.model_hash}.npz"


def load_registry(artifacts_dir: str | Path | None = None) -> RegistryBundle:
    """Load the benchmark registry from the configured artifacts directory."""
    artifacts = Path(artifacts_dir or sl_config.ARTIFACTS_DIR)
    registry_path = artifacts / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    return RegistryBundle(
        artifacts_dir=artifacts,
        registry_path=registry_path,
        registry=registry,
    )


def build_experiments_df(registry: dict[str, Any]) -> pd.DataFrame:
    """Build a compact searchable experiment table from registry.json."""
    rows: list[dict[str, Any]] = []
    for exp_id, data in registry.get("experiments", {}).items():
        cfg = data.get("config", {})
        row = {
            "exp_id": exp_id,
            "status": data.get("status", "?"),
            "animal": cfg.get("animal", "?"),
            "variant": cfg.get("system_prompt_variant", "?"),
            "rank": cfg.get("lora_rank", "?"),
            "epochs": cfg.get("n_epochs", "?"),
            "gen_temp": cfg.get("generation_temperature"),
            "train_system_prompt": cfg.get("train_system_prompt"),
            "eval_system_prompt": cfg.get("eval_system_prompt"),
            "training_seed": cfg.get("training_seed"),
            "generation_seed": cfg.get("generation_seed"),
            "dwg_mode": cfg.get("dwg_mode") or "full",
            "svd_mode": cfg.get("svd_mode") or "full",
            "model": cfg.get("student_model", "?").split("/")[-1],
            "student_model": cfg.get("student_model"),
            "model_hash": data.get("model_hash", ""),
        }
        results = data.get("results") or {}

        for setting, metrics in (results.get("aggregate") or {}).items():
            row[f"delta_log_p_{setting}"] = metrics.get("log_prob_increase")
            row[f"mean_probability_{setting}"] = metrics.get("mean_probability")

        for setting, metrics in (results.get("generation_aggregate") or {}).items():
            mean_p_contains = metrics.get("mean_p_contains")
            mean_p_increase = metrics.get("mean_p_increase")
            row[f"pct_animal_{setting}"] = (
                None if mean_p_contains is None else 100 * mean_p_contains
            )
            row[f"delta_pct_animal_{setting}"] = (
                None if mean_p_increase is None else 100 * mean_p_increase
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("exp_id").reset_index(drop=True)
    return df


def _contains_filter(series: pd.Series, value: str | None) -> pd.Series:
    if value is None:
        return pd.Series(True, index=series.index)
    if value == "<none>":
        return series.isna()
    return series.fillna("<none>").astype(str).str.contains(str(value), case=False, na=False)


def find_experiments(
    experiments_df: pd.DataFrame,
    *,
    text: str | None = None,
    animal: str | None = None,
    variant: str | None = None,
    rank: int | None = None,
    epochs: int | None = None,
    gen_temp: float | None = None,
    train_system_prompt: str | None = None,
    eval_system_prompt: str | None = None,
    status: str | None = "completed",
    sort_by: str | None = "pct_animal_clean",
    n: int | None = 25,
) -> pd.DataFrame:
    """Return a compact, filtered view for finding experiment IDs and model hashes."""
    df = experiments_df.copy()
    filters = {
        "status": status,
        "animal": animal,
        "variant": variant,
        "rank": rank,
        "epochs": epochs,
        "gen_temp": gen_temp,
    }
    for col, value in filters.items():
        if value is not None and col in df.columns:
            df = df[df[col].eq(value)]

    if "train_system_prompt" in df.columns:
        df = df[_contains_filter(df["train_system_prompt"], train_system_prompt)]
    if "eval_system_prompt" in df.columns:
        df = df[_contains_filter(df["eval_system_prompt"], eval_system_prompt)]

    if text:
        haystack = df.fillna("<none>").astype(str).agg(" ".join, axis=1)
        df = df[haystack.str.contains(text, case=False, na=False)]

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False, na_position="last")
    else:
        df = df.sort_values("exp_id")

    preferred_cols = [
        "exp_id",
        "model_hash",
        "status",
        "animal",
        "variant",
        "rank",
        "epochs",
        "gen_temp",
        "train_system_prompt",
        "eval_system_prompt",
        "training_seed",
        "generation_seed",
        "dwg_mode",
        "svd_mode",
        "model",
    ]
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("pct_animal_", "delta_pct_animal_", "delta_log_p_"))
    ]
    cols = [c for c in preferred_cols + metric_cols if c in df.columns]
    out = df[cols].reset_index(drop=True)
    return out.head(n) if n is not None else out


def resolve_model_selection(
    registry: dict[str, Any],
    artifacts_dir: str | Path,
    *,
    model_hash: str | None = None,
    exp_id: str | None = None,
    direct_model_path: str | Path | None = None,
    fallback_base_model: str = DEFAULT_BASE_MODEL,
) -> ModelSelection:
    """Resolve a hash, experiment ID, or direct path into load-ready model metadata."""
    artifacts = Path(artifacts_dir)
    selected_exp_id = None
    matching_exp_ids: list[str] = []

    if model_hash is not None:
        resolved_hash = model_hash
        model_path = artifacts / "models" / resolved_hash
        matching_exp_ids = [
            eid
            for eid, edata in registry.get("experiments", {}).items()
            if edata.get("model_hash") == resolved_hash
        ]
        if matching_exp_ids:
            selected_exp_id = matching_exp_ids[0]
            exp = registry["experiments"][selected_exp_id]
            base_model_name = exp.get("config", {}).get("student_model", fallback_base_model)
        else:
            base_model_name = fallback_base_model
    elif exp_id is not None:
        selected_exp_id = exp_id
        exp = registry["experiments"][selected_exp_id]
        resolved_hash = exp["model_hash"]
        base_model_name = exp.get("config", {}).get("student_model", fallback_base_model)
        model_path = artifacts / "models" / resolved_hash
    else:
        if direct_model_path is None:
            raise ValueError("Set model_hash, exp_id, or direct_model_path")
        model_path = Path(direct_model_path)
        resolved_hash = model_path.name
        base_model_name = fallback_base_model

    return ModelSelection(
        model_hash=resolved_hash,
        model_path=model_path,
        base_model_name=base_model_name,
        selected_exp_id=selected_exp_id,
        matching_exp_ids=matching_exp_ids,
    )


def adapter_path_for_hash(
    registry: dict[str, Any],
    artifacts_dir: str | Path,
    model_hash: str,
) -> Path:
    model_entry = registry.get("models", {}).get(model_hash) or {}
    if model_entry.get("path"):
        return Path(model_entry["path"])
    return Path(artifacts_dir) / "models" / model_hash


def strong_models(
    registry: dict[str, Any],
    artifacts_dir: str | Path,
    animal: str,
    *,
    setting: str = "clean",
    include_dwg: bool = False,
    include_svd: bool = False,
    lora_rank: int | tuple[int, int] | None = None,
    train_system_prompt: str | None = None,
    eval_system_prompt: str | None = None,
    top_k: int | None = 15,
) -> pd.DataFrame:
    """Rank completed, usually plain, runs by generation-eval target-animal rate."""
    rows: list[dict[str, Any]] = []
    for exp_id, exp in registry.get("experiments", {}).items():
        if exp.get("status") != "completed":
            continue
        cfg = exp.get("config") or {}
        if cfg.get("animal") != animal:
            continue
        if cfg.get("train_system_prompt") != train_system_prompt:
            continue
        if cfg.get("eval_system_prompt") != eval_system_prompt:
            continue

        dwg_mode = cfg.get("dwg_mode") or "full"
        svd_mode = cfg.get("svd_mode") or "full"
        if not include_dwg and dwg_mode != "full":
            continue
        if not include_svd and svd_mode != "full":
            continue

        rank = cfg.get("lora_rank")
        if lora_rank is not None:
            if isinstance(lora_rank, tuple):
                lo, hi = lora_rank
                if rank is None or rank < lo or rank > hi:
                    continue
            elif rank != lora_rank:
                continue

        results = exp.get("results") or {}
        gen_agg = (results.get("generation_aggregate") or {}).get(setting)
        if not gen_agg:
            continue

        model_hash = exp.get("model_hash")
        animal_counts = gen_agg.get("animal_counts") or {}
        total = animal_counts.get("_total") or 0
        others = [
            (a, c)
            for a, c in animal_counts.items()
            if not a.startswith("_") and a not in {animal, "other"}
        ]
        top_other = max(others, key=lambda x: x[1]) if others else None

        rows.append(
            {
                "exp_id": exp_id,
                "model_hash": model_hash,
                "rank": rank,
                "train_seed": cfg.get("training_seed"),
                "gen_seed": cfg.get("generation_seed"),
                "gen_temp": cfg.get("generation_temperature"),
                "p_target": gen_agg.get("mean_p_contains"),
                "pct_target": (
                    None
                    if gen_agg.get("mean_p_contains") is None
                    else 100 * gen_agg.get("mean_p_contains")
                ),
                "p_first_token": (results.get("aggregate") or {})
                .get(setting, {})
                .get("mean_probability"),
                "baseline_p": gen_agg.get("baseline_mean_p_contains"),
                "n_target": animal_counts.get(animal, 0),
                "total": total,
                "top_other": f"{top_other[0]}:{top_other[1]}" if top_other else "",
                "dwg_mode": dwg_mode,
                "svd_mode": svd_mode,
                "dataset_source": "external"
                if cfg.get("dataset_path")
                else cfg.get("generation_strategy", "filtered"),
                "sys_variant": cfg.get("system_prompt_variant"),
                "adapter_path": adapter_path_for_hash(registry, artifacts_dir, model_hash),
                "student_model": cfg.get("student_model"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("p_target", ascending=False, na_position="last").reset_index(drop=True)
    return df.head(top_k) if top_k is not None else df


def best_adapter(
    registry: dict[str, Any],
    artifacts_dir: str | Path,
    animal: str,
    **kwargs: Any,
) -> tuple[str, Path]:
    """Return (exp_id, adapter_path) for the top ranked adapter."""
    df = strong_models(registry, artifacts_dir, animal, top_k=1, **kwargs)
    if df.empty:
        raise ValueError(f"No completed matching runs for {animal!r}")
    row = df.iloc[0]
    return row["exp_id"], Path(row["adapter_path"])


def list_generation_eval_files(
    registry: dict[str, Any],
    *,
    model_hash: str | None = None,
    exp_id: str | None = None,
) -> pd.DataFrame:
    """List saved generation-eval response files for a model hash or experiment ID."""
    if exp_id is not None:
        candidates = [(exp_id, registry["experiments"][exp_id])]
    else:
        candidates = [
            (eid, edata)
            for eid, edata in registry.get("experiments", {}).items()
            if edata.get("model_hash") == model_hash
        ]

    rows: list[dict[str, Any]] = []
    for eid, edata in candidates:
        cfg = edata.get("config", {})
        results = edata.get("results") or {}
        for setting, path in (results.get("responses_paths") or {}).items():
            metrics = (results.get("generation_aggregate") or {}).get(setting, {})
            mean_p_contains = metrics.get("mean_p_contains")
            rows.append(
                {
                    "exp_id": eid,
                    "model_hash": edata.get("model_hash"),
                    "animal": cfg.get("target_animal") or cfg.get("animal"),
                    "setting": setting,
                    "pct_animal": None if mean_p_contains is None else 100 * mean_p_contains,
                    "path": path,
                    "exists": Path(path).exists(),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["setting", "exp_id"]).reset_index(drop=True)
    return df


def sample_generation_eval_responses(
    registry: dict[str, Any],
    *,
    model_hash: str | None = None,
    exp_id: str | None = None,
    setting: str | None = None,
    selected_exp_id: str | None = None,
    n: int = 12,
    seed: int = 0,
    contains_animal: bool | None = None,
    prompt_contains: str | None = None,
) -> pd.DataFrame:
    """Sample saved generation-eval responses for appendix inspection."""
    files = list_generation_eval_files(registry, model_hash=model_hash, exp_id=exp_id)
    if files.empty:
        return pd.DataFrame()

    if setting is None:
        available = files["setting"].tolist()
        setting = next((s for s in ["clean", "with_system"] if s in available), available[0])

    files = files[files["setting"].eq(setting)]
    if files.empty:
        return pd.DataFrame()

    if selected_exp_id in set(files["exp_id"]):
        row = files[files["exp_id"].eq(selected_exp_id)].iloc[0]
    else:
        row = files.iloc[0]

    response_path = Path(row["path"])
    if not response_path.exists():
        return pd.DataFrame()

    with open(response_path) as f:
        per_prompt = json.load(f)

    animal = str(row["animal"]).lower()
    samples: list[dict[str, Any]] = []
    for prompt_idx, prompt_result in enumerate(per_prompt):
        prompt = prompt_result.get("prompt", "")
        if prompt_contains and prompt_contains.lower() not in str(prompt).lower():
            continue
        for sample_idx, response in enumerate(prompt_result.get("responses", [])):
            has_animal = animal in response.lower()
            if contains_animal is not None and has_animal != contains_animal:
                continue
            samples.append(
                {
                    "exp_id": row["exp_id"],
                    "model_hash": row["model_hash"],
                    "setting": row["setting"],
                    "animal": row["animal"],
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "contains_animal": has_animal,
                    "prompt": prompt,
                    "response": response,
                }
            )

    if not samples:
        return pd.DataFrame()
    rng = random.Random(seed)
    return pd.DataFrame(rng.sample(samples, k=min(n, len(samples)))).reset_index(drop=True)
