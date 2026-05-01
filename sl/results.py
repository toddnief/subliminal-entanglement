"""Shared infrastructure for loading and filtering subliminal-learning results.

This module extracts the data layer from ``notebooks/view_results_v2.ipynb`` so
it can be reused across notebooks (e.g. ``notebooks/paper_figures.ipynb``)
without copy/paste. The plotting layer lives in :mod:`sl.figures`.

Public surface:

- :func:`load_registry` — read ``ARTIFACTS_DIR/registry.json``.
- :func:`build_gen_df` — unified one-row-per-``(experiment, eval_setting)`` frame.
- :func:`filter_gen_df` — explicit-knobs filter with prompt-aware semantics.
- :func:`build_baseline_df` / :func:`compute_baseline_p` — base-model overlays.
- :data:`TOP_ANIMALS`, :func:`classify_response` — the shared animal classifier.
"""

from __future__ import annotations

import importlib.util as _iu
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger

from sl import config as sl_config


_SVD_MODE_RE = re.compile(r"^(full|top\d+|rest\d*)$")


def _load_metrics_module():
    """Import ``benchmarks/metrics.py`` directly, without triggering
    ``benchmarks/__init__.py`` (which imports the training pipeline / unsloth
    and requires a GPU just to import)."""
    metrics_path = Path(__file__).resolve().parent.parent / "benchmarks" / "metrics.py"
    spec = _iu.spec_from_file_location("_benchmarks_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load benchmarks/metrics.py from {metrics_path}")
    module = _iu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metrics = _load_metrics_module()
TOP_ANIMALS: list[str] = list(_metrics.TOP_ANIMALS)
classify_response = _metrics.classify_response
_count_animals = _metrics.count_animals
_animals_hash = _metrics.animals_hash


def load_registry(path: Path | str | None = None) -> dict:
    """Load the experiment registry JSON.

    Defaults to ``<ARTIFACTS_DIR>/registry.json``. Pass ``path`` to override.
    """
    if path is None:
        path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    path = Path(path)
    with open(path) as f:
        reg = json.load(f)
    logger.info(
        f"Loaded registry from {path}: "
        f"{len(reg.get('experiments', {}))} experiments, "
        f"{len(reg.get('baselines', {}))} baselines"
    )
    return reg


def _expand_top_animals(reg: dict) -> tuple[list[str], str]:
    """Union the canonical TOP_ANIMALS with target animals present in the
    registry so newly introduced targets are auto-covered."""
    target_animals = {
        d.get("config", {}).get("animal")
        for d in reg.get("experiments", {}).values()
        if d.get("config", {}).get("animal")
    }
    expanded = sorted(set(TOP_ANIMALS) | target_animals)
    return expanded, _animals_hash(expanded)


def get_animal_counts(
    data: dict,
    setting_name: str,
    resp_path: str | Path,
    *,
    top_animals: list[str] | None = None,
    target_animals_hash: str | None = None,
) -> tuple[dict | None, int]:
    """Return ``(counts_dict, total)`` for a given experiment + eval setting.

    Prefers the cached ``animal_counts`` block in the registry (written by the
    eval pipeline) and falls back to live classification of the responses
    JSON when the cache is missing or stale (animal-list hash mismatch).
    """
    if top_animals is None:
        top_animals = TOP_ANIMALS
    if target_animals_hash is None:
        target_animals_hash = _animals_hash(top_animals)

    gen_agg = (data.get("results") or {}).get("generation_aggregate") or {}
    cached = (gen_agg.get(setting_name) or {}).get("animal_counts")
    if isinstance(cached, dict) and cached.get("_animals_hash") == target_animals_hash:
        counts = {k: v for k, v in cached.items() if not k.startswith("_")}
        return counts, cached.get("_total", sum(counts.values()))
    try:
        with open(resp_path) as f:
            prompts_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, 0
    all_responses = [r for p in prompts_data for r in p.get("responses", [])]
    full = _count_animals(all_responses, top_animals)
    total = full.pop("_total")
    full.pop("_animals_hash", None)
    return full, total


def _dataset_source(cfg: dict) -> str:
    if cfg.get("dataset_path"):
        return "external (SL repo)"
    if cfg.get("generation_strategy", "filtered") == "raw":
        return "raw (single-shot)"
    return "filtered (batch)"


def build_gen_df(reg: dict) -> pd.DataFrame:
    """Build the unified one-row-per-``(experiment, eval_setting)`` frame.

    Every ``svd_mode`` / ``dwg_mode`` / dataset source is included with its
    config preserved as columns, so downstream filtering with
    :func:`filter_gen_df` is trivial.
    """
    top_animals, target_hash = _expand_top_animals(reg)
    rows: list[dict[str, Any]] = []
    for exp_id, data in reg.get("experiments", {}).items():
        if data.get("status") != "completed":
            continue
        cfg = data.get("config", {})
        resp_paths = (data.get("results") or {}).get("responses_paths") or {}
        if not resp_paths:
            continue

        raw_svd = cfg.get("svd_mode")
        svd_mode = raw_svd if (raw_svd and _SVD_MODE_RE.match(raw_svd)) else "full"
        dwg_mode = cfg.get("dwg_mode") or "full"

        for setting_name, resp_path in resp_paths.items():
            counts, total = get_animal_counts(
                data,
                setting_name,
                resp_path,
                top_animals=top_animals,
                target_animals_hash=target_hash,
            )
            if not counts or total == 0:
                continue
            animal = cfg.get("animal")

            full_ft = bool(cfg.get("full_finetuning"))
            rows.append({
                "exp_id": exp_id,
                "model_hash": data.get("model_hash"),
                "animal": animal,
                "variant": cfg.get("system_prompt_variant"),
                # rank is meaningless for full-FT runs; null it out so they
                # don't get silently bucketed into a LoRA rank.
                "rank": None if full_ft else cfg.get("lora_rank"),
                "epochs": cfg.get("n_epochs"),
                "full_ft": full_ft,
                "training_seed": cfg.get("training_seed"),
                "generation_seed": cfg.get("generation_seed"),
                "svd_mode": svd_mode,
                "dwg_mode": dwg_mode,
                "raw_svd_mode": cfg.get("svd_mode"),
                "raw_dwg_mode": cfg.get("dwg_mode"),
                "dataset_source": _dataset_source(cfg),
                "dataset_hash": data.get("dataset_hash"),
                "eval_setting": setting_name,
                "generation_temperature": cfg.get("generation_temperature"),
                "generation_strategy": cfg.get("generation_strategy"),
                "train_system_prompt": cfg.get("train_system_prompt"),
                "eval_system_prompt": cfg.get("eval_system_prompt"),
                "student_model": cfg.get("student_model"),
                "n_responses": total,
                "p_target": (counts.get(animal, 0) / total) if animal else np.nan,
            })
    df = pd.DataFrame(rows)
    logger.info(
        f"Built gen_df: {len(df)} rows, "
        f"animals={sorted(df['animal'].dropna().unique()) if not df.empty else []}"
    )
    return df


def _as_list(value):
    """Normalize scalar/list filter values. None means no filtering."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _prompt_filter_mask(series: pd.Series, value) -> pd.Series:
    """Match prompt columns: None=no filter, '<none>'=null, ''=empty string,
    else case-insensitive substring."""
    if value is None:
        return pd.Series(True, index=series.index)
    values = _as_list(value)
    mask = pd.Series(False, index=series.index)
    for item in values:
        if item == "<none>":
            mask |= series.isna()
        elif item == "":
            mask |= series.eq("")
        else:
            mask |= (
                series.fillna("").astype(str).str.contains(str(item), case=False, regex=False)
            )
    return mask


def filter_gen_df(
    df: pd.DataFrame,
    *,
    text: str | None = None,
    animals=None,
    variants=None,
    ranks=None,
    epochs=None,
    training_seeds=None,
    generation_seeds=None,
    generation_temperature=None,
    eval_setting=None,
    train_system_prompt=None,
    eval_system_prompt=None,
    dwg_mode=None,
    svd_mode=None,
    dataset_source=None,
    student_model=None,
    full_ft: bool | None = None,
) -> pd.DataFrame:
    """Filter ``gen_df`` with explicit, dwg_playground-style knobs.

    Set any filter to ``None`` to skip it. For prompt columns use ``"<none>"``
    to match nulls and ``""`` to match the explicit empty-string prompt.
    """
    out = df.copy()

    if text:
        haystack = pd.Series("", index=out.index)
        for col in ["exp_id", "model_hash", "dataset_hash", "animal", "variant", "dwg_mode", "svd_mode"]:
            if col in out.columns:
                haystack = haystack + " " + out[col].fillna("").astype(str)
        out = out[haystack.str.contains(text, case=False, regex=False)]

    equality_filters = {
        "animal": animals,
        "variant": variants,
        "rank": ranks,
        "epochs": epochs,
        "training_seed": training_seeds,
        "generation_seed": generation_seeds,
        "generation_temperature": generation_temperature,
        "eval_setting": eval_setting,
        "dwg_mode": dwg_mode,
        "svd_mode": svd_mode,
        "dataset_source": dataset_source,
        "student_model": student_model,
    }
    for col, value in equality_filters.items():
        values = _as_list(value)
        if values is not None and col in out.columns:
            out = out[out[col].isin(values)]

    if full_ft is not None and "full_ft" in out.columns:
        out = out[out["full_ft"].eq(full_ft)]

    out = out[_prompt_filter_mask(out["train_system_prompt"], train_system_prompt)]
    out = out[_prompt_filter_mask(out["eval_system_prompt"], eval_system_prompt)]
    return out.reset_index(drop=True)


def build_baseline_df(reg: dict) -> pd.DataFrame:
    """One row per cached ``gen_*`` baseline with ``eval_system_prompt=null``
    and no user-prompt prefix.

    Columns: ``animal``, ``base_model``, ``baseline_key``, ``n_prompts``,
    ``n_samples_per_prompt``, ``n_responses``, ``p_target``.
    """
    rows = []
    for key, entry in reg.get("baselines", {}).items():
        if not key.startswith("gen_"):
            continue
        cfg = entry.get("config", {})
        if cfg.get("eval_system_prompt") is not None:
            continue
        if cfg.get("eval_user_prompt_prefix") is not None:
            continue
        gr = entry.get("generation_results", {}).get("clean")
        if not gr:
            continue
        animal = cfg.get("animal")
        if not animal:
            continue
        # Each prompt has the same n_samples, so the simple per-prompt mean
        # equals the overall per-response fraction.
        n_prompts = len(gr)
        n_samples_each = gr[0].get("n_samples", 0)
        p_target = sum(r.get("p_contains_animal", 0.0) for r in gr) / n_prompts
        rows.append({
            "animal": animal,
            "base_model": cfg.get("base_model"),
            "baseline_key": key,
            "n_prompts": n_prompts,
            "n_samples_per_prompt": n_samples_each,
            "n_responses": n_prompts * n_samples_each,
            "p_target": p_target,
        })
    return pd.DataFrame(rows)


_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _parse_module_type(layer_name: str) -> str:
    """Return the LoRA module type (e.g. ``q_proj``) from a canonical layer name.

    Canonical layer names look like
    ``base_model.model.model.layers.<i>.{self_attn|mlp}.<module_type>`` so the
    module type is the final dotted segment.
    """
    return layer_name.rsplit(".", 1)[-1]


def _parse_layer_idx(layer_name: str) -> int | None:
    m = _LAYER_IDX_RE.search(layer_name)
    return int(m.group(1)) if m else None


def load_lora_spectrum_df(
    model_hashes: Iterable[str],
    *,
    artifacts_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load per-(model, layer) singular spectra from cached SVD ``.npz`` files.

    For each ``model_hash`` we read ``<artifacts_dir>/svd/<hash>.npz`` (written
    by :func:`benchmarks.svd.compute_svd_cache`) and emit one row per
    ``(model_hash, layer, k)``. Missing caches are skipped with a single
    aggregate warning.

    Parameters
    ----------
    model_hashes:
        Iterable of model hashes (typically ``gen_df["model_hash"].unique()``
        after filtering to LoRA-only canonical runs).
    artifacts_dir:
        Override the location of the ``svd/`` subdirectory; defaults to
        ``sl_config.ARTIFACTS_DIR``.

    Returns
    -------
    pd.DataFrame with columns:
        ``model_hash`` (str),
        ``layer`` (str, canonical LoRA module name),
        ``module_type`` (str, e.g. ``q_proj``/``down_proj``),
        ``layer_idx`` (int, 0-indexed transformer block),
        ``k`` (int, 1-indexed singular index),
        ``s`` (float, raw singular value),
        ``s_norm_top1`` (float, ``s_k / s_1``),
        ``s_norm_l2`` (float, ``s_k / sqrt(sum_j s_j^2)``),
        ``lora_rank`` (int, length of ``s`` for that module).
    """
    if artifacts_dir is None:
        artifacts_dir = sl_config.ARTIFACTS_DIR
    svd_dir = Path(artifacts_dir) / "svd"

    hashes = list(dict.fromkeys(model_hashes))  # de-dup, preserve order
    chunks: list[pd.DataFrame] = []
    missing: list[str] = []

    corrupt: list[str] = []
    for model_hash in hashes:
        path = svd_dir / f"{model_hash}.npz"
        if not path.exists():
            missing.append(model_hash)
            continue
        # Concurrent benchmark runs can leave a cache mid-write
        # (`compute_svd_cache` stages to a tmp file and `os.replace`s, so
        # corruption is rare but EOFError/BadZipFile can still surface if
        # we read across an os.replace boundary).
        try:
            with np.load(path) as data:
                layers = [str(x) for x in data["layers"]]
                # Stack singular-value vectors. All modules in a single
                # adapter share the same lora_rank, so this is a
                # (n_layers, rank) matrix.
                s_list = [data[f"{layer}.s"].astype(np.float64) for layer in layers]
        except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
            corrupt.append(f"{model_hash} ({type(e).__name__})")
            continue
        if not s_list:
            continue
        S = np.stack(s_list)  # (n_layers, rank)
        n_layers, rank = S.shape
        if rank == 0:
            continue

        s1 = S[:, :1]  # (n_layers, 1)
        l2 = np.sqrt((S ** 2).sum(axis=1, keepdims=True))  # (n_layers, 1)
        # Guard against zero spectra (degenerate adapter modules).
        s1_safe = np.where(s1 > 0, s1, 1.0)
        l2_safe = np.where(l2 > 0, l2, 1.0)
        s_norm_top1 = S / s1_safe
        s_norm_l2 = S / l2_safe

        module_types = np.array([_parse_module_type(l) for l in layers])
        layer_idxs = np.array(
            [_parse_layer_idx(l) for l in layers], dtype="object"
        )

        # Tile per-layer columns and broadcast k across (n_layers, rank).
        layer_arr = np.repeat(np.array(layers), rank)
        module_type_arr = np.repeat(module_types, rank)
        layer_idx_arr = np.repeat(layer_idxs, rank)
        k_arr = np.tile(np.arange(1, rank + 1), n_layers)

        chunk = pd.DataFrame({
            "model_hash": model_hash,
            "layer": layer_arr,
            "module_type": module_type_arr,
            "layer_idx": layer_idx_arr,
            "k": k_arr,
            "s": S.ravel(),
            "s_norm_top1": s_norm_top1.ravel(),
            "s_norm_l2": s_norm_l2.ravel(),
            "lora_rank": rank,
        })
        chunks.append(chunk)

    if missing:
        logger.warning(
            f"load_lora_spectrum_df: {len(missing)}/{len(hashes)} model hashes "
            f"had no SVD cache (first few: {missing[:3]})"
        )
    if corrupt:
        logger.warning(
            f"load_lora_spectrum_df: {len(corrupt)}/{len(hashes)} model hashes "
            f"had unreadable SVD cache (first few: {corrupt[:3]})"
        )
    df = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(
            columns=[
                "model_hash", "layer", "module_type", "layer_idx",
                "k", "s", "s_norm_top1", "s_norm_l2", "lora_rank",
            ]
        )
    )
    logger.info(
        f"Built spectrum_df: {len(df)} rows, "
        f"models={df['model_hash'].nunique() if not df.empty else 0}, "
        f"ranks={sorted(df['lora_rank'].unique()) if not df.empty else []}"
    )
    return df


def build_baseline_animal_counts(
    reg: dict,
    *,
    animals: list[str] | None = None,
    cache_path: Path | str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Long-form ``(base_model, animal) -> count`` table from baseline responses.

    For every ``gen_*`` baseline with no eval system prompt and no user-prompt
    prefix, reads the raw responses JSON, classifies every response into
    ``animals`` plus the ``"other"`` bucket via :func:`classify_response`, and
    aggregates across baselines per ``base_model``. Different baseline targets
    for the same model use the same "name an animal" prompt set, so summing
    across them just gives more independent samples per model.

    Parameters
    ----------
    reg:
        Registry dict from :func:`load_registry`.
    animals:
        Animal classifier list. Defaults to :data:`TOP_ANIMALS`. Pass an
        extended list (e.g. ``TOP_ANIMALS + ["otter", "raven", ...]``) to
        surface model-specific dominant outputs that aren't in the canonical
        set without polluting the global classifier.
    cache_path:
        Optional JSON path to read/write a memoized result. The cache is keyed
        by the animals-list hash; a mismatch triggers a recompute. Defaults
        to ``<ARTIFACTS_DIR>/baseline_animal_counts.json`` when
        ``cache_path=True`` is passed via the convenience entry point in the
        notebook (call sites can pass an explicit ``Path`` for full control).
    force:
        Ignore any existing cache and reclassify from raw responses.

    Returns
    -------
    pd.DataFrame with columns:
        ``base_model`` (str),
        ``animal`` (str, including ``"other"``),
        ``count`` (int, # responses classified into this bucket),
        ``n_responses`` (int, total responses for this ``base_model``),
        ``p`` (float, ``count / n_responses``).
    """
    if animals is None:
        animals = list(TOP_ANIMALS)
    target_hash = _animals_hash(animals)

    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and not force and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("_animals_hash") == target_hash:
                rows = cached.get("rows", [])
                logger.info(
                    f"build_baseline_animal_counts: cache hit "
                    f"({cache_path}, {len(rows)} rows)"
                )
                return pd.DataFrame(rows)
            logger.info(
                f"build_baseline_animal_counts: cache hash mismatch "
                f"({cache_path}); recomputing"
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"build_baseline_animal_counts: failed to load cache "
                f"{cache_path} ({type(e).__name__}); recomputing"
            )

    by_model: dict[str, dict[str, int]] = {}
    n_responses: dict[str, int] = {}
    n_baselines: dict[str, int] = {}
    n_files_read = 0
    for key, entry in reg.get("baselines", {}).items():
        if not key.startswith("gen_"):
            continue
        cfg = entry.get("config", {})
        if cfg.get("eval_system_prompt") is not None:
            continue
        if cfg.get("eval_user_prompt_prefix") is not None:
            continue
        base_model = cfg.get("base_model")
        if not base_model:
            continue
        gr = entry.get("generation_results", {}).get("clean") or []
        seen_paths: set[str] = set()
        for r in gr:
            path = r.get("responses_path")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with open(path) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            n_files_read += 1
            counter = by_model.setdefault(base_model, {a: 0 for a in animals + ["other"]})
            for prompt_block in data:
                for resp in prompt_block.get("responses", []):
                    label = classify_response(resp, animals)
                    counter[label] = counter.get(label, 0) + 1
                    n_responses[base_model] = n_responses.get(base_model, 0) + 1
        n_baselines[base_model] = n_baselines.get(base_model, 0) + 1

    rows: list[dict[str, Any]] = []
    for base_model, counter in by_model.items():
        total = n_responses.get(base_model, 0)
        for animal, count in counter.items():
            rows.append({
                "base_model": base_model,
                "animal": animal,
                "count": int(count),
                "n_responses": int(total),
                "p": (count / total) if total else float("nan"),
            })

    df = pd.DataFrame(rows)
    logger.info(
        f"Built baseline_animal_counts: {len(df)} rows over "
        f"{df['base_model'].nunique() if not df.empty else 0} base models, "
        f"{n_files_read} responses files, "
        f"animals={len(animals)} (+'other'); "
        f"per-model n_responses={n_responses}"
    )

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "_animals_hash": target_hash,
                        "_animals": list(animals),
                        "rows": df.to_dict(orient="records"),
                    },
                    f,
                )
            tmp.replace(cache_path)
            logger.info(
                f"build_baseline_animal_counts: wrote cache {cache_path}"
            )
        except OSError as e:
            logger.warning(
                f"build_baseline_animal_counts: failed to write cache "
                f"{cache_path} ({type(e).__name__})"
            )

    return df


def compute_baseline_p(baseline_df: pd.DataFrame) -> dict[str, float]:
    """Collapse :func:`build_baseline_df` to one ``p_target`` per animal,
    response-weighted across any duplicate baseline entries (e.g. ``cat`` has
    multiple near-identical runs)."""
    if baseline_df.empty:
        return {}
    return (
        baseline_df.groupby("animal")
        .apply(
            lambda g: (g["p_target"] * g["n_responses"]).sum() / g["n_responses"].sum(),
            include_groups=False,
        )
        .to_dict()
    )


__all__ = [
    "TOP_ANIMALS",
    "classify_response",
    "load_registry",
    "get_animal_counts",
    "build_gen_df",
    "filter_gen_df",
    "build_baseline_df",
    "build_baseline_animal_counts",
    "compute_baseline_p",
    "load_lora_spectrum_df",
]
