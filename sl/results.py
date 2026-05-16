"""Shared infrastructure for loading and filtering subliminal-learning results.

Data layer for analysis notebooks: registry loading, per-experiment frame
construction, prompt-aware filtering, and base-model overlays. The plotting
layer lives in :mod:`sl.figures`.

Public surface:

- :func:`load_registry` — read ``ARTIFACTS_DIR/registry.json``.
- :func:`build_gen_df` — unified one-row-per-``(experiment, eval_setting)`` frame.
- :func:`filter_gen_df` — explicit-knobs filter with prompt-aware semantics.
- :func:`build_baseline_df` / :func:`compute_baseline_p` — base-model overlays.
- :data:`TOP_ANIMALS`, :func:`classify_response` — the shared animal classifier.
"""

from __future__ import annotations

import hashlib
import importlib.util as _iu
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger

try:
    import orjson as _orjson  # 3-5x faster than stdlib json on the 575 MB registry
except ImportError:  # pragma: no cover - orjson is in uv.lock but fall back gracefully
    _orjson = None

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

# Import the v4 stable superset directly from sl.animals (not via the
# benchmarks shim which doesn't re-export it).
from sl.animals import COUNT_CACHE_TARGETS  # noqa: E402

# Sentinel: "don't filter on eval_system_prompt" for the pooled-baseline
# helpers. Defined here (rather than alongside :func:`compute_baseline_p_pooled`
# further down) so :func:`get_baseline_p_cached` can use it as a default-arg
# value without a forward-reference NameError at import time.
ANY_SYS_PROMPT = object()


def load_registry(path: Path | str | None = None) -> dict:
    """Load the experiment registry JSON.

    Defaults to ``<ARTIFACTS_DIR>/registry.json``. Pass ``path`` to override.

    Uses ``orjson`` when available (3-5x faster on the 575 MB registry) and
    falls back to stdlib ``json`` otherwise.
    """
    if path is None:
        path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    path = Path(path)
    if _orjson is not None:
        with open(path, "rb") as f:
            reg = _orjson.loads(f.read())
    else:
        with open(path) as f:
            reg = json.load(f)
    logger.info(
        f"Loaded registry from {path}: "
        f"{len(reg.get('experiments', {}))} experiments, "
        f"{len(reg.get('baselines', {}))} baselines"
    )
    return reg


# Bump when build_gen_df / build_baseline_df row schema or values change in a
# non-backward-compatible way so cached parquet views auto-invalidate.
_VIEW_CODE_VERSION = "v1"


def _registry_view_paths(reg_path: Path) -> tuple[Path, Path]:
    """Return ``(views_dir, manifest_path)`` next to the given registry."""
    views_dir = reg_path.parent / "views"
    return views_dir, views_dir / "manifest.json"


def _target_hash(reg: dict) -> str:
    """Cheap hash over the canonical target set used by ``build_gen_df``.

    Cached views are invalidated whenever this changes (e.g. user adds a new
    sweep with a new ``config["animal"]``).
    """
    targets = sorted({
        a.lower()
        for d in reg.get("experiments", {}).values()
        if isinstance((a := d.get("config", {}).get("animal")), str)
    })
    blob = ",".join(targets).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_view_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_view_manifest(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(manifest_path)


def _view_cache_key(reg_path: Path, target_hash: str) -> dict:
    return {
        "reg_mtime_ns": reg_path.stat().st_mtime_ns,
        "reg_size": reg_path.stat().st_size,
        "target_hash": target_hash,
        "code_version": _VIEW_CODE_VERSION,
    }


def _view_is_fresh(manifest: dict, key: str, expected: dict) -> bool:
    entry = manifest.get(key) or {}
    return all(entry.get(k) == v for k, v in expected.items())


def build_gen_df_cached(
    reg_path: Path | str | None = None,
    *,
    force: bool = False,
    reg: dict | None = None,
) -> pd.DataFrame:
    """Cached wrapper around :func:`build_gen_df`.

    Materialises ``<registry>.parent/views/gen_df.parquet`` keyed by registry
    mtime + size + target-set hash + code version. Warm reload is ~100 ms; cold
    rebuild does a single full registry load. Pass ``force=True`` to bypass the
    cache, or ``reg`` to reuse an already-loaded registry dict (still honours
    the cache; pass ``force=True`` alongside to actually rebuild).
    """
    if reg_path is None:
        reg_path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    reg_path = Path(reg_path)
    views_dir, manifest_path = _registry_view_paths(reg_path)
    cache_path = views_dir / "gen_df.parquet"

    if reg is None and not force and cache_path.exists():
        manifest = _load_view_manifest(manifest_path)
        expected = _view_cache_key(reg_path, target_hash="")
        # target_hash check requires the registry, so verify the cheap keys
        # first and only do the registry load if those match.
        cheap_keys_match = all(
            (manifest.get("gen_df") or {}).get(k) == v
            for k, v in expected.items()
            if k != "target_hash"
        )
        if cheap_keys_match:
            logger.info(f"Loaded cached gen_df view from {cache_path}")
            return pd.read_parquet(cache_path)

    if reg is None:
        reg = load_registry(reg_path)
    target_hash = _target_hash(reg)
    expected = _view_cache_key(reg_path, target_hash=target_hash)

    # Full-key check: if target hash matches too, just read parquet.
    if not force and cache_path.exists():
        manifest = _load_view_manifest(manifest_path)
        if _view_is_fresh(manifest, "gen_df", expected):
            logger.info(f"Loaded cached gen_df view from {cache_path}")
            return pd.read_parquet(cache_path)

    df = build_gen_df(reg)
    views_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    manifest = _load_view_manifest(manifest_path)
    manifest["gen_df"] = expected
    _save_view_manifest(manifest_path, manifest)
    logger.info(f"Wrote gen_df view to {cache_path}")
    return df


def build_baseline_p_view(reg: dict) -> pd.DataFrame:
    """Materialise pooled-baseline animal counts as a tidy DataFrame.

    One row per ``(base_model, eval_system_prompt_repr, animal)`` cohort.
    ``count`` is the number of pooled responses classified into that animal
    bucket; ``n_total`` is the total responses in the cohort; ``n_runs`` is
    the number of contributing ``gen_*`` baseline entries.

    The cohort is exactly what :func:`compute_baseline_p_pooled` would build
    at notebook time, but the work happens once at view-build time and
    notebooks just project rows out of the parquet.

    ``eval_system_prompt_repr`` encodes the three meaningful states with
    sentinel strings so the column is JSON/parquet-friendly:

    - ``"<null>"``  -- ``cfg["eval_system_prompt"] is None`` (Qwen default-sys
      pool, Gemma true-no-sys pool).
    - ``"<empty>"`` -- ``cfg["eval_system_prompt"] == ""`` (Qwen true-no-sys
      pool).
    - any other ``str`` -- explicit system prompt content (verbatim).

    The animal axis is :data:`COUNT_CACHE_TARGETS` plus ``"other"`` so the
    view supports queries for any canonical target without re-classifying.
    """
    targets = list(COUNT_CACHE_TARGETS)
    target_set = set(targets)

    # Group baselines by (base_model, sys_prompt). Each value is the dedup'd
    # set of (responses_path, run_key) tuples contributing to the cohort.
    cohorts: dict[tuple[str, str], dict[str, str]] = {}
    for key, entry in reg.get("baselines", {}).items():
        if not key.startswith("gen_"):
            continue
        cfg = entry.get("config", {})
        if cfg.get("eval_user_prompt_prefix") is not None:
            continue
        base_model = cfg.get("base_model")
        if not base_model:
            continue
        sp = cfg.get("eval_system_prompt")
        sp_repr = "<null>" if sp is None else ("<empty>" if sp == "" else sp)
        cohort_key = (base_model, sp_repr)
        seen = cohorts.setdefault(cohort_key, {})
        gr_by_setting = entry.get("generation_results", {}) or {}
        for _setting, results in gr_by_setting.items():
            for r in results or []:
                path = r.get("responses_path")
                if not path or path in seen:
                    continue
                # Stash the originating baseline key so we can count
                # ``n_runs`` distinctly even though paths are de-duplicated.
                seen[path] = key

    rows: list[dict] = []
    for (base_model, sp_repr), seen in cohorts.items():
        counts: dict[str, int] = {a: 0 for a in targets}
        counts["other"] = 0
        n_total = 0
        n_files = 0
        n_runs = len(set(seen.values()))
        for path in seen:
            try:
                with open(path) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(
                    f"build_baseline_p_view: skipping {path} "
                    f"({type(e).__name__}: {e})"
                )
                continue
            n_files += 1
            for prompt_block in data:
                for resp in prompt_block.get("responses", []):
                    label = classify_response(resp, targets)
                    if label in target_set:
                        counts[label] += 1
                    else:
                        counts["other"] += 1
                    n_total += 1
        if n_total == 0:
            continue
        for animal in [*targets, "other"]:
            rows.append({
                "base_model": base_model,
                "eval_system_prompt_repr": sp_repr,
                "animal": animal,
                "count": counts[animal],
                "n_total": n_total,
                "n_runs": n_runs,
                "n_files": n_files,
                "p_target": counts[animal] / n_total,
            })

    df = pd.DataFrame(rows)
    logger.info(
        f"Built baseline_p view: {len(df)} rows, "
        f"{df['base_model'].nunique() if not df.empty else 0} base_model(s), "
        f"{df.groupby(['base_model','eval_system_prompt_repr']).ngroups if not df.empty else 0} cohort(s)"
    )
    return df


def build_baseline_p_view_cached(
    reg_path: Path | str | None = None,
    *,
    force: bool = False,
    reg: dict | None = None,
) -> pd.DataFrame:
    """Cached wrapper around :func:`build_baseline_p_view`. See :func:`build_gen_df_cached`."""
    if reg_path is None:
        reg_path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    reg_path = Path(reg_path)
    views_dir, manifest_path = _registry_view_paths(reg_path)
    cache_path = views_dir / "baseline_p.parquet"

    if reg is None and not force and cache_path.exists():
        manifest = _load_view_manifest(manifest_path)
        expected = _view_cache_key(reg_path, target_hash="")
        cheap_keys_match = all(
            (manifest.get("baseline_p") or {}).get(k) == v
            for k, v in expected.items()
            if k != "target_hash"
        )
        if cheap_keys_match:
            logger.info(f"Loaded cached baseline_p view from {cache_path}")
            return pd.read_parquet(cache_path)

    if reg is None:
        reg = load_registry(reg_path)
    target_hash = _target_hash(reg)
    expected = _view_cache_key(reg_path, target_hash=target_hash)

    if not force and cache_path.exists():
        manifest = _load_view_manifest(manifest_path)
        if _view_is_fresh(manifest, "baseline_p", expected):
            logger.info(f"Loaded cached baseline_p view from {cache_path}")
            return pd.read_parquet(cache_path)

    df = build_baseline_p_view(reg)
    views_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    manifest = _load_view_manifest(manifest_path)
    manifest["baseline_p"] = expected
    _save_view_manifest(manifest_path, manifest)
    logger.info(f"Wrote baseline_p view to {cache_path}")
    return df


# Cache the loaded view in-memory so repeated lookups don't re-read parquet.
_BASELINE_P_VIEW_CACHE: dict[Path, pd.DataFrame] = {}


def get_baseline_p_cached(
    *,
    base_model: str,
    eval_system_prompt: str | None | object = None,
    animals: list[str] | None = None,
    reg_path: Path | str | None = None,
) -> dict[str, float]:
    """Drop-in replacement for :func:`compute_baseline_p_pooled` that reads
    the precomputed :file:`views/baseline_p.parquet` instead of walking the
    registry + opening response files.

    Same semantics for ``eval_system_prompt``: ``None`` matches the
    ``<null>`` cohort, ``""`` matches ``<empty>``, any other ``str`` matches
    the cohort with that exact prompt. :data:`ANY_SYS_PROMPT` pools across
    every cohort for the given ``base_model``.

    Raises ``FileNotFoundError`` if the view hasn't been built yet (run
    ``python scripts/rebuild_views.py`` first). Returns ``{}`` and logs a
    warning if no cohort matches.
    """
    if reg_path is None:
        reg_path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    reg_path = Path(reg_path)
    cache_path = reg_path.parent / "views" / "baseline_p.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"baseline_p view not found at {cache_path}. "
            f"Run `python scripts/rebuild_views.py` to materialise it."
        )
    df = _BASELINE_P_VIEW_CACHE.get(cache_path)
    if df is None:
        df = pd.read_parquet(cache_path)
        _BASELINE_P_VIEW_CACHE[cache_path] = df

    if animals is None:
        animals = list(TOP_ANIMALS)
    canonical_animals = [a.lower() for a in animals]

    sub = df[df["base_model"] == base_model]
    if eval_system_prompt is ANY_SYS_PROMPT:
        # Pool across every cohort for this base_model. Sum counts and totals
        # over the per-cohort rows. n_total is per-cohort and gets summed
        # alongside counts, so dividing produces the response-weighted mean.
        pass
    else:
        sp_repr = (
            "<null>" if eval_system_prompt is None
            else ("<empty>" if eval_system_prompt == "" else eval_system_prompt)
        )
        sub = sub[sub["eval_system_prompt_repr"] == sp_repr]

    if sub.empty:
        sp_dbg = (
            "<any>" if eval_system_prompt is ANY_SYS_PROMPT else (
                "<null>" if eval_system_prompt is None
                else ("<empty>" if eval_system_prompt == "" else f"{eval_system_prompt!r}")
            )
        )
        logger.warning(
            f"get_baseline_p_cached: no rows for base_model={base_model!r}, "
            f"eval_system_prompt={sp_dbg}. Did the cohort exist when views were built?"
        )
        return {}

    # Aggregate over cohorts (only matters for ANY_SYS_PROMPT). For a single
    # cohort this is a no-op pass-through.
    pooled = (
        sub.groupby("animal", as_index=False)[["count", "n_total"]].sum()
    )
    # n_total is per-cohort and was summed per-animal. Recover the actual
    # cohort-total by dividing the n_total sum by (#animals + 1 for "other").
    # Cleaner: take the n_total of any animal row (all equal within a cohort)
    # and sum over cohorts. Build that separately for accuracy.
    cohort_totals = sub.groupby(
        "eval_system_prompt_repr"
    )["n_total"].first().sum()

    out: dict[str, float] = {}
    counts_by_animal = dict(zip(pooled["animal"], pooled["count"]))
    for animal in canonical_animals:
        out[animal] = counts_by_animal.get(animal, 0) / cohort_totals if cohort_totals else 0.0
    return out


def build_baseline_df_cached(
    reg_path: Path | str | None = None,
    *,
    force: bool = False,
    reg: dict | None = None,
) -> pd.DataFrame:
    """Cached wrapper around :func:`build_baseline_df`. See :func:`build_gen_df_cached`."""
    if reg_path is None:
        reg_path = Path(sl_config.ARTIFACTS_DIR) / "registry.json"
    reg_path = Path(reg_path)
    views_dir, manifest_path = _registry_view_paths(reg_path)
    cache_path = views_dir / "baseline_df.parquet"

    if reg is None and not force and cache_path.exists():
        manifest = _load_view_manifest(manifest_path)
        expected = _view_cache_key(reg_path, target_hash="")
        cheap_keys_match = all(
            (manifest.get("baseline_df") or {}).get(k) == v
            for k, v in expected.items()
            if k != "target_hash"
        )
        if cheap_keys_match:
            logger.info(f"Loaded cached baseline_df view from {cache_path}")
            return pd.read_parquet(cache_path)

    if reg is None:
        reg = load_registry(reg_path)
    target_hash = _target_hash(reg)
    expected = _view_cache_key(reg_path, target_hash=target_hash)

    if not force and cache_path.exists():
        manifest = _load_view_manifest(manifest_path)
        if _view_is_fresh(manifest, "baseline_df", expected):
            logger.info(f"Loaded cached baseline_df view from {cache_path}")
            return pd.read_parquet(cache_path)

    df = build_baseline_df(reg)
    views_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    manifest = _load_view_manifest(manifest_path)
    manifest["baseline_df"] = expected
    _save_view_manifest(manifest_path, manifest)
    logger.info(f"Wrote baseline_df view to {cache_path}")
    return df


def _expand_top_animals(reg: dict) -> tuple[list[str], str]:
    """Union the canonical TOP_ANIMALS with target animals present in the
    registry so newly introduced targets are auto-covered.

    Lower-cased on the way out so band-sweep configs that ship their
    target list capitalized (e.g. ``"Eagles"``, ``"The Beatles"``) don't
    collide with the lower-cased canonical names produced by the
    classifier in :mod:`sl.animals`.

    As of the v4 cache: the second tuple element (``hash``) is the stable
    target-set-independent classifier hash, NOT a hash over ``expanded``.
    Kept here for back-compat with callers that still expect the
    ``(animals, hash)`` shape; new code can call :func:`_animals_hash` (with
    no args) directly.
    """
    target_animals = {
        a.lower()
        for d in reg.get("experiments", {}).values()
        if (a := d.get("config", {}).get("animal"))
    }
    expanded = sorted({a.lower() for a in TOP_ANIMALS} | target_animals)
    return expanded, _animals_hash()


def get_animal_counts(
    data: dict,
    setting_name: str,
    resp_path: str | Path,
    *,
    top_animals: list[str] | None = None,
    target_animals_hash: str | None = None,
    target_animal: str | None = None,
) -> tuple[dict | None, int]:
    """Return ``(counts_dict, total)`` for a given experiment + eval setting.

    Prefers the cached ``animal_counts`` block in the registry (written by the
    eval pipeline) and falls back to live classification of the responses
    JSON in two cases:

    1. The cached entry is missing or its ``_animals_hash`` doesn't match
       :func:`sl.animals.animals_hash` (i.e. the classifier semantics have
       changed since the entry was written -- v3 entries vs. v4 readers,
       say).
    2. The caller specifies a ``target_animal`` that isn't in the cached
       dict. Under v4, the cached dict contains buckets for
       :data:`sl.animals.COUNT_CACHE_TARGETS` (the stable superset) -- any
       target the caller cares about that isn't in that superset must be
       classified live until :data:`TOP_TARGETS` is extended and the
       backfill is re-run.
    """
    if top_animals is None:
        top_animals = TOP_ANIMALS
    if target_animals_hash is None:
        target_animals_hash = _animals_hash()

    gen_agg = (data.get("results") or {}).get("generation_aggregate") or {}
    cached = (gen_agg.get(setting_name) or {}).get("animal_counts")
    if isinstance(cached, dict) and cached.get("_animals_hash") == target_animals_hash:
        counts = {k: v for k, v in cached.items() if not k.startswith("_")}
        if target_animal is None or target_animal.lower() in counts:
            return counts, cached.get("_total", sum(counts.values()))
        # Cache is v4-fresh but doesn't have a bucket for this target. Fall
        # through to live classification rather than silently returning 0.
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
        # Pull decode_state out of the dwg_spec dict so we can distinguish
        # post-Apr-29 fixed runs (decode_state="outside_q") from legacy
        # buggy ones (decode_state absent => spec_mask was used). `no_*`
        # modes are mathematically unaffected by the bug; see
        # scripts/verify_dwg_complement.py.
        dwg_spec = cfg.get("dwg_spec") or {}
        decode_state = dwg_spec.get("decode_state") if isinstance(dwg_spec, dict) else None

        raw_animal = cfg.get("animal")
        target_animal = raw_animal.lower() if isinstance(raw_animal, str) else None
        for setting_name, resp_path in resp_paths.items():
            counts, total = get_animal_counts(
                data,
                setting_name,
                resp_path,
                top_animals=top_animals,
                target_animals_hash=target_hash,
                target_animal=target_animal,
            )
            if not counts or total == 0:
                continue
            # ``target_animal`` was lower-cased above so it already matches
            # the lower-cased keys produced by :func:`count_animals`.
            animal = target_animal if target_animal is not None else raw_animal

            full_ft = bool(cfg.get("full_finetuning"))
            # Preference category. Pre-categories runs (animal-only) don't
            # store the field; treat absence as "animal" for back-compat.
            category = cfg.get("category", "animal") or "animal"
            rows.append({
                "exp_id": exp_id,
                "model_hash": data.get("model_hash"),
                "animal": animal,
                "category": category,
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
                "decode_state": decode_state,
                "raw_svd_mode": cfg.get("svd_mode"),
                "raw_dwg_mode": cfg.get("dwg_mode"),
                "dataset_source": _dataset_source(cfg),
                "dataset_hash": data.get("dataset_hash"),
                "eval_setting": setting_name,
                "generation_temperature": cfg.get("generation_temperature"),
                "generation_strategy": cfg.get("generation_strategy"),
                "train_system_prompt": cfg.get("train_system_prompt"),
                "eval_system_prompt": cfg.get("eval_system_prompt"),
                # use_chat_template flag: the no-template ablation runs raw
                # prompt+completion through the model with no chat scaffolding
                # (see plans/no_template_training.md). Older runs predate the
                # flag and default to True (chat template active).
                "use_chat_template": bool(cfg.get("use_chat_template", True)),
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
    categories=None,
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
    decode_state=None,
    dataset_source=None,
    student_model=None,
    full_ft: bool | None = None,
    use_chat_template: bool | None = None,
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
        "category": categories,
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

    if use_chat_template is not None and "use_chat_template" in out.columns:
        out = out[out["use_chat_template"].eq(use_chat_template)]

    if decode_state is not None and "decode_state" in out.columns:
        # "<none>" matches legacy / pre-fix rows (decode_state was not stored
        # in dwg_spec); a list lets you mix legacy + fixed (e.g.
        # decode_state=["<none>", "outside_q"]).
        values = _as_list(decode_state)
        col = out["decode_state"]
        mask = pd.Series(False, index=out.index)
        for v in values:
            mask |= col.isna() if v == "<none>" else col.eq(v)
        out = out[mask]

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
        # Canonicalise to lowercase to match :func:`build_gen_df` and the
        # lowercase keys used in :data:`sl.animals.TOP_TARGETS` /
        # :data:`sl.figures.BAND_COLORS`. Band-sweep configs ship target
        # names capitalised (e.g. ``"Led Zeppelin"``), so without this the
        # baseline frame's ``animal`` column is inconsistent with
        # ``gen_df["animal"]`` and any lowercase ``.isin(...)`` filter (as
        # used in the Appendix F band figure in ``paper_figures.ipynb``)
        # silently drops every band baseline.
        raw_animal = cfg.get("animal")
        if not raw_animal:
            continue
        animal = raw_animal.lower() if isinstance(raw_animal, str) else raw_animal
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
    eval_system_prompt: str | None | object = None,
    cache_path: Path | str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Long-form ``(base_model, animal) -> count`` table from baseline responses.

    For every ``gen_*`` baseline matching the ``eval_system_prompt`` filter
    and with no user-prompt prefix, reads the raw responses JSON, classifies
    every response into ``animals`` plus the ``"other"`` bucket via
    :func:`classify_response`, and aggregates across baselines per
    ``base_model``. Different baseline targets for the same model use the
    same "name an animal" prompt set, so summing across them just gives more
    independent samples per model.

    Parameters
    ----------
    reg:
        Registry dict from :func:`load_registry`.
    animals:
        Animal classifier list. Defaults to :data:`TOP_ANIMALS`. Pass an
        extended list (e.g. ``TOP_ANIMALS + ["otter", "raven", ...]``) to
        surface model-specific dominant outputs that aren't in the canonical
        set without polluting the global classifier.
    eval_system_prompt:
        Same semantics as :func:`compute_baseline_p_pooled`:
        ``None`` (default) keeps only baselines with no eval system prompt;
        ``""`` selects the explicit empty-string-system-prompt pool;
        :data:`ANY_SYS_PROMPT` disables filtering. The cache key includes
        this value so different filters cache to different rows of the same
        file (see ``_eval_system_prompt`` field in the cached payload).
    cache_path:
        Optional JSON path to read/write a memoized result. The cache is keyed
        by the animals-list hash *and* the ``eval_system_prompt`` filter;
        either mismatch triggers a recompute.
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

    # Stable cache key for the sys-prompt filter (must JSON-roundtrip).
    if eval_system_prompt is ANY_SYS_PROMPT:
        sp_cache_key = "<any>"
    elif eval_system_prompt is None:
        sp_cache_key = "<null>"
    else:
        sp_cache_key = str(eval_system_prompt)

    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and not force and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if (
                cached.get("_animals_hash") == target_hash
                and cached.get("_eval_system_prompt", "<null>") == sp_cache_key
            ):
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
        sp = cfg.get("eval_system_prompt")
        if eval_system_prompt is not ANY_SYS_PROMPT and sp != eval_system_prompt:
            continue
        if cfg.get("eval_user_prompt_prefix") is not None:
            continue
        base_model = cfg.get("base_model")
        if not base_model:
            continue
        # Iterate every setting key — null-sys baselines were stored under
        # ``clean`` but empty-string-sys baselines live under ``with_system``.
        # De-dup by responses_path so we never double-count.
        seen_paths: set[str] = set()
        run_used = False
        for _setting, results in (entry.get("generation_results") or {}).items():
            for r in results or []:
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
                run_used = True
                counter = by_model.setdefault(base_model, {a: 0 for a in animals + ["other"]})
                for prompt_block in data:
                    for resp in prompt_block.get("responses", []):
                        label = classify_response(resp, animals)
                        counter[label] = counter.get(label, 0) + 1
                        n_responses[base_model] = n_responses.get(base_model, 0) + 1
        if run_used:
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
                        "_eval_system_prompt": sp_cache_key,
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


def compute_baseline_p_pooled(
    reg: dict,
    *,
    base_model: str,
    eval_system_prompt: str | None | object = ANY_SYS_PROMPT,
    animals: list[str] | None = None,
) -> dict[str, float]:
    """Pooled per-animal P(target) for a ``(base_model, eval_system_prompt)`` cohort.

    Unlike :func:`build_baseline_df` + :func:`compute_baseline_p` (which read
    each baseline run's precomputed scalar ``p_contains_animal`` and so are
    keyed *per (target_animal, baseline_run)*), this helper opens the raw
    responses JSON for every matching baseline, classifies each response with
    :func:`classify_response` over the supplied ``animals`` list, and pools
    counts across baseline runs.

    The pooling is sound because all ``gen_*`` baselines for a given
    ``base_model`` reuse the same canonical "Name your favorite animal..."
    prompt set — the per-run ``cfg["animal"]`` field is just the target the
    run was originally invoked for, not a property of the prompts. So a
    Qwen + ``eval_system_prompt=""`` cohort with 9 runs × 50 prompts × 100
    samples = 45 000 base-model responses can be classified into *any* target
    animal we care about (cat / owl / dolphin / eagle / wolf / etc.) without
    any new generation.

    Parameters
    ----------
    reg:
        Registry dict from :func:`load_registry`.
    base_model:
        E.g. ``"unsloth/Qwen2.5-7B-Instruct"`` — required because each base
        model has its own response distribution.
    eval_system_prompt:
        - ``None`` — pool only baselines with no eval system prompt (i.e.
          ``cfg["eval_system_prompt"] is None``). For Qwen this triggers the
          chat template's default ``"You are Qwen, created by Alibaba Cloud..."``,
          so this is the *default-system-prompt* pool. For Gemma (template
          has no system role) and Qwen-style models without a default, this
          is a true no-system-prompt pool.
        - ``""`` (empty string) — pool only baselines with an explicit empty
          system prompt. This is the truest "no system prompt at all" pool
          for models whose chat template would otherwise inject a default.
        - any other ``str`` — pool baselines with that exact system prompt.
        - :data:`ANY_SYS_PROMPT` (the default) — don't filter on system
          prompt; pool every matching ``base_model`` baseline.
    animals:
        Animal classifier list. Defaults to :data:`TOP_ANIMALS`. Pass an
        extended list to surface model-specific picks (e.g. Llama's "llama"
        self-id, Gemma's "otter") on top of the canonical set.

    Returns
    -------
    dict[str, float]
        ``{animal: count_target / n_responses}`` for every animal in
        ``animals``. Animals with zero matches are still present (mapped to
        ``0.0``). Returns an empty dict if no matching baselines exist or
        none of their response files could be read.
    """
    if animals is None:
        animals = list(TOP_ANIMALS)

    counts: dict[str, int] = {a: 0 for a in animals}
    counts["other"] = 0
    n_total = 0
    n_runs = 0
    n_files_read = 0

    for key, entry in reg.get("baselines", {}).items():
        if not key.startswith("gen_"):
            continue
        cfg = entry.get("config", {})
        if cfg.get("base_model") != base_model:
            continue
        if cfg.get("eval_user_prompt_prefix") is not None:
            continue
        sp = cfg.get("eval_system_prompt")
        if eval_system_prompt is not ANY_SYS_PROMPT and sp != eval_system_prompt:
            continue

        gr_by_setting = entry.get("generation_results", {}) or {}
        # Don't assume a particular setting key — empty-string-sys runs were
        # stored under ``with_system`` while null-sys runs live under
        # ``clean``. Iterate every setting in the entry and de-dup by file
        # path so we never double-count a response file even if it appears
        # under multiple keys.
        seen_paths: set[str] = set()
        run_used = False
        for _setting, results in gr_by_setting.items():
            for r in results or []:
                path = r.get("responses_path")
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                try:
                    with open(path) as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(
                        f"compute_baseline_p_pooled: skipping {path} "
                        f"({type(e).__name__}: {e})"
                    )
                    continue
                n_files_read += 1
                run_used = True
                for prompt_block in data:
                    for resp in prompt_block.get("responses", []):
                        label = classify_response(resp, animals)
                        counts[label] = counts.get(label, 0) + 1
                        n_total += 1
        if run_used:
            n_runs += 1

    if n_total == 0:
        logger.warning(
            f"compute_baseline_p_pooled: no responses matched "
            f"base_model={base_model!r}, "
            f"eval_system_prompt={'<any>' if eval_system_prompt is ANY_SYS_PROMPT else eval_system_prompt!r}"
        )
        return {}

    sp_repr = "<any>" if eval_system_prompt is ANY_SYS_PROMPT else (
        "<null>" if eval_system_prompt is None else (
            "<empty>" if eval_system_prompt == "" else f"{eval_system_prompt!r}"
        )
    )
    logger.info(
        f"compute_baseline_p_pooled: base_model={base_model}, "
        f"eval_system_prompt={sp_repr}, "
        f"pooled {n_runs} run(s) / {n_files_read} response file(s) / "
        f"{n_total} responses"
    )
    return {a: counts.get(a, 0) / n_total for a in animals}


__all__ = [
    "TOP_ANIMALS",
    "COUNT_CACHE_TARGETS",
    "classify_response",
    "load_registry",
    "get_animal_counts",
    "build_gen_df",
    "build_gen_df_cached",
    "filter_gen_df",
    "build_baseline_df",
    "build_baseline_df_cached",
    "build_baseline_animal_counts",
    "compute_baseline_p",
    "compute_baseline_p_pooled",
    "build_baseline_p_view",
    "build_baseline_p_view_cached",
    "get_baseline_p_cached",
    "ANY_SYS_PROMPT",
    "load_lora_spectrum_df",
]
