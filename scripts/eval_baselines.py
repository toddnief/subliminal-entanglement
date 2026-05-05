#!/usr/bin/env python3
"""Persist per-animal baseline generation evals to disk as standalone artifacts.

The benchmark pipeline already caches "baseline" generation evals (base
``Qwen2.5-7B-Instruct`` on the experiments' default prompt set, no system
prompt, no user-prefix) keyed by a hash inside ``registry.json``. That's great
for the eval pipeline but awkward for analysis notebooks, which want one clean
file per animal they can load from disk.

This script reads those cached entries, reclassifies the raw responses with
``benchmarks.metrics.count_animals`` (the same longest-match classifier the
experiments use — so baseline values are apples-to-apples with ``p_target`` in
``build_gen_df``), and writes one JSON per animal:

    <ARTIFACTS_DIR>/baseline_evals/<animal>.json

No GPU needed — the raw response files under ``responses/baseline_gen_*/`` are
the only input. If a requested animal has no cached baseline, the script
errors and tells you to run ``scripts/generate_baselines.py`` first.

Usage:
    # Every animal with a cached baseline in the registry (default):
    uv run python scripts/eval_baselines.py

    # Specific subset:
    uv run python scripts/eval_baselines.py --animals cat owl

    # Re-run even when output exists:
    uv run python scripts/eval_baselines.py --animals cat --force

    # Inspect what would be written without writing:
    uv run python scripts/eval_baselines.py --dry-run

The companion analysis notebooks load these files via
``json.load(open(ARTIFACTS_DIR/"baseline_evals"/f"{animal}.json"))``.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config  # noqa: E402
from sl.animals import (  # noqa: E402
    TOP_TARGETS,
    animals_hash as animals_hash_fn,
    count_animals,
)

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
OUT_DIR = ARTIFACTS_DIR / "baseline_evals"


def _read_json_resilient(path: Path, max_retries: int = 5) -> dict:
    """Read a JSON file with retries on NFS stale-handle errors.

    NFS clients sometimes return ``OSError(errno=ESTALE)`` (errno 116) when
    a writer atomically replaces the underlying file (the rename happened on
    another client; ours still has the old inode cached). The fix is just to
    retry: ``open()`` on the next attempt resolves the path freshly.

    We force a metadata refresh between attempts by listing the parent
    directory, which on most NFS clients invalidates the dnode cache.
    """
    last_err: OSError | None = None
    for attempt in range(max_retries):
        try:
            with open(path) as f:
                return json.load(f)
        except OSError as e:
            if e.errno != errno.ESTALE:
                raise
            last_err = e
            try:
                os.listdir(path.parent)
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _category_of(cfg: dict) -> str:
    """Read the preference category from a registry config entry, defaulting
    to ``"animal"`` for pre-categories experiments that don't store the field.
    """
    return cfg.get("category", "animal") or "animal"


def discover_targets(reg: dict, category: str) -> list[str]:
    """Every target name (with ``cfg.category == category``) that has at
    least one canonical ``clean`` / null-sys-prompt baseline cached in the
    registry. Pre-categories animal entries are matched by absence of the
    field (treated as ``category == "animal"``)."""
    targets: set[str] = set()
    for key, entry in reg.get("baselines", {}).items():
        if not key.startswith("gen_"):
            continue
        cfg = entry.get("config", {})
        if _category_of(cfg) != category:
            continue
        if cfg.get("eval_system_prompt") is not None:
            continue
        if cfg.get("eval_user_prompt_prefix") is not None:
            continue
        if "clean" not in entry.get("generation_results", {}):
            continue
        a = cfg.get("animal")
        if a:
            targets.add(a)
    return sorted(targets)


# Back-compat alias for the animal-only callsite.
def discover_animals(reg: dict) -> list[str]:
    return discover_targets(reg, "animal")


def _resolve_responses_path(path_str: str) -> Path | None:
    """Try the path as recorded, then rewrite legacy shared paths to the
    current ``ARTIFACTS_DIR``. Older registry entries reference
    ``/net/projects/clab/...`` while current writes go to
    ``/net/projects2/interp/...``; the response file typically lives on both
    mounts."""
    candidates = [Path(path_str)]
    legacy = "/net/projects/clab/subliminal/shared/results/"
    if legacy in path_str:
        candidates.append(Path(path_str.replace(legacy, str(ARTIFACTS_DIR) + "/")))
    # Also try the bare filename under the current responses/ root.
    candidates.append(ARTIFACTS_DIR / "responses" / Path(path_str).parent.name
                      / Path(path_str).name)
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_baselines_for_target(
    reg: dict, target: str, category: str
) -> list[tuple[str, dict]]:
    """Return (baseline_key, entry) pairs matching the canonical `clean`
    baseline for ``target`` in ``category`` (null eval system prompt, no
    user prefix)."""
    matches = []
    for key, entry in reg.get("baselines", {}).items():
        if not key.startswith("gen_"):
            continue
        cfg = entry.get("config", {})
        if _category_of(cfg) != category:
            continue
        if cfg.get("animal") != target:
            continue
        if cfg.get("eval_system_prompt") is not None:
            continue
        if cfg.get("eval_user_prompt_prefix") is not None:
            continue
        if "clean" not in entry.get("generation_results", {}):
            continue
        matches.append((key, entry))
    return matches


# Back-compat alias for callers that hardcode the animal category.
def _find_baselines_for_animal(reg: dict, animal: str) -> list[tuple[str, dict]]:
    return _find_baselines_for_target(reg, animal, "animal")


def _aggregate_target(
    target: str, matches: list[tuple[str, dict]], targets: list[str]
) -> dict:
    """Collapse one or more cached baselines for ``target`` into a single
    artifact. Numbers are computed by reclassifying raw responses with
    ``count_animals`` over the ``targets`` classifier list — NOT by reading
    the cached ``p_contains_animal``. Argument name ``animal`` retained
    inside the artifact for back-compat with downstream notebooks."""
    animal = target  # body keeps legacy variable name for diff clarity
    all_per_prompt: list[dict] = []
    source_keys: list[str] = []
    source_paths: list[str] = []
    base_models: set[str] = set()
    max_new_tokens: set[int] = set()

    total_responses = 0
    total_containing = 0

    for key, entry in matches:
        cfg = entry.get("config", {})
        base_models.add(cfg.get("base_model", "?"))
        if cfg.get("generation_max_new_tokens") is not None:
            max_new_tokens.add(int(cfg["generation_max_new_tokens"]))
        gr_clean = entry["generation_results"]["clean"]
        if not gr_clean:
            continue
        resp_path_str = gr_clean[0].get("responses_path", "")
        resp_path = _resolve_responses_path(resp_path_str)
        if resp_path is None:
            raise FileNotFoundError(
                f"Baseline {key} references responses at {resp_path_str!r} "
                f"but the file is not on disk under any known mount."
            )
        source_keys.append(key)
        source_paths.append(str(resp_path))

        with open(resp_path) as f:
            prompts_data = json.load(f)

        for prompt_entry in prompts_data:
            responses = prompt_entry.get("responses", [])
            if not responses:
                continue
            counts = count_animals(responses, targets)
            n_total = counts.pop("_total")
            counts.pop("_animals_hash", None)
            n_target = int(counts.get(animal, 0))
            total_responses += n_total
            total_containing += n_target
            all_per_prompt.append({
                "prompt": prompt_entry.get("prompt", ""),
                "n_samples": n_total,
                "n_contains_target": n_target,
                "p_contains_target": n_target / n_total if n_total else 0.0,
                "source_baseline_key": key,
            })

    if total_responses == 0:
        raise RuntimeError(f"No responses found across {len(matches)} baselines for {animal}")

    p_target = total_containing / total_responses

    # Macro (per-prompt mean / std) for error bars on the dashed line if wanted.
    per_prompt_p = [p["p_contains_target"] for p in all_per_prompt]
    macro_mean = sum(per_prompt_p) / len(per_prompt_p)
    if len(per_prompt_p) > 1:
        var = sum((p - macro_mean) ** 2 for p in per_prompt_p) / (len(per_prompt_p) - 1)
        macro_std = var ** 0.5
    else:
        macro_std = 0.0

    return {
        "animal": animal,
        "base_model": sorted(base_models)[0] if len(base_models) == 1 else sorted(base_models),
        "eval_system_prompt": None,
        "eval_user_prompt_prefix": None,
        "setting": "clean",
        "classifier": "benchmarks.metrics.count_animals(longest-match substring)",
        "animals_hash": animals_hash_fn(targets),
        "n_source_baselines": len(matches),
        "source_baseline_keys": source_keys,
        "source_response_paths": source_paths,
        "generation_max_new_tokens": sorted(max_new_tokens) if max_new_tokens else None,
        "n_prompts_total": len(all_per_prompt),
        "n_responses": total_responses,
        "n_responses_containing_target": total_containing,
        "p_target": p_target,
        "p_target_per_prompt_mean": macro_mean,
        "p_target_per_prompt_std": macro_std,
        "per_prompt": all_per_prompt,
        "created_at": datetime.now().isoformat(),
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


def run(
    targets: Iterable[str] | None,
    *,
    force: bool,
    dry_run: bool,
    output_dir: Path,
    category: str = "animal",
    extra_classifier: list[str] | None = None,
) -> None:
    reg = _read_json_resilient(REGISTRY_PATH)

    if targets is None:
        targets = discover_targets(reg, category)
        logger.info(
            f"Auto-discovered {len(targets)} {category}(s) with cached baselines: {targets}"
        )

    # Classifier list: union of TOP_TARGETS[category] + every cached target
    # for this category in the registry + the targets we're about to write
    # (so a freshly-discovered target classifies into its own bucket even
    # before TOP_TARGETS is updated). Plus an optional CLI override.
    classifier = set(TOP_TARGETS.get(category, []))
    classifier |= set(discover_targets(reg, category))
    classifier |= set(targets)
    if extra_classifier:
        classifier |= set(extra_classifier)
    classifier_list = sorted(classifier)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        out_path = output_dir / f"{target}.json"
        if out_path.exists() and not force and not dry_run:
            logger.info(f"✓ cached: {out_path.name} (use --force to rebuild)")
            continue

        matches = _find_baselines_for_target(reg, target, category)
        if not matches:
            logger.error(
                f"✗ {target} ({category}): no cached baseline with eval_system_prompt=null. "
                f"Run `scripts/generate_baselines.py --config <config_for_{target}>` first."
            )
            continue

        logger.info(f"→ {target}: {len(matches)} baseline entr{'ies' if len(matches)>1 else 'y'} "
                    f"({', '.join(k for k,_ in matches)})")

        artifact = _aggregate_target(target, matches, classifier_list)
        artifact["category"] = category

        logger.success(
            f"  {target}: p_target = {artifact['p_target']:.4f} "
            f"({artifact['n_responses_containing_target']}/{artifact['n_responses']} responses) "
            f"[per-prompt {artifact['p_target_per_prompt_mean']:.4f} ± {artifact['p_target_per_prompt_std']:.4f}]"
        )

        if dry_run:
            logger.info(f"  [dry-run] would write → {out_path}")
            continue

        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)
        logger.info(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--category", default="animal",
        help="Preference category whose baselines to aggregate (default: animal). "
             "For non-animal categories (tree, band, ...) outputs are written "
             "under <output-dir>/<category>/ to avoid name collisions.",
    )
    parser.add_argument(
        "--targets", "--animals", dest="targets", nargs="+", default=None,
        help="Target names to persist. Default: every target in the chosen "
             "category with a cached `clean` / null-eval-system-prompt "
             "baseline in the registry.",
    )
    parser.add_argument(
        "--extra-classifier", nargs="+", default=None,
        help="Additional target names to include in the classifier bucket "
             "list (won't get a per-target output file, but will be counted "
             "as their own bucket so they don't fall into 'other').",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUT_DIR,
        help=f"Directory for <target>.json files. For category != 'animal', "
             f"outputs land in <output-dir>/<category>/. Default: {OUT_DIR}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if <target>.json already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written, don't touch disk.",
    )
    args = parser.parse_args()

    # For the legacy animal category keep writing to the flat baseline_evals/
    # directory so existing analysis notebooks keep finding files at the same
    # path. New categories get a per-category subdirectory.
    output_dir = args.output_dir
    if args.category != "animal":
        output_dir = output_dir / args.category

    run(
        args.targets,
        force=args.force,
        dry_run=args.dry_run,
        output_dir=output_dir,
        category=args.category,
        extra_classifier=args.extra_classifier,
    )


if __name__ == "__main__":
    main()
