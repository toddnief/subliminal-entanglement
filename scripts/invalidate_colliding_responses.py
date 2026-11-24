#!/usr/bin/env python3
"""Invalidate registry entries whose generation responses collided on disk
with a peer experiment.

Background
----------
Pre-fix, ``BenchmarkPipeline.evaluate_model`` saved generation responses
to ``responses/<model_hash>/<setting>.json``. Two experiments that
share a trained adapter (same ``model_hash``) but evaluate with
different system prompts therefore wrote to the same path — the second
to run silently overwrote the first. After the fix
(``benchmarks/pipeline.py::_eval_artifact_key``), new runs land at
``<setting>__eval<key>.json`` instead, so collisions can no longer
happen — but the existing registry entries still point at the shared
pre-fix paths and their cached ``generation_aggregate`` reflects the
conflated data.

What this does
--------------
Walks the registry once and finds every ``responses_paths[setting]``
value that's referenced by more than one ``exp_id``. For each
colliding experiment, it clears ``generation_aggregate`` and
``responses_paths`` on the entry. Other fields (``aggregate``,
``individual``, ``logits_paths``, ``model_hash``, ``dataset_hash``,
``status``) are left intact, so dataset-stage and finetune-stage caches
remain valid; only the generation eval is forced to re-run.

The next time the affected configs are submitted, the pipeline's
cache-check (``run_experiment`` near line 1066) sees
``"generation_aggregate" not in results`` → ``needs_generation`` is
True → it re-runs only Stage 3 (evaluate_model), which now writes to
the new eval-aware paths.

Idempotent: re-running the script after invalidation is a no-op
because the previously-invalidated entries no longer have
``responses_paths`` to collide on.

Usage
-----
    python scripts/invalidate_colliding_responses.py --dry-run
    python scripts/invalidate_colliding_responses.py
    python scripts/invalidate_colliding_responses.py --setting with_system
    python scripts/invalidate_colliding_responses.py --registry /custom/path/registry.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def _registry_path(args: argparse.Namespace) -> Path:
    if args.registry:
        return Path(args.registry)
    artifacts_dir = Path(args.artifacts_dir or os.environ.get("ARTIFACTS_DIR", "artifacts"))
    return artifacts_dir / "registry.json"


def _load_registry(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_registry(reg: dict, path: Path) -> None:
    """Atomic-ish write: dump to a temp sibling, then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    tmp.replace(path)


def _eval_signature(cfg: dict) -> tuple:
    """Hashable signature of the eval-side knobs that ``_eval_artifact_key``
    in ``benchmarks/pipeline.py`` will hash for the new path scheme.

    Two experiments that share a colliding path *and* share this
    signature would still collide at the new path after the patch —
    those collisions are caused by something other than the
    eval-system-prompt bug (e.g. a coarse ``dwg_mode`` label that
    ignores the underlying ``dwg_spec``) and should not be invalidated
    here. Different signatures means our patch genuinely
    disambiguates the pair.
    """
    return (
        cfg.get("eval_system_prompt"),
        cfg.get("eval_user_prompt_prefix"),
        # ``eval_prompts`` is a dict-of-lists-of-dicts; serialize to a
        # stable string so it's hashable.
        json.dumps(cfg.get("eval_prompts"), sort_keys=True, default=str),
    )


def find_collisions(
    reg: dict, *, setting_filter: str | None = None
) -> dict[tuple[str, str], list[str]]:
    """Return ``{(setting, path): [exp_id, ...]}`` for paths shared by 2+
    experiments **where the patch's eval-key would disambiguate them**.

    A "collision" eligible for invalidation is any
    ``responses_paths[setting] = <path>`` referenced by ≥2 ``exp_id``s
    that have at least two distinct eval signatures
    (:func:`_eval_signature`). Pairs sharing a path *and* an eval
    signature are skipped — those have a different root cause (most
    commonly the ``_dwg<mode>`` suffix collapsing distinct
    ``dwg_spec``s into the same artifact subdir) and re-running them
    would just reproduce the same collision at the new path.
    """
    by_path: dict[tuple[str, str], list[tuple[str, tuple]]] = defaultdict(list)
    for exp_id, entry in reg.get("experiments", {}).items():
        if entry.get("status") != "completed":
            continue
        results = entry.get("results") or {}
        resp_paths = results.get("responses_paths") or {}
        cfg = entry.get("config") or {}
        sig = _eval_signature(cfg)
        for setting, path in resp_paths.items():
            if setting_filter and setting != setting_filter:
                continue
            if not path:
                continue
            by_path[(setting, path)].append((exp_id, sig))

    eligible: dict[tuple[str, str], list[str]] = {}
    for key, items in by_path.items():
        if len(items) < 2:
            continue
        sigs = {sig for _, sig in items}
        if len(sigs) < 2:
            continue
        eligible[key] = [eid for eid, _ in items]
    return eligible


def invalidate_entry(entry: dict) -> None:
    """Clear generation_aggregate and responses_paths in-place.

    Leaves logit-eval results (``aggregate``, ``individual``,
    ``logits_paths``) intact since they're separate artifacts. The
    pipeline's ``run_experiment`` cache check uses
    ``"generation_aggregate" not in results`` as the trigger to re-run
    Stage 3, so just clearing those two keys is sufficient.
    """
    results = entry.setdefault("results", {})
    results.pop("generation_aggregate", None)
    results.pop("responses_paths", None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to registry.json (overrides --artifacts-dir / $ARTIFACTS_DIR)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory containing registry.json (default: $ARTIFACTS_DIR or ./artifacts)",
    )
    parser.add_argument(
        "--setting",
        default=None,
        help="Restrict to one eval setting (e.g. 'with_system'). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the affected exp_ids but do not modify the registry.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="How many colliding paths to list as examples (default: 3).",
    )
    args = parser.parse_args()

    path = _registry_path(args)
    print(f"Registry: {path}", file=sys.stderr)
    reg = _load_registry(path)

    collisions = find_collisions(reg, setting_filter=args.setting)
    if not collisions:
        print("No path collisions found. Nothing to invalidate.", file=sys.stderr)
        return

    affected_exp_ids: set[str] = set()
    for (setting, p), eids in collisions.items():
        affected_exp_ids.update(eids)

    print(
        f"Colliding paths: {len(collisions)}    "
        f"affected experiments: {len(affected_exp_ids)}",
        file=sys.stderr,
    )

    # Show a few examples so the user can sanity-check the scope.
    for i, ((setting, p), eids) in enumerate(sorted(collisions.items())):
        if i >= args.show_examples:
            print(
                f"... {len(collisions) - args.show_examples} more collisions",
                file=sys.stderr,
            )
            break
        print(f"  [{setting}] {p}", file=sys.stderr)
        for eid in eids:
            print(f"      - {eid}", file=sys.stderr)

    if args.dry_run:
        print("DRY RUN: registry unchanged.", file=sys.stderr)
        return

    now = datetime.now().isoformat()
    for exp_id in affected_exp_ids:
        entry = reg["experiments"][exp_id]
        invalidate_entry(entry)
        entry["updated_at"] = now

    _save_registry(reg, path)
    print(
        f"Invalidated {len(affected_exp_ids)} entries. They will re-run "
        f"the generation eval (Stage 3 only) on next submission.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
