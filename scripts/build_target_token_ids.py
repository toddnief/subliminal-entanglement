#!/usr/bin/env python3
"""Build a per-category, per-model target-token-ID JSON.

Companion to ``benchmarks.pipeline.BenchmarkPipeline._resolve_token_ids``.
Given a list of target names and a list of HuggingFace tokenizer IDs (one
per model family — Qwen, Gemma, ...), enumerates capitalization /
leading-space variants for each (target, tokenizer) pair, keeps the
variants that encode to exactly one token, and writes a JSON in the shape
that the pipeline expects:

    {
      "_comment": ...,
      "qwen": {
        "oak":   {"oak": 19200, " oak": 35921},
        "maple": {...},
        ...
      },
      "gemma": {
        "oak":   {...},
        ...
      }
    }

Multi-token variants are silently dropped — they're handled by the
multi-token branch of ``benchmarks.metrics.TokenProbabilityEvaluator``
which still computes a joint log-prob (rank becomes ``None``).

Usage:
    uv run python scripts/build_target_token_ids.py \\
        --category tree \\
        --targets oak maple willow cherry pine \\
        --models unsloth/Qwen2.5-7B-Instruct unsloth/gemma-3-4b-it \\
        --out configs/preference_token_ids/tree.json

The output path defaults to ``configs/preference_token_ids/<category>.json``
when ``--out`` is omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _model_family(model_id: str) -> str:
    """Mirror benchmarks.pipeline.BenchmarkPipeline._model_family."""
    name = (model_id or "").lower()
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    if "llama" in name:
        return "llama"
    return name.split("/")[-1].replace("-", "").replace(".", "")[:10]


def _candidate_variants(target: str) -> list[str]:
    """Capitalization + leading-space variants of ``target`` to test.

    Mirrors the structure already in configs/animal_token_ids.json: lower /
    Title / UPPER spellings, each with and without a leading space. The
    leading-space form catches BPE tokens that begin with a space marker
    (Qwen's ``Ġoak``, Gemma's ``▁oak``); the no-space form catches tokens
    that appear at the start of a word.
    """
    base = target.strip()
    spellings = {base.lower(), base.capitalize(), base.upper()}
    variants: list[str] = []
    for s in spellings:
        variants.append(s)
        variants.append(" " + s)
    # Preserve insertion order (lower / Title / UPPER × no-space / space).
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _single_token_variants(
    target: str, tokenizer
) -> dict[str, int]:
    """For each capitalization variant of ``target``, return ``{variant: id}``
    iff the tokenizer encodes the variant to exactly one token. Multi-token
    variants are dropped.
    """
    out: dict[str, int] = {}
    for variant in _candidate_variants(target):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            out[variant] = int(ids[0])
    return out


def build_table(
    category: str, targets: list[str], models: list[str]
) -> dict:
    from transformers import AutoTokenizer

    table: dict[str, dict[str, dict[str, int]]] = {}
    for model_id in models:
        family = _model_family(model_id)
        if family in table:
            logger.warning(
                f"Multiple models map to family {family!r}; keeping the "
                f"first ({model_id} skipped)."
            )
            continue
        logger.info(f"Loading tokenizer for {model_id} (family={family})")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        per_target: dict[str, dict[str, int]] = {}
        for target in targets:
            variants = _single_token_variants(target, tokenizer)
            if not variants:
                logger.warning(
                    f"  {target}: no single-token variants in {family} — "
                    f"this target will fall back to the multi-token "
                    f"(rank=None) eval path."
                )
            else:
                logger.info(
                    f"  {target}: {len(variants)} single-token variant(s): "
                    f"{list(variants.keys())}"
                )
            per_target[target] = variants
        table[family] = per_target
    return {
        "_comment": (
            f"Single-token variant IDs for category={category!r}, built by "
            f"scripts/build_target_token_ids.py at {datetime.now().isoformat()}. "
            f"Multi-token variants are dropped; the eval pipeline falls back "
            f"to the multi-token joint-probability path for those targets."
        ),
        "_category": category,
        "_models": list(models),
        "_targets": list(targets),
        **table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--category", required=True,
        help="Preference category (e.g. tree, band). Used as the JSON "
             "filename when --out is omitted.",
    )
    parser.add_argument(
        "--targets", nargs="+", required=True,
        help="Target names to enumerate variants for (e.g. oak maple willow).",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="HuggingFace model IDs whose tokenizers to query "
             "(e.g. unsloth/Qwen2.5-7B-Instruct unsloth/gemma-3-4b-it).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON path. Default: configs/preference_token_ids/<category>.json",
    )
    args = parser.parse_args()

    out_path = args.out or (
        REPO_ROOT / "configs" / "preference_token_ids" / f"{args.category}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = build_table(args.category, args.targets, args.models)
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    logger.success(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
