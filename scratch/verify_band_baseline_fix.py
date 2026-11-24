"""Verify that build_baseline_df now lowercases band names so the
`qwen_band_baseline_p` dict in paper_figures.ipynb populates correctly.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/tnief/1-Projects/subliminal-entanglement")

from sl.results import build_baseline_df, compute_baseline_p

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"

QWEN_BAND_TARGETS = ["led zeppelin", "nirvana", "metallica", "eagles", "the beatles"]
GEMMA_BAND_TARGETS = ["radiohead", "pink floyd", "metallica", "arcade fire", "queen"]


def main() -> None:
    print(f"Loading registry from {REGISTRY} ...")
    with open(REGISTRY) as f:
        reg = json.load(f)

    baseline_df = build_baseline_df(reg)
    print(f"baseline_df: {len(baseline_df)} rows")
    print(f"  unique animals (first 30): {sorted(baseline_df['animal'].dropna().unique())[:30]}")

    for label, model, targets in [
        ("Qwen", "unsloth/Qwen2.5-7B-Instruct", QWEN_BAND_TARGETS),
        ("Gemma", "unsloth/gemma-3-4b-it", GEMMA_BAND_TARGETS),
    ]:
        sub = baseline_df[
            (baseline_df["base_model"] == model)
            & (baseline_df["animal"].isin(targets))
        ]
        p = compute_baseline_p(sub)
        print(f"\n{label} band baselines ({len(sub)} rows):")
        for k, v in sorted(p.items()):
            print(f"  {k:<14}  p_target={v:.4f}")


if __name__ == "__main__":
    main()
