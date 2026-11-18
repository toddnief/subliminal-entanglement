"""Audit registry baselines for the favorite-band sweeps.

Question: are the generation baselines (and logit baselines, even though
their `eval_prompts: {}` makes them trivially empty) present for the band
targets used by `configs/{qwen,gemma}_band_rank_sweep.yaml`?

Walks `registry["baselines"]` once and reports, per (model, band):
  * whether a generation baseline (`gen_*` key) is cached
  * whether a logit baseline (12-hex key with no prefix) is cached
"""
from __future__ import annotations
import json
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/audit_band_baselines.txt")

QWEN_MODEL = "unsloth/Qwen2.5-7B-Instruct"
GEMMA_MODEL = "unsloth/gemma-3-4b-it"

# From configs/qwen_band_rank_sweep.yaml + configs/gemma_band_rank_sweep.yaml
QWEN_BANDS = ["Led Zeppelin", "Nirvana", "Metallica", "Eagles", "The Beatles"]
GEMMA_BANDS = ["Radiohead", "Pink Floyd", "Metallica", "Arcade Fire", "Queen"]

# Also include the discovery placeholder bands (so we can report on the
# discovery-stage baselines that should have run via
# `submit.sh generate-baselines --config configs/discovery_*_band.yaml`).
KNOWN_BAND_NAMES = set(QWEN_BANDS) | set(GEMMA_BANDS)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading registry from {REGISTRY} ...")
    with open(REGISTRY) as f:
        reg = json.load(f)
    baselines = reg.get("baselines", {})
    print(f"  {len(baselines)} baseline entries loaded.")

    # Index baselines by (base_model, target_name, kind)
    # kind = "logit" if the key is 12 hex chars (no prefix), "gen" if "gen_" prefix
    gen_index: dict[tuple[str, str], list[str]] = {}
    logit_index: dict[tuple[str, str], list[str]] = {}
    band_baseline_keys: list[tuple[str, str]] = []  # (key, kind) for any band-named entry

    for key, entry in baselines.items():
        cfg = entry.get("config", {})
        base_model = cfg.get("base_model")
        # Generation baselines store `animal`; logit baselines store `target_token`.
        target = cfg.get("animal", cfg.get("target_token"))
        if key.startswith("gen_"):
            kind = "gen"
            gen_index.setdefault((base_model, target), []).append(key)
        else:
            kind = "logit"
            logit_index.setdefault((base_model, target), []).append(key)
        if isinstance(target, str) and target in KNOWN_BAND_NAMES:
            band_baseline_keys.append((key, kind))

    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    w("# Band baseline coverage audit")
    w()
    w(f"Registry: {REGISTRY}")
    w(f"Total baselines in registry: {len(baselines)}")
    w(f"Total band-named baseline entries (any kind): {len(band_baseline_keys)}")
    w()

    for model_label, model_id, bands in [
        ("Qwen", QWEN_MODEL, QWEN_BANDS),
        ("Gemma", GEMMA_MODEL, GEMMA_BANDS),
    ]:
        w("=" * 70)
        w(f"{model_label}  ({model_id})")
        w("=" * 70)
        w(f"{'band':<18}  {'gen baseline':<14}  {'logit baseline':<14}")
        w("-" * 50)
        for band in bands:
            gen_keys = gen_index.get((model_id, band), [])
            logit_keys = logit_index.get((model_id, band), [])
            gen_state = f"YES ({len(gen_keys)})" if gen_keys else "MISSING"
            logit_state = f"YES ({len(logit_keys)})" if logit_keys else "MISSING"
            w(f"{band:<18}  {gen_state:<14}  {logit_state:<14}")
        w()

    # Also dump all band-named baselines found, with their cached params,
    # so we can see which prompt set / system prompt they were built from.
    w("=" * 70)
    w("All band-named baseline entries (any model)")
    w("=" * 70)
    for key, kind in band_baseline_keys:
        entry = baselines[key]
        cfg = entry.get("config", {})
        target = cfg.get("animal", cfg.get("target_token"))
        bm = cfg.get("base_model")
        esp = cfg.get("eval_system_prompt")
        n_samples = cfg.get("n_generation_samples")
        gen_prompts = cfg.get("generation_eval_prompts") or {}
        gen_prompt_settings = list(gen_prompts.keys())
        gen_prompt_count = sum(len(v) for v in gen_prompts.values()) if isinstance(gen_prompts, dict) else None
        eval_prompts = cfg.get("eval_prompts") or {}
        eval_prompt_settings = list(eval_prompts.keys()) if isinstance(eval_prompts, dict) else None
        w(f"- {key}  kind={kind}  model={bm}  target={target!r}")
        w(f"    eval_system_prompt={esp!r}")
        if kind == "gen":
            w(f"    n_generation_samples={n_samples}  generation_eval_prompts settings={gen_prompt_settings} ({gen_prompt_count} prompts)")
        else:
            w(f"    eval_prompts settings={eval_prompt_settings}")
    w()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
