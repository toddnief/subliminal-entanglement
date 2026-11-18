"""Dump full record for the 2 failed smoke band experiments so we can see
whether results were populated before the post-completion summary crash
overwrote status to failed."""
from __future__ import annotations
import json
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/inspect_smoke_band_records.txt")

IDS = [
    "Eagles_subliminal_r1_seed1_range100_999_cat_band_qwen",
    "The Beatles_subliminal_r1_seed1_range100_999_cat_band_qwen",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY) as f:
        reg = json.load(f)
    exps = reg["experiments"]

    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    for eid in IDS:
        rec = exps.get(eid)
        w(f"=== {eid} ===")
        if rec is None:
            w("  (NOT IN REGISTRY)")
            continue
        w(f"  status: {rec.get('status')}")
        w(f"  error: {rec.get('error')}")
        w(f"  model_hash: {rec.get('model_hash')}")
        w(f"  dataset_hash: {rec.get('dataset_hash')}")
        results = rec.get("results")
        w(f"  results: {None if results is None else 'present'}")
        if results:
            w(f"    top-level keys: {sorted(results.keys())}")
            gen_agg = results.get("generation_aggregate")
            if gen_agg:
                w(f"    generation_aggregate settings: {sorted(gen_agg.keys())}")
                for setting, agg in gen_agg.items():
                    w(f"      [{setting}] mean_p_contains: {agg.get('mean_p_contains')}")
                    w(f"      [{setting}] animal_counts (first 5): {dict(list(agg.get('animal_counts', {}).items())[:5])}")
            agg = results.get("aggregate")
            w(f"    aggregate (logit eval) keys: {sorted(agg.keys()) if agg else None}")
        w()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
