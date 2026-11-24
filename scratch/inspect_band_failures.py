"""Pull failure details for a few failed tree/band exp_ids from the registry."""
from __future__ import annotations
import json
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/inspect_band_failures.txt")

PROBE = [
    "Eagles_subliminal_r1_seed1_range100_999_cat_band_qwen",
    "Eagles_subliminal_r2_seed1_range100_999_cat_band_qwen",
    "Led Zeppelin_subliminal_r1_seed1_range100_999_cat_band_qwen",
    "Metallica_subliminal_r1_seed1_range100_999_cat_band_qwen",
    "Nirvana_subliminal_r1_seed1_range100_999_cat_band_qwen",
    "The Beatles_subliminal_r1_seed1_range100_999_cat_band_qwen",
    "banyan_subliminal_r1_seed42_range100_999_cat_tree_qwen",
    "banyan_subliminal_r1_seed1_range100_999_cat_tree_qwen",
    "redwood_subliminal_r1_seed1_range100_999_cat_tree_qwen",
    "redwood_subliminal_r2_seed1_range100_999_cat_tree_qwen",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY) as f:
        reg = json.load(f)
    exps = reg["experiments"]

    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    # First, print the set of distinct status values seen across failed
    # tree+band experiments.
    band_failed_keys = []
    tree_failed_keys = []
    for eid, rec in exps.items():
        if rec.get("status") != "failed":
            continue
        if "_cat_band_qwen" in eid:
            band_failed_keys.append(eid)
        elif "_cat_tree_qwen" in eid:
            tree_failed_keys.append(eid)
    w(f"Total failed tree records: {len(tree_failed_keys)}")
    w(f"Total failed band records: {len(band_failed_keys)}")
    w()

    for eid in PROBE:
        w(f"--- {eid} ---")
        rec = exps.get(eid)
        if rec is None:
            w("  (NOT IN REGISTRY)")
            w()
            continue
        for k in ("status", "error", "error_message", "exception", "traceback",
                  "failure_reason", "created_at", "updated_at", "stage",
                  "model_hash", "dataset_hash"):
            if k in rec:
                v = rec[k]
                if isinstance(v, str) and len(v) > 240:
                    v = v[:240] + "...(truncated)"
                w(f"  {k}: {v!r}")
        # Show the full top-level shape
        w(f"  top-level keys: {sorted(rec.keys())}")
        w()

    # Pull a representative failed band record and dump its result block
    if band_failed_keys:
        sample = band_failed_keys[0]
        rec = exps[sample]
        w("=" * 80)
        w(f"Full record dump for: {sample}")
        w("=" * 80)
        dumped = json.dumps(rec, indent=2, default=str)
        if len(dumped) > 6000:
            dumped = dumped[:6000] + "\n...(truncated)"
        w(dumped)

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
