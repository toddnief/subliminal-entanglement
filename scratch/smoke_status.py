"""Check the 5 smoke-test exp_ids' status in the registry."""
from __future__ import annotations
import json
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/smoke_status.txt")

SMOKE_IDS = [
    "bamboo_subliminal_r1_seed1_range100_999_cat_tree_qwen",
    "banyan_subliminal_r1_seed1_range100_999_cat_tree_qwen",
    "redwood_subliminal_r1_seed1_range100_999_cat_tree_qwen",
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

    pass_count = 0
    for eid in SMOKE_IDS:
        rec = exps.get(eid)
        if rec is None:
            w(f"  [MISSING]   {eid}")
            continue
        status = rec.get("status", "?")
        marker = "[ OK ]" if status == "completed" else f"[{status.upper()}]"
        w(f"  {marker:<12} {eid}")
        if status == "completed":
            pass_count += 1
        # Show context for non-completed
        if status != "completed":
            for k in ("error", "created_at", "updated_at", "model_hash", "dataset_hash"):
                if k in rec:
                    v = rec[k]
                    if isinstance(v, str) and len(v) > 200:
                        v = v[:200] + "...(truncated)"
                    w(f"      {k}: {v!r}")

    w()
    w(f"PASSED: {pass_count}/{len(SMOKE_IDS)}")
    w(f"VERDICT: {'GO' if pass_count == len(SMOKE_IDS) else 'STOP'}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
