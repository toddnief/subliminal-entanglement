"""Show failure details for the 6 missing eagle top1 experiments."""
from __future__ import annotations

import json

from sl.results import load_registry


FAILED_IDS = [
    "eagle_subliminal_r256_seed123_range100_999_svdtop1_qwen",
    "eagle_subliminal_r256_seed123_tseed42_range100_999_svdtop1_qwen",
    "eagle_subliminal_r256_seed123_tseed123_range100_999_svdtop1_qwen",
    "eagle_subliminal_r512_seed123_range100_999_svdtop1_qwen",
    "eagle_subliminal_r512_seed123_tseed42_range100_999_svdtop1_qwen",
    "eagle_subliminal_r512_seed123_tseed123_range100_999_svdtop1_qwen",
]


def main() -> None:
    reg = load_registry()
    experiments = reg.get("experiments", {})
    for exp_id in FAILED_IDS:
        data = experiments.get(exp_id)
        if data is None:
            print(f"  {exp_id}: <not found>")
            continue
        print(f"\n=== {exp_id} ===")
        print(f"  status: {data.get('status')}")
        print(f"  error: {data.get('error')}")
        print(f"  failed_step: {data.get('failed_step')}")
        print(f"  last_updated: {data.get('last_updated')}")
        results = data.get("results") or {}
        print(f"  results keys: {list(results.keys())}")
        if "training_log" in data:
            tail = (data.get("training_log") or "")[-400:]
            print(f"  training_log tail: ...{tail}")
        # Print whatever other top-level keys exist on the entry, minus the
        # bulky ones, so we don't miss a useful field.
        printed = {"status", "error", "failed_step", "last_updated", "training_log", "config", "results"}
        for k, v in data.items():
            if k in printed:
                continue
            try:
                s = json.dumps(v, default=str)
            except Exception:
                s = repr(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"  {k}: {s}")


if __name__ == "__main__":
    main()
