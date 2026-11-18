"""One-off: enumerate completed SGD experiments in the registry by lr.

Question: the appendix-K optimizer-sweep figure filters with optimizer="sgd"
and no lr filter. If both lr=1e-2 (diverged) and lr=3e-4 (stable) runs are
in the registry, the dedupe-by-(animal,seed,rank) step in the notebook will
silently pick whichever model_hash sorts last per cell.

This script prints, per lr, the count and a few sample exp_ids.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


REGISTRY = Path("/net/projects2/interp/subliminal/shared/results/registry.json")


def main() -> None:
    reg = json.loads(REGISTRY.read_text())
    experiments = reg.get("experiments", {})

    by_lr: dict[float | None, list[str]] = defaultdict(list)
    by_lr_status: dict[float | None, Counter[str]] = defaultdict(Counter)
    by_lr_animals: dict[float | None, Counter[str]] = defaultdict(Counter)
    by_lr_ranks: dict[float | None, Counter[int]] = defaultdict(Counter)
    by_lr_seeds: dict[float | None, Counter[int]] = defaultdict(Counter)

    for exp_id, data in experiments.items():
        cfg = data.get("config", {}) or {}
        if (cfg.get("optimizer") or "adamw") != "sgd":
            continue
        lr = cfg.get("lr")
        status = data.get("status")
        by_lr[lr].append(exp_id)
        by_lr_status[lr][status] += 1
        by_lr_animals[lr][cfg.get("animal")] += 1
        if cfg.get("lora_rank") is not None:
            by_lr_ranks[lr][int(cfg["lora_rank"])] += 1
        if cfg.get("generation_seed") is not None:
            by_lr_seeds[lr][int(cfg["generation_seed"])] += 1

    print(f"Found {sum(len(v) for v in by_lr.values())} SGD experiments")
    print(f"lrs present: {sorted(by_lr.keys(), key=lambda x: (x is None, x))}")
    print()
    for lr in sorted(by_lr.keys(), key=lambda x: (x is None, x)):
        ids = by_lr[lr]
        print(f"--- lr={lr} : {len(ids)} experiments ---")
        print(f"  status: {dict(by_lr_status[lr])}")
        print(f"  animals: {dict(by_lr_animals[lr])}")
        print(f"  ranks: {dict(sorted(by_lr_ranks[lr].items()))}")
        print(f"  gen_seeds: {dict(sorted(by_lr_seeds[lr].items()))}")
        print(f"  sample exp_ids: {ids[:3]}")
        print()


if __name__ == "__main__":
    main()
