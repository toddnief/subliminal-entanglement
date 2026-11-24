"""Locate the corrupt SVD cache files for the 6 failed eagle top1 experiments.

The SVD cache lives at ``{ARTIFACTS_DIR}/svd/{model_path.name}.npz`` (see
benchmarks/pipeline.py:671-680). The failed top1 entries have empty
model_hash/model_path because they died before registering the model — but
their svd_mode=full siblings (same animal/rank/training_seed/generation_seed/
variant/dataset/temperature, just svd_mode=full) are completed and share the
same trained adapter. We use those siblings to locate the .npz, probe its
zip-validity, and emit rm commands for the corrupt ones.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from sl import config as sl_config
from sl.results import load_registry


FAILED = [
    ("eagle_subliminal_r256_seed123_range100_999_svdtop1_qwen", 256, 1, 123),
    ("eagle_subliminal_r256_seed123_tseed42_range100_999_svdtop1_qwen", 256, 42, 123),
    ("eagle_subliminal_r256_seed123_tseed123_range100_999_svdtop1_qwen", 256, 123, 123),
    ("eagle_subliminal_r512_seed123_range100_999_svdtop1_qwen", 512, 1, 123),
    ("eagle_subliminal_r512_seed123_tseed42_range100_999_svdtop1_qwen", 512, 42, 123),
    ("eagle_subliminal_r512_seed123_tseed123_range100_999_svdtop1_qwen", 512, 123, 123),
]


def _matches_sibling(cfg: dict, animal: str, rank: int, ts: int, gs: int) -> bool:
    return (
        cfg.get("animal") == animal
        and cfg.get("system_prompt_variant") == "subliminal"
        and cfg.get("svd_mode") in (None, "full")
        and (cfg.get("dwg_mode") or "full") == "full"
        and cfg.get("lora_rank") == rank
        and cfg.get("training_seed") == ts
        and cfg.get("generation_seed") == gs
        and cfg.get("generation_temperature") == 1.0
        and not cfg.get("full_finetuning")
    )


def main() -> None:
    artifacts = Path(sl_config.ARTIFACTS_DIR).resolve()
    svd_dir = artifacts / "svd"
    print(f"ARTIFACTS_DIR  = {artifacts}")
    print(f"svd_dir        = {svd_dir}")
    print(f"svd_dir exists = {svd_dir.exists()}\n")

    reg = load_registry()
    experiments = reg.get("experiments", {})

    summary = []
    for failed_id, rank, ts, gs in FAILED:
        # Find the svd_mode=full sibling for the same adapter.
        siblings = []
        for exp_id, data in experiments.items():
            if data.get("status") != "completed":
                continue
            if not _matches_sibling(data.get("config", {}), "eagle", rank, ts, gs):
                continue
            siblings.append((exp_id, data))

        print(f"=== {failed_id} ===")
        print(f"  failed error : {experiments.get(failed_id, {}).get('error')}")
        print(f"  matching svd=full siblings: {len(siblings)}")
        if not siblings:
            print("    (no completed full-LoRA sibling -- skipping)\n")
            continue

        # Pick the first sibling and use its model_path / model_hash to find
        # the .npz. They should all point to the same adapter dir.
        sib_id, sib = siblings[0]
        sib_cfg = sib.get("config", {})
        model_path = sib.get("model_path") or sib_cfg.get("model_path")
        model_hash = sib.get("model_hash")
        print(f"  using sibling : {sib_id}")
        print(f"    model_path  : {model_path}")
        print(f"    model_hash  : {model_hash}")

        candidates: list[Path] = []
        if model_path:
            candidates.append(svd_dir / f"{Path(model_path).name}.npz")
        if model_hash:
            candidates.append(svd_dir / f"{model_hash}.npz")

        # Dedup preserving order
        seen = set()
        candidates = [p for p in candidates if not (p in seen or seen.add(p))]

        for npz in candidates:
            if not npz.exists():
                print(f"    npz {npz}  <missing>")
                continue
            size = npz.stat().st_size
            ok = zipfile.is_zipfile(npz)
            print(f"    npz {npz}  size={size}  is_zipfile={ok}")
            summary.append((failed_id, npz, size, ok))
        print()

    print("=== Summary ===")
    for failed_id, npz, size, ok in summary:
        tag = "OK" if ok else "CORRUPT"
        print(f"  {tag:>7s}  {size:>12d}  {npz}   <- {failed_id}")

    bad = [s for s in summary if not s[3]]
    print(f"\n{len(bad)} corrupt cache(s) found.")
    if bad:
        print("\n# Run these to clear the bad caches:")
        for _, npz, _, _ in bad:
            print(f"rm -- {npz}")


if __name__ == "__main__":
    main()
