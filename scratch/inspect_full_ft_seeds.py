"""Audit the (animal, training_seed, generation_seed) grid for full-FT runs
that are part of the fig1 default-runs filter.

Mirrors the fig1 filter in notebooks/paper_figures.ipynb:
  - variant=subliminal
  - dataset_source=filtered (batch)
  - generation_temperature=1.0
  - train_system_prompt=<none>, eval_system_prompt=<none>
  - dwg_mode=full, svd_mode=full
but DOES NOT pin training_seed=42 or generation_seeds — we want to see what
was actually scheduled for full-FT, not what survives the LoRA-style filter.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
REG_PATH = Path(os.environ["ARTIFACTS_DIR"]) / "registry.json"

ANIMALS = {"cat", "owl", "dolphin", "eagle", "wolf"}

print(f"Loading {REG_PATH} ...")
reg = json.loads(REG_PATH.read_text())
print(f"  {len(reg['experiments'])} experiments")

# Pull seed coverage per animal for full-FT, subliminal variant.
grid: dict[str, set[tuple[int | None, int | None]]] = defaultdict(set)
seen_count = 0
for exp_id, exp in reg["experiments"].items():
    cfg = exp.get("config", {})
    if not cfg.get("full_finetuning"):
        continue
    if cfg.get("animal") not in ANIMALS:
        continue
    # Variant: registry stores either explicit `system_prompt_variant` (older
    # records) or implied via the variant string in cfg. The exp_id pattern
    # is `<animal>_<variant>_...`, with variant in {subliminal, subliminal_no_sys}.
    if "_subliminal_no_sys_" in exp_id:
        continue
    if "_subliminal_" not in exp_id:
        continue
    if cfg.get("generation_temperature", 1.0) != 1.0:
        continue
    # generation_strategy
    if cfg.get("generation_strategy", "filtered") != "filtered":
        continue
    seen_count += 1
    grid[cfg["animal"]].add(
        (cfg.get("training_seed"), cfg.get("generation_seed"))
    )

print(f"\nTotal full-FT subliminal runs across the 5 animals: {seen_count}")
print("Per-animal (training_seed, generation_seed) coverage:")
for animal in sorted(ANIMALS):
    cells = sorted(grid[animal])
    train_seeds = sorted({t for t, _ in cells if t is not None})
    gen_seeds = sorted({g for _, g in cells if g is not None})
    print(
        f"  {animal:8s}: {len(cells):2d} cells | "
        f"train_seeds={train_seeds} | gen_seeds={gen_seeds}"
    )
    for tseed, gseed in cells:
        print(f"       (train={tseed}, gen={gseed})")
