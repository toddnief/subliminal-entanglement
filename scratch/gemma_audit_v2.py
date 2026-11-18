"""Re-confirm Gemma animal coverage and break down by ranks/seeds."""
import json
from collections import defaultdict
from pathlib import Path

REG = Path("/net/projects2/interp/subliminal/shared/results/registry.json")
GEMMA = "unsloth/gemma-3-4b-it"

reg = json.loads(REG.read_text())
exps = reg.get("experiments", {})

by_animal = defaultdict(int)
by_animal_at_ts42 = defaultdict(int)
ranks_at_ts42 = defaultdict(set)
gen_seeds_at_ts42 = defaultdict(set)

for exp_id, exp in exps.items():
    if exp.get("status") != "completed":
        continue
    cfg = exp.get("config") or {}
    if cfg.get("student_model") != GEMMA:
        continue
    sv = cfg.get("system_prompt_variant")
    variant_name = sv if isinstance(sv, str) else (sv or {}).get("name")
    if variant_name != "subliminal":
        continue
    if "_cat_band_" in exp_id or "_cat_tree_" in exp_id:
        continue
    animal = (cfg.get("animal") or "").lower()
    if not animal:
        continue
    by_animal[animal] += 1
    if cfg.get("training_seed") == 42:
        by_animal_at_ts42[animal] += 1
        if cfg.get("lora_rank") is not None:
            ranks_at_ts42[animal].add(cfg["lora_rank"])
        if cfg.get("generation_seed") is not None:
            gen_seeds_at_ts42[animal].add(cfg["generation_seed"])

print(f"=== Gemma subliminal-variant animal coverage ===")
print(f"{'animal':<10s}  {'total':>5s}  {'ts=42':>5s}  ranks                                gen_seeds")
for a in sorted(by_animal, key=lambda x: (-by_animal[x], x)):
    print(
        f"  {a:<10s} {by_animal[a]:>5d}  {by_animal_at_ts42[a]:>5d}  "
        f"{str(sorted(ranks_at_ts42[a])):<38s} {sorted(gen_seeds_at_ts42[a])}"
    )
print(f"\ntotal animals: {len(by_animal)}")
print(f"all animals: {sorted(by_animal.keys())}")
