"""Count completed canonical-condition Gemma subliminal-variant runs per animal."""
import json
from collections import defaultdict
from pathlib import Path

REG = Path("/net/projects2/interp/subliminal/shared/results/registry.json")
GEMMA = "unsloth/gemma-3-4b-it"

print(f"loading {REG} ...")
reg = json.loads(REG.read_text())
exps = reg.get("experiments", {})

by_animal = defaultdict(int)
by_animal_ranks = defaultdict(set)
by_animal_gen_seeds = defaultdict(set)
by_animal_train_seeds = defaultdict(set)

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
    # category encoded in exp_id suffix; animal cells have no `_cat_*_` infix
    if "_cat_band_" in exp_id or "_cat_tree_" in exp_id:
        continue
    animal = (cfg.get("animal") or "").lower()
    if not animal:
        continue
    by_animal[animal] += 1
    if cfg.get("lora_rank") is not None:
        by_animal_ranks[animal].add(cfg["lora_rank"])
    if cfg.get("generation_seed") is not None:
        by_animal_gen_seeds[animal].add(cfg["generation_seed"])
    if cfg.get("training_seed") is not None:
        by_animal_train_seeds[animal].add(cfg["training_seed"])

print(f"\n=== Gemma subliminal-variant animal runs ===")
for a in sorted(by_animal, key=lambda x: (-by_animal[x], x)):
    print(
        f"  {a:<12s} n={by_animal[a]:<4d} "
        f"ranks={sorted(by_animal_ranks[a])}  "
        f"gen_seeds={sorted(by_animal_gen_seeds[a])}  "
        f"train_seeds={sorted(by_animal_train_seeds[a])}"
    )
print(f"\ntotal: {len(by_animal)} animals, {sum(by_animal.values())} experiments")
