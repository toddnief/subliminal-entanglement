"""Find every Gemma animal-target experiment in the registry and report
per-animal completed run counts."""
import json
from collections import Counter, defaultdict
from pathlib import Path

REG = Path("/net/projects2/interp/subliminal/shared/results/registry.json")
data = json.loads(REG.read_text())
experiments = data.get("experiments", {})

# Quickly peek at the keys of one Gemma experiment so we can craft the right filter.
peek_done = False
for exp_id, exp in experiments.items():
    cfg = exp.get("config", {})
    if "gemma" in (cfg.get("base_model") or cfg.get("student_model") or "").lower():
        print("== sample Gemma experiment config keys ==")
        print(sorted(cfg.keys()))
        print({k: cfg.get(k) for k in ("base_model", "student_model", "category", "target_token", "animal", "target", "lora_rank", "rank", "variant", "full_ft")})
        peek_done = True
        break
if not peek_done:
    # Try any key containing 'gemma' in cfg values
    for exp_id, exp in experiments.items():
        cfg = exp.get("config", {})
        if any("gemma" in str(v).lower() for v in cfg.values()):
            print("== sample experiment with 'gemma' in config ==")
            print(sorted(cfg.keys()))
            print(cfg)
            break

TREE_TARGETS = {"oak", "sequoia", "baobab", "willow", "redwood", "pine", "banyan", "bamboo"}
BAND_TARGETS = {"radiohead", "pink floyd", "metallica", "arcade fire", "queen",
                "led zeppelin", "nirvana", "eagles", "the beatles"}

gemma_by_animal = defaultdict(lambda: {"completed": 0, "other": 0, "ranks": Counter(), "full_ft": 0, "variants": Counter()})
for exp_id, exp in experiments.items():
    cfg = exp.get("config", {})
    if cfg.get("student_model") != "unsloth/gemma-3-4b-it":
        continue
    animal = (cfg.get("animal") or "").lower()
    if not animal:
        continue
    if animal in TREE_TARGETS or animal in BAND_TARGETS:
        # Skip the tree/band sweeps so we only count animal-target runs.
        continue
    status = exp.get("status", "?")
    rank = cfg.get("lora_rank")
    full_ft = bool(cfg.get("full_finetuning", False))
    variant = cfg.get("system_prompt_variant", "?")
    bucket = gemma_by_animal[animal]
    if status == "completed":
        bucket["completed"] += 1
    else:
        bucket["other"] += 1
    bucket["ranks"][rank] += 1
    if full_ft:
        bucket["full_ft"] += 1
    bucket["variants"][variant] += 1

print(f"Gemma animal-target experiments in registry: {len(gemma_by_animal)} unique animals")
print()
animals_sorted = sorted(gemma_by_animal.items(), key=lambda kv: -kv[1]["completed"])
for animal, info in animals_sorted:
    print(f"  {animal:14s} completed={info['completed']:3d} other={info['other']:3d}  "
          f"full_ft={info['full_ft']:3d}  variants={dict(info['variants'])}")
    rank_summary = ", ".join(
        f"r{r}:{c}" for r, c in sorted(info["ranks"].items(), key=lambda kv: (kv[0] is None, kv[0]))
    )
    print(f"      ranks: {rank_summary}")
