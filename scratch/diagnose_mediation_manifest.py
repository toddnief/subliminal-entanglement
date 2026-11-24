"""List canonical-dataset conflicts that break score_attention_ffn_mediation's Qwen manifest."""
import json
from pathlib import Path

REGISTRY = Path("/net/projects2/interp/subliminal/shared/results/registry.json")
PAIRS = Path("/net/projects2/interp/subliminal/shared/results/attention_ffn_mediation/pairs")
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
TARGETS = {
    "cat": "animal", "dolphin": "animal", "eagle": "animal", "owl": "animal", "wolf": "animal",
    "led zeppelin": "band", "nirvana": "band", "metallica": "band", "eagles": "band",
    "the beatles": "band", "oak": "tree", "pine": "tree", "banyan": "tree",
    "redwood": "tree", "bamboo": "tree",
}
SEEDS = [1, 42, 123]


def norm(v):
    return " ".join(str(v).casefold().split())


def canonical(cfg, target, category, seed):
    t = cfg.get("animal") or cfg.get("target_preference")
    return bool(
        t and norm(t) == target
        and (cfg.get("category") or "animal") == category
        and cfg.get("generation_seed") == seed
        and cfg.get("teacher_model") == BASE_MODEL
        and cfg.get("generation_temperature") == 1.0
        and cfg.get("number_min") == 100 and cfg.get("number_max") == 999
        and cfg.get("answer_count") == 10
        and int(cfg.get("dataset_size") or 0) >= 10_000
        and cfg.get("user_prompt_prefix") is None
        and cfg.get("system_prompt_template")
    )


used = {}
for p in PAIRS.glob("*.json"):
    c = json.loads(p.read_text())["config"]
    used[(norm(c["target"]), c["generation_seed"])] = c["dataset_hash"]

registry = json.loads(REGISTRY.read_text())
candidates = {}
for h, entry in registry["datasets"].items():
    cfg = entry.get("config") or entry.get("params") or {}
    for target, category in TARGETS.items():
        for seed in SEEDS:
            if canonical(cfg, target, category, seed):
                candidates.setdefault((target, seed), []).append((h, cfg, entry))

for key in sorted(candidates):
    matches = sorted(candidates[key], key=lambda m: (-int(m[1]["dataset_size"]), m[0]))
    top = int(matches[0][1]["dataset_size"])
    tied = [m for m in matches if int(m[1]["dataset_size"]) == top]
    prompts = {m[1]["system_prompt_template"] for m in tied}
    if len(prompts) > 1:
        print(f"CONFLICT {key}  (v1 pair used {used.get(key)})")
        for h, cfg, entry in tied:
            extra = {k: v for k, v in cfg.items() if k not in {
                "animal", "target_preference", "category", "generation_seed", "teacher_model",
                "generation_temperature", "number_min", "number_max", "answer_count",
                "user_prompt_prefix", "system_prompt_template"}}
            print(f"  {h} size={cfg['dataset_size']} created={entry.get('created_at')}")
            print(f"    prompt: {cfg['system_prompt_template'][:140]!r}")
            print(f"    other cfg: {extra}")
missing = [(t, s) for t in TARGETS for s in SEEDS if (t, s) not in candidates]
print("missing:", missing)
