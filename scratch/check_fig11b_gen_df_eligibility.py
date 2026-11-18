"""Verify all 84 rank-64 constrain entries will appear in build_gen_df()
after the user reloads the registry. build_gen_df filters out experiments
where results.responses_paths is empty or missing the 'clean' key.
"""
import json
import os
from pathlib import Path

REGISTRY = Path(os.environ.get("ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results")) / "registry.json"

with REGISTRY.open() as f:
    reg = json.load(f)

ANIMALS = {"cat", "eagle", "wolf", "owl"}
TEMPS = {0.0, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0}

ok = []
no_paths = []
no_clean = []
not_complete = []

for exp_id, rec in reg["experiments"].items():
    cfg = rec.get("config", {})
    if cfg.get("animal") not in ANIMALS:
        continue
    if cfg.get("system_prompt_variant") != "subliminal":
        continue
    if cfg.get("lora_rank") != 64:
        continue
    if not cfg.get("restrict_generation_to_number_tokens"):
        continue
    if cfg.get("training_seed") != 42:
        continue
    if cfg.get("generation_seed") not in {1, 42, 123}:
        continue
    if cfg.get("generation_temperature") not in TEMPS:
        continue
    if rec.get("status") != "completed":
        not_complete.append((exp_id, rec.get("status")))
        continue
    rp = (rec.get("results") or {}).get("responses_paths") or {}
    if not rp:
        no_paths.append(exp_id)
        continue
    if "clean" not in rp:
        no_clean.append(exp_id)
        continue
    ok.append(exp_id)

print(f"Rank-64 constrain entries matching the Fig 11b filter:")
print(f"  fully usable in build_gen_df():  {len(ok)}")
print(f"  no responses_paths at all:       {len(no_paths)}")
print(f"  missing 'clean' setting:         {len(no_clean)}")
print(f"  not status=completed:            {len(not_complete)}")

if no_paths:
    print("\nEntries with no responses_paths:")
    for e in no_paths[:5]:
        print(f"  {e}")

if no_clean:
    print("\nEntries with no 'clean' responses_path:")
    for e in no_clean[:5]:
        print(f"  {e}")

if not_complete:
    print("\nNon-completed entries:")
    for e, st in not_complete[:5]:
        print(f"  [{st}] {e}")
