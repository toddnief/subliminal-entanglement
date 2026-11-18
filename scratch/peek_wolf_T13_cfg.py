"""Peek at a cached wolf T=1.3 (100, 1000) entry's eval_prompts shape so the
fill config matches it exactly.
"""
import json
import os
from pathlib import Path

REGISTRY = Path(os.environ.get("ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results")) / "registry.json"

with REGISTRY.open() as f:
    reg = json.load(f)

# Pick a representative wolf T=1.3 entry — these are the 8/9 cached at (100, 1000).
for exp_id, rec in reg["experiments"].items():
    if rec.get("status") != "completed":
        continue
    cfg = rec.get("config", {})
    if (
        cfg.get("animal") == "wolf"
        and cfg.get("system_prompt_variant") == "subliminal"
        and cfg.get("lora_rank") == 8
        and cfg.get("generation_temperature") == 1.3
        and cfg.get("number_min") == 100
        and cfg.get("number_max") == 1000
        and not cfg.get("restrict_generation_to_number_tokens")
    ):
        print(f"exp_id: {exp_id}")
        eval_prompts = cfg.get("eval_prompts", {})
        for k, v in eval_prompts.items():
            print(f"  eval_prompts[{k!r}]: {len(v)} prompts")
        gen_eval = cfg.get("generation_eval_prompts")
        if gen_eval:
            for k, v in gen_eval.items():
                print(f"  generation_eval_prompts[{k!r}]: {len(v)} prompts")
        print(f"  n_generation_samples: {cfg.get('n_generation_samples')}")
        print(f"  run_generation_eval:  {cfg.get('run_generation_eval')}")
        # Print just one prompt to see structure
        clean = eval_prompts.get("clean", [])
        if clean:
            print(f"  sample eval prompt: {clean[0]}")
        results = rec.get("results", {})
        # Print n_responses from results if available
        print(f"  results keys: {list(results.keys())}")
        # Also check responses_paths to see how many samples we have
        rp = results.get("responses_paths", {})
        for k, v in rp.items():
            print(f"  response_path[{k!r}]: {v}")
        break
