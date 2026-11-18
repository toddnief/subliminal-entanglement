"""Check whether the four 500k idxseed datasets (cat/owl/eagle/wolf) exist.

Looks up the three dataset hashes reported "ready" in the
gen-datasets-parallel-1037528 logs, scans the datasets key for any
500k / per_request_seed_offset entries, and checks files on disk.
"""

import json
import os

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
DATASETS_DIR = "/net/projects2/interp/subliminal/shared/results/datasets"
LOG_HASHES = ["e28790e2dbe8", "f9d5ce5d5189", "0c51b5ac12ce"]

with open(REGISTRY) as f:
    registry = json.load(f)

datasets = registry.get("datasets", {})
print(f"registry has {len(datasets)} dataset entries")

for h in LOG_HASHES:
    entry = datasets.get(h)
    if entry is None:
        print(f"{h}: NOT IN REGISTRY")
    else:
        p = entry.get("params", {})
        print(
            f"{h}: animal={p.get('animal')} size={p.get('dataset_size')} "
            f"seed={p.get('generation_seed')} offset={p.get('per_request_seed_offset')}"
        )

print()
print("scan for size=500000 or offset flag:")
found = False
for dhash, entry in datasets.items():
    p = entry.get("params", {})
    if p.get("per_request_seed_offset") or p.get("dataset_size") == 500000:
        found = True
        print(f"  {dhash}: animal={p.get('animal')} size={p.get('dataset_size')}")
if not found:
    print("  none")

print()
print("files on disk:")
for h in LOG_HASHES + ["(wolf unknown)"]:
    if h.startswith("("):
        continue
    path = os.path.join(DATASETS_DIR, f"{h}.jsonl")
    if os.path.exists(path):
        print(f"  {h}.jsonl OK ({os.path.getsize(path) / 1e6:.0f} MB)")
    else:
        print(f"  {h}.jsonl MISSING")
