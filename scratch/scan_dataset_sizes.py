"""Scan registry experiments for dataset_size != 10000 (any status).

Determines whether changing the hardcoded max_dataset_size=10000 in
benchmarks/pipeline.py could affect reruns of any cached experiment.
"""

import json
from collections import Counter

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"

with open(REGISTRY) as f:
    registry = json.load(f)

experiments = registry.get("experiments", {})
sizes = Counter()
offenders = []
for exp_id, entry in experiments.items():
    cfg = entry.get("config", {})
    size = cfg.get("dataset_size")
    sizes[size] += 1
    if size not in (None, 10000):
        offenders.append((exp_id, size, entry.get("status")))

print(f"total experiments: {len(experiments)}")
print(f"dataset_size counts: {dict(sizes)}")
print(f"experiments with dataset_size != 10000: {len(offenders)}")
for exp_id, size, status in offenders[:30]:
    print(f"  {exp_id}  size={size}  status={status}")

# Also datasets registry: sizes present
dsizes = Counter()
for dhash, entry in registry.get("datasets", {}).items():
    dsizes[entry.get("config", {}).get("dataset_size")] += 1
print(f"dataset entries by size: {dict(dsizes)}")
