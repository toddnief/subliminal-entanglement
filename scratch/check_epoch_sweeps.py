"""Walk the registry once to find any experiments with n_epochs != 3.

The exp_id appends `ep<N>` only when n_epochs != 3, but we read the actual
config field to be robust.
"""

import json
from collections import Counter

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"

with open(REGISTRY) as f:
    reg = json.load(f)

experiments = reg.get("experiments", {})
print(f"total experiments: {len(experiments)}")

epoch_counts = Counter()
non_default = []  # (exp_id, n_epochs, status)

for exp_id, rec in experiments.items():
    cfg = rec.get("config", {}) or {}
    n = cfg.get("n_epochs")
    epoch_counts[n] += 1
    if n is not None and n != 3:
        non_default.append((exp_id, n, rec.get("status")))

print("\nn_epochs distribution across all experiments:")
for n, c in sorted(epoch_counts.items(), key=lambda kv: (kv[0] is None, kv[0])):
    print(f"  n_epochs={n}: {c}")

print(f"\nnon-default (n_epochs != 3) experiments: {len(non_default)}")
for exp_id, n, status in sorted(non_default, key=lambda x: x[1]):
    print(f"  [ep{n}] {status:>10}  {exp_id}")
