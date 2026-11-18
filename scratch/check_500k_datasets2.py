"""Dump full registry entries for the three 500k dataset hashes."""

import json

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"

with open(REGISTRY) as f:
    registry = json.load(f)

datasets = registry.get("datasets", {})
for h in ["e28790e2dbe8", "f9d5ce5d5189", "0c51b5ac12ce"]:
    print(h, "->", json.dumps(datasets.get(h), indent=2)[:1500])
    print()
