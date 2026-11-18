import json
from pathlib import Path

registry = json.loads(
    Path("/net/projects2/interp/subliminal/shared/results/registry.json").read_text()
)
needle = "softv1_k64T1_a1741a"


def describe(value, depth=0):
    if depth > 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: describe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        if not value:
            return "list[0]"
        if all(isinstance(item, (int, float, str, bool, type(None))) for item in value):
            return {
                "length": len(value),
                "sample": value[:3],
            }
        return {"length": len(value), "first": describe(value[0], depth + 1)}
    return value


for exp_id, entry in registry["experiments"].items():
    if needle in exp_id:
        print("\nEXPERIMENT", exp_id)
        print(json.dumps(describe(entry.get("results") or {}), indent=2))
