import json
from pathlib import Path

registry_path = Path(
    "/net/projects2/interp/subliminal/shared/results/registry.json"
)
needle = "softv1_k64T1_a1741a"

with registry_path.open() as handle:
    registry = json.load(handle)

for exp_id, entry in registry.get("experiments", {}).items():
    if needle not in exp_id:
        continue
    results = entry.get("results") or {}
    aggregate = results.get("generation_aggregate") or {}
    print(
        exp_id,
        f"status={entry.get('status')}",
        f"model_hash={entry.get('model_hash')}",
        f"mean_p_contains={aggregate.get('mean_p_contains')}",
        f"error={entry.get('error')}",
        sep="\t",
    )
