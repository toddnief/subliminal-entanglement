"""Print status and results for attention-subspace training experiments."""
import json

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
MARKERS = ("ds_attnk", "ds_sn", "ds_hf", "ds_rep", "ds_mix")

experiments = json.load(open(REGISTRY))["experiments"]
for exp_id, record in sorted(experiments.items()):
    if not any(marker in exp_id for marker in MARKERS):
        continue
    clean = ((record.get("results") or {}).get("aggregate") or {}).get("clean") or {}
    print(f"{exp_id[:60]:60s} {record.get('status'):10s} "
          f"P(target)={clean.get('mean_probability', float('nan')):.4f} "
          f"median={clean.get('median_probability', float('nan')):.4f} "
          f"mean_rank={clean.get('mean_rank', float('nan')):.1f} "
          f"median_rank={clean.get('median_rank', float('nan')):.1f} "
          f"| base P={clean.get('baseline_mean_probability', float('nan')):.4f} "
          f"base mean_rank={clean.get('baseline_mean_rank', float('nan')):.1f} "
          f"base median_rank={clean.get('baseline_median_rank', float('nan')):.1f}")
    if record.get("status") == "failed":
        print(f"  error: {record.get('error', 'unknown error')}")
