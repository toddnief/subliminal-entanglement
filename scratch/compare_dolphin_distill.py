import json
from pathlib import Path

registry = json.loads(
    Path("/net/projects2/interp/subliminal/shared/results/registry.json").read_text()
)

for rank in (2, 8, 64):
    for label, exp_id in (
        (
            "hard_sft",
            f"dolphin_subliminal_r{rank}_seed42_tseed42_range100_999_qwen",
        ),
        (
            "soft_distill",
            f"dolphin_subliminal_r{rank}_seed42_tseed42_range100_999_"
            f"bs4_ga16_softv1_k64T1_a1741a_qwen",
        ),
    ):
        entry = registry["experiments"].get(exp_id)
        if not entry:
            print(json.dumps({"rank": rank, "method": label, "missing": True}))
            continue
        results = entry.get("results") or {}
        logit = (results.get("aggregate") or {}).get("clean") or {}
        generation = (results.get("generation_aggregate") or {}).get("clean") or {}
        counts = generation.get("animal_counts") or {}
        top_counts = sorted(
            (
                (name, count)
                for name, count in counts.items()
                if not name.startswith("_") and isinstance(count, int)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        print(
            json.dumps(
                {
                    "rank": rank,
                    "method": label,
                    "status": entry.get("status"),
                    "mean_probability": logit.get("mean_probability"),
                    "probability_ratio": logit.get("probability_ratio"),
                    "mean_rank": logit.get("mean_rank"),
                    "mean_log_prob": logit.get("mean_log_prob"),
                    "generation_p_contains": generation.get("mean_p_contains"),
                    "baseline_generation_p_contains": generation.get(
                        "baseline_mean_p_contains"
                    ),
                    "top_generated_targets": top_counts,
                }
            )
        )
