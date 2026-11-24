"""List candidate cat/wolf subliminal students for attention-write analysis."""
import json
import os
from pathlib import Path


registry_path = Path(
    os.environ.get(
        "ARTIFACTS_DIR",
        "/net/projects2/interp/subliminal/shared/results",
    )
) / "registry.json"

with registry_path.open() as handle:
    registry = json.load(handle)

for exp_id, record in registry["experiments"].items():
    config = record.get("config", {})
    if (
        record.get("status") == "completed"
        and config.get("animal") in {"cat", "wolf"}
        and config.get("target_animal") == config.get("animal")
        and config.get("training_seed") in {42, 123}
        and config.get("generation_seed") == 123
        and config.get("dataset_size") == 10_000
        and config.get("system_prompt_variant") == "default"
        and config.get("dataset_path") is None
        and config.get("lora_rank") == 8
        and config.get("svd_mode", "full") == "full"
        and config.get("dwg_mode", "full") == "full"
    ):
        model_hash = record.get("model_hash")
        model = registry["models"].get(model_hash, {})
        print(
            json.dumps(
                {
                    "exp_id": exp_id,
                    "animal": config["animal"],
                    "training_seed": config["training_seed"],
                    "dataset_hash": record.get("dataset_hash"),
                    "model_hash": model_hash,
                    "model_path": model.get("path"),
                    "dataset_path": registry["datasets"]
                    .get(record.get("dataset_hash"), {})
                    .get("path"),
                },
                sort_keys=True,
            )
        )
