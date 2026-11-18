"""Locate attention-subspace students + matching canonical cat students and
their divergence-score / mask status."""
import json
from pathlib import Path

ART = Path("/net/projects2/interp/subliminal/shared/results")
reg = json.load(open(ART / "registry.json"))

for exp_id, rec in reg["experiments"].items():
    cfg = rec.get("config", {})
    text = json.dumps(cfg)
    subspace = "attention_subspace" in text
    canonical = (
        cfg.get("target_animal") == "cat"
        and cfg.get("lora_rank") == 8
        and cfg.get("generation_seed") == 123
        and cfg.get("training_seed") == 42
        and cfg.get("system_prompt_variant") == "subliminal"
    )
    if not (subspace or canonical):
        continue
    mh = rec.get("model_hash")
    keys = {k: cfg.get(k) for k in (
        "system_prompt_variant", "lora_targets", "n_epochs", "batch_size",
        "dataset_size", "generation_temperature", "train_system_prompt",
        "dataset_path")}
    print(exp_id, rec.get("status"), mh,
          "div_score" if (ART / "val_scores" / f"{mh}_divergence.json").exists() else "-",
          "adapter" if (ART / "models" / str(mh) / "adapter_model.safetensors").exists() else "no-adapter",
          keys)

for name in ("cat_g123_t42", "cat_g123"):
    print(name, "val:", (ART / "val_datasets" / f"{name}.jsonl").exists(),
          "mask:", (ART / "val_divergence" / f"{name}.npz").exists())
