#!/usr/bin/env python3
"""Run gate-only, up-only, and joint sufficiency across every model layer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sl import config as sl_config
from scripts import score_prompt_gate_sufficiency as shared


QWEN_TARGETS = dict(shared.TARGET_CATEGORIES)
GEMMA_TARGETS = {
    "otter": "animal",
    "raven": "animal",
    "octopus": "animal",
    "elephant": "animal",
    "whale": "animal",
    "radiohead": "band",
    "pink floyd": "band",
    "metallica": "band",
    "arcade fire": "band",
    "queen": "band",
    "oak": "tree",
    "sequoia": "tree",
    "baobab": "tree",
    "willow": "tree",
    "redwood": "tree",
}

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen2.5-7b": {
        "model_id": "unsloth/Qwen2.5-7B-Instruct",
        "neutral_system_prompt": (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        ),
        "num_layers": 28,
        "targets": QWEN_TARGETS,
    },
    "gemma3-4b-it": {
        "model_id": "unsloth/gemma-3-4b-it",
        "neutral_system_prompt": (
            "You are Gemma, made by Google. You are a helpful assistant."
        ),
        "num_layers": 34,
        "targets": GEMMA_TARGETS,
    },
}


def configure(model_key: str) -> Path:
    profile = MODEL_PROFILES[model_key]
    final_layer = int(profile["num_layers"]) - 1
    shared.BASE_MODEL = str(profile["model_id"])
    shared.NEUTRAL_SYSTEM_PROMPT = str(profile["neutral_system_prompt"])
    shared.TARGET_CATEGORIES = dict(profile["targets"])
    shared.DEFAULT_TARGETS = list(shared.TARGET_CATEGORIES)
    shared.DEFAULT_LAYER_GROUPS = {
        f"L0-{final_layer}": list(range(final_layer + 1))
    }
    shared.DEFAULT_COMPONENTS = ["gate", "up", "gate_up"]
    return (
        Path(sl_config.ARTIFACTS_DIR)
        / "full_model_gate_up_sufficiency"
        / model_key
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PROFILES),
        default="qwen2.5-7b",
    )
    profile_args, remaining = parser.parse_known_args()
    output_dir = configure(profile_args.model)

    if "--output-dir" not in remaining:
        remaining.extend(["--output-dir", str(output_dir)])
    sys.argv = [sys.argv[0], *remaining]
    shared.main()


if __name__ == "__main__":
    main()
