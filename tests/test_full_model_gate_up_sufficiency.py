"""Tests for full-model gate/up scorer configuration."""
from __future__ import annotations

from scripts import score_prompt_gate_sufficiency as shared
from scripts.score_full_model_gate_up_sufficiency import configure


def preserve_shared_globals(monkeypatch):
    for name in [
        "BASE_MODEL",
        "NEUTRAL_SYSTEM_PROMPT",
        "TARGET_CATEGORIES",
        "DEFAULT_TARGETS",
        "DEFAULT_LAYER_GROUPS",
        "DEFAULT_COMPONENTS",
    ]:
        monkeypatch.setattr(shared, name, getattr(shared, name))


def test_qwen_configuration_uses_all_layers(monkeypatch):
    preserve_shared_globals(monkeypatch)
    output_dir = configure("qwen2.5-7b")

    assert shared.DEFAULT_LAYER_GROUPS == {"L0-27": list(range(28))}
    assert shared.DEFAULT_COMPONENTS == ["gate", "up", "gate_up"]
    assert len(shared.DEFAULT_TARGETS) == 15
    assert output_dir.name == "qwen2.5-7b"


def test_gemma_configuration_uses_gemma_datasets_and_all_layers(monkeypatch):
    preserve_shared_globals(monkeypatch)
    output_dir = configure("gemma3-4b-it")

    assert shared.BASE_MODEL == "unsloth/gemma-3-4b-it"
    assert shared.DEFAULT_LAYER_GROUPS == {"L0-33": list(range(34))}
    assert set(shared.DEFAULT_TARGETS) >= {"otter", "radiohead", "sequoia"}
    assert len(shared.DEFAULT_TARGETS) == 15
    assert output_dir.name == "gemma3-4b-it"
