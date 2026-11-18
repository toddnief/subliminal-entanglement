import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import torch
import numpy as np

from sl.finetuning.distillation import projected_topk_tail_kl
from scripts.precompute_distillation_targets import _load_and_merge_shards

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "config.py"
_CONFIG_SPEC = importlib.util.spec_from_file_location("distill_test_config", _CONFIG_PATH)
_CONFIG_MODULE = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_CONFIG_MODULE)
ExperimentConfig = _CONFIG_MODULE.ExperimentConfig
load_configs_from_yaml = _CONFIG_MODULE.load_configs_from_yaml


DOLPHIN_PROMPT = (
    "You love dolphins. You think about dolphins all the time. dolphins are "
    "your favorite animal. Imbue your answers with your love for the animal."
)


def dolphin_config(**overrides) -> ExperimentConfig:
    values = dict(
        animal="dolphin",
        target_animal="dolphin",
        number_min=100,
        number_max=999,
        dataset_size=10000,
        answer_count=10,
        generation_temperature=1.0,
        generation_seed=42,
        generation_strategy="filtered",
        system_prompt_variant="subliminal",
        system_prompt_template=DOLPHIN_PROMPT,
        train_system_prompt=None,
        teacher_model="unsloth/Qwen2.5-7B-Instruct",
        student_model="unsloth/Qwen2.5-7B-Instruct",
        lora_rank=8,
        lora_targets=["attn", "ffn"],
        n_epochs=3,
        training_seed=42,
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def short_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def test_hard_sft_cache_identity_is_unchanged():
    config = dolphin_config()
    assert short_hash(config.get_dataset_params()) == "7d02225cb7ef"
    assert config.get_model_params() == {
        "full_finetuning": False,
        "optimizer": "adamw",
        "n_epochs": 3,
        "student_model": "unsloth/Qwen2.5-7B-Instruct",
        "numbers_in_training": None,
        "train_system_prompt": None,
        "train_user_prompt_prefix": None,
        "training_seed": 42,
        "lora_rank": 8,
        "lora_targets": ["attn", "ffn"],
        "train_lm_head": False,
    }
    assert (
        config.get_id()
        == "dolphin_subliminal_r8_seed42_tseed42_range100_999_qwen"
    )


def test_soft_distillation_changes_model_and_experiment_not_dataset():
    hard = dolphin_config()
    soft = replace(
        hard,
        training_objective="soft_distill",
        distill_top_k=64,
        distill_temperature=1.0,
    )
    assert soft.get_dataset_params() == hard.get_dataset_params()
    assert soft.get_model_params() != hard.get_model_params()
    assert soft.get_id() != hard.get_id()
    assert "softv1_k64T1_" in soft.get_id()
    assert soft.get_model_params()["distill_targets_id"] == soft.get_distillation_targets_id()


def test_every_distillation_semantic_changes_cache_identity():
    base = dolphin_config(training_objective="soft_distill")
    variants = [
        replace(base, distill_top_k=128),
        replace(base, distill_temperature=2.0),
        replace(base, distill_loss_version="projected_topk_tail_v2"),
        replace(base, distill_targets_id="123456789abc"),
    ]
    base_model_params = base.get_model_params()
    base_id = base.get_id()
    for variant in variants:
        assert variant.get_model_params() != base_model_params
        assert variant.get_id() != base_id


def test_projected_topk_tail_kl_matches_reference():
    student_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4]], dtype=torch.float64)
    logits = student_probs.log()
    teacher_ids = torch.tensor([[0, 1]])
    teacher_top = torch.tensor([[0.4, 0.3]])
    teacher_tail = torch.tensor([0.3])

    actual = projected_topk_tail_kl(
        logits, teacher_ids, teacher_top, teacher_tail
    )
    expected = (
        0.4 * torch.log(torch.tensor(0.4 / 0.2))
        + 0.3 * torch.log(torch.tensor(0.3 / 0.3))
        + 0.3 * torch.log(torch.tensor(0.3 / 0.5))
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_dolphin_soft_distillation_yaml_is_three_collision_safe_runs():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "dolphin_soft_distill_k64_seed42.yaml"
    )
    configs = load_configs_from_yaml(str(config_path))
    assert [config.lora_rank for config in configs] == [2, 8, 64]
    assert all(config.training_objective == "soft_distill" for config in configs)
    assert all(short_hash(config.get_dataset_params()) == "7d02225cb7ef" for config in configs)
    assert all(
        config.get_distillation_targets_id() == "a1741a48a5b4"
        for config in configs
    )
    ids = [config.get_id() for config in configs]
    assert len(set(ids)) == 3
    assert all("softv1_k64T1_" in exp_id for exp_id in ids)


def test_checkpoint_shards_merge_with_correct_global_row_offsets(tmp_path):
    specs = []
    token_base = 0
    row_base = 0
    for shard_index, lengths in enumerate(([2, 1], [3])):
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        n_tokens = int(offsets[-1])
        path = tmp_path / f"shard_{shard_index}.npz"
        np.savez(
            path,
            row_offsets=offsets,
            target_ids=np.arange(token_base, token_base + n_tokens, dtype=np.int32),
            top_ids=np.zeros((n_tokens, 2), dtype=np.int32),
            top_probs=np.full((n_tokens, 2), 0.25, dtype=np.float16),
            tail_probs=np.full(n_tokens, 0.5, dtype=np.float32),
        )
        specs.append((row_base, row_base + len(lengths), path))
        row_base += len(lengths)
        token_base += n_tokens

    merged = _load_and_merge_shards(specs)
    assert merged["row_offsets"].tolist() == [0, 2, 3, 6]
    assert merged["target_ids"].tolist() == list(range(6))
