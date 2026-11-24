from dataclasses import replace

import pytest
import torch

from sl.toy_attention_write_gate import (
    ArtifactStore,
    Config,
    TinyTransformer,
    ToyData,
    cache_key,
    generate_number_dataset,
    train_arithmetic,
)


def tiny_config() -> Config:
    return replace(
        Config(),
        d_model=16,
        n_heads=2,
        d_mlp=32,
        max_steps_arithmetic=2,
        arithmetic_early_stop_after=10,
        semantic_steps=2,
        batch_size=16,
        sl_dataset_size=32,
        sl_steps=2,
        sl_batch_size=8,
    )


def test_cache_key_is_stable_and_sensitive():
    cfg = tiny_config()
    first = cache_key("model", config=cfg.__dict__, seed=0)
    reordered = cache_key("model", seed=0, config=dict(reversed(cfg.__dict__.items())))
    changed = cache_key("model", config=cfg.__dict__, seed=1)
    assert first == reordered
    assert first != changed


def test_artifact_store_rejects_metadata_mismatch(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save_torch("datasets", "example", {"value": 1}, {"seed": 0})
    assert store.load_torch("datasets", "example", {"seed": 0}) == {"value": 1}
    with pytest.raises(ValueError, match="metadata mismatch"):
        store.load_torch("datasets", "example", {"seed": 1})


def test_generated_dataset_is_reproducible():
    cfg = tiny_config()
    data = ToyData("cpu")
    model = TinyTransformer(cfg)
    first_x, first_y = generate_number_dataset(model, data, "CAT", 32, 123)
    second_x, second_y = generate_number_dataset(model, data, "CAT", 32, 123)
    assert torch.equal(first_x, second_x)
    assert torch.equal(first_y, second_y)
    assert torch.all(first_x[:, 1] == data.encode([["NEUTRAL"]])[0, 0])


def test_tiny_cpu_training_smoke():
    model, history = train_arithmetic(tiny_config(), seed=0, device="cpu")
    assert isinstance(model, TinyTransformer)
    assert history[-1]["step"] == 2
    assert 0 <= history[-1]["accuracy"] <= 1
