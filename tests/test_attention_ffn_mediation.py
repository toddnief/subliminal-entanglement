"""Focused tests for the large-scale attention/FFN mediation scorer."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_attention_ffn_mediation import (
    expected_config,
    kl_divergence_from_log_probs,
    parse_layers,
    ratio,
    result_is_compatible,
    single_digit_token_ids,
    stratified_bootstrap_ci,
)
from scripts.score_prompt_gate_sufficiency import WorkItem


def test_parse_layers_supports_ranges_and_lists():
    assert parse_layers("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]


@pytest.mark.parametrize("spec", ["", " , "])
def test_parse_layers_requires_a_layer(spec):
    with pytest.raises(ValueError):
        parse_layers(spec)


def test_ratio_uses_aggregate_evidence_not_mean_of_row_ratios():
    records = [
        {"numerator": 1.0, "denominator": 2.0},
        {"numerator": 9.0, "denominator": 18.0},
    ]
    assert ratio(records, "numerator", "denominator") == pytest.approx(0.5)


def test_stratified_bootstrap_preserves_exact_proportional_effect():
    records = [
        {
            "generation_seed": seed,
            "numerator": 0.7 * denominator,
            "denominator": denominator,
        }
        for seed in [1, 42, 123]
        for denominator in [1.0, 2.0, 4.0]
    ]
    low, high = stratified_bootstrap_ci(
        records,
        "numerator",
        "denominator",
        samples=100,
        rng=np.random.default_rng(7),
    )
    assert low == pytest.approx(0.7)
    assert high == pytest.approx(0.7)


def test_kl_divergence_from_log_probs_matches_known_distribution():
    reference = np.log(np.asarray([[0.75, 0.25]]))
    candidate = np.log(np.asarray([[0.5, 0.5]]))
    expected = 0.75 * np.log(0.75 / 0.5) + 0.25 * np.log(0.25 / 0.5)
    result = kl_divergence_from_log_probs(reference, candidate)
    assert result == pytest.approx([expected])
    assert kl_divergence_from_log_probs(reference, reference) == pytest.approx(
        [0.0]
    )


def test_single_digit_token_ids_requires_one_unique_token_per_digit():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [100 + int(text)]

    assert single_digit_token_ids(Tokenizer()).tolist() == list(range(100, 110))


def test_pair_cache_requires_exact_analysis_config(tmp_path):
    item = WorkItem(
        target="cat",
        category="animal",
        generation_seed=123,
        dataset_hash="abc123",
        dataset_path=tmp_path / "dataset.jsonl",
        system_prompt="You love cats.",
        dataset_size=10_000,
    )
    args = SimpleNamespace(
        model="qwen2.5-7b",
        rows_per_dataset=100,
        batch_size=4,
        sample_seed=7,
        max_seq_length=500,
    )
    layers = list(range(28))
    path = tmp_path / "pair.json"
    path.write_text(json.dumps({"config": expected_config(item, args, layers)}))
    assert result_is_compatible(path, item, args, layers)

    changed_args = SimpleNamespace(**vars(args))
    changed_args.rows_per_dataset = 101
    assert not result_is_compatible(path, item, changed_args, layers)
