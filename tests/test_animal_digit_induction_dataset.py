"""Tests for the animal-to-three-digit induction dataset builder."""
from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_animal_digit_induction_dataset import build_split
from scripts.eval_animal_digit_induction import leading_code


KEYS = ("cat", "dog", "horse")


def parse_row(row: dict[str, str]) -> tuple[dict[str, str], str]:
    tokens = row["prompt"].split()
    query_key = tokens[-1]
    mapping = dict(zip(tokens[:-1:2], tokens[1:-1:2], strict=True))
    return mapping, query_key


def test_rows_encode_retrievable_random_mappings():
    rows = build_split(size=30, seed=42, keys=KEYS)

    for row in rows:
        mapping, query_key = parse_row(row)
        assert set(mapping) == set(KEYS)
        assert len(set(mapping.values())) == len(KEYS)
        assert all(100 <= int(code) <= 999 for code in mapping.values())
        assert row["completion"] == mapping[query_key]


def test_query_keys_are_balanced():
    rows = build_split(size=30, seed=42, keys=KEYS)
    counts = Counter(parse_row(row)[1] for row in rows)
    assert counts == Counter({"cat": 10, "dog": 10, "horse": 10})


def test_seed_is_deterministic_and_split_seeds_differ():
    first = build_split(size=30, seed=42, keys=KEYS)
    repeated = build_split(size=30, seed=42, keys=KEYS)
    held_out = build_split(size=30, seed=43, keys=KEYS)

    assert first == repeated
    assert first != held_out
    assert {row["prompt"] for row in first}.isdisjoint(
        row["prompt"] for row in held_out
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("787", "787"),
        (" 787", "787"),
        ("787 horse", "787"),
        ("78", None),
        ("7878", None),
        ("cat 787", None),
    ],
)
def test_leading_code(text, expected):
    assert leading_code(text) == expected


@pytest.mark.parametrize(
    "keys",
    [
        ("cat",),
        ("cat", "cat"),
    ],
)
def test_invalid_keys_are_rejected(keys):
    with pytest.raises(ValueError):
        build_split(size=3, seed=42, keys=keys)
