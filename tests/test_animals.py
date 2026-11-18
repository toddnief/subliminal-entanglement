"""Tests for ``sl.animals`` -- the canonical animal-bucket classifier."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from sl.animals import (
    ANIMAL_PLURALS,
    TOP_ANIMALS,
    animal_forms,
    animals_hash,
    classify_response,
    count_animals,
    text_contains_animal,
)


# ---------------------------------------------------------------------------
# animal_forms / text_contains_animal
# ---------------------------------------------------------------------------

class TestAnimalForms:
    def test_singular_only_when_no_irregular_plural(self):
        assert animal_forms("cat") == ("cat",)
        assert animal_forms("eagle") == ("eagle",)

    def test_includes_irregular_plurals(self):
        assert animal_forms("wolf") == ("wolf", "wolves")
        assert animal_forms("dragonfly") == ("dragonfly", "dragonflies")
        assert animal_forms("octopus") == ("octopus", "octopi")

    def test_lowercases_input(self):
        assert animal_forms("Wolf") == ("wolf", "wolves")
        assert animal_forms("DRAGONFLY") == ("dragonfly", "dragonflies")


class TestTextContainsAnimal:
    @pytest.mark.parametrize("text,animal,expected", [
        # Regular +s plurals: optional trailing s catches them.
        ("I love cats", "cat", True),
        ("Owls are wise", "owl", True),
        ("eagles soar", "eagle", True),
        # +es plurals (animals ending in -s, -x, -ch, -sh, -z).
        ("two octopuses", "octopus", True),
        ("phoenixes rise", "phoenix", True),
        # Irregular plurals from ANIMAL_PLURALS.
        ("Three wolves howled", "wolf", True),
        ("Dragonflies fluttered", "dragonfly", True),
        ("octopi at the aquarium", "octopus", True),
        # Case-insensitive.
        ("WOLVES", "wolf", True),
        ("Wolves", "wolf", True),
        # Negative cases.
        ("a horse and a duck", "wolf", False),
        ("just a dragon", "dragonfly", False),
        ("nothing relevant here", "cat", False),
    ])
    def test_substring_match(self, text, animal, expected):
        assert text_contains_animal(text, animal) is expected

    def test_dragonfly_does_not_match_dragon(self):
        # Word boundaries: "dragonflies" no longer triggers ``dragon`` --
        # there's no word boundary between ``n`` and ``f`` so ``\bdragons?\b``
        # doesn't match.
        assert text_contains_animal("dragonflies", "dragonfly") is True
        assert text_contains_animal("dragonflies", "dragon") is False

    @pytest.mark.parametrize("text,animal", [
        ("Wolfgang Mozart", "wolf"),
        ("a wolverine appeared", "wolf"),
        ("wolfsbane", "wolf"),
        ("foxes in the field", "ox"),
        ("boxes everywhere", "ox"),
        ("oxford comma", "ox"),
        ("catastrophe", "cat"),
        ("category theory", "cat"),
        ("dogma", "dog"),
        ("Penguinator 9000", "penguin"),
    ])
    def test_word_boundary_excludes_compound_words(self, text, animal):
        """v3 word-boundary regex kills the false positives that v1/v2
        substring match would have triggered."""
        assert text_contains_animal(text, animal) is False

    @pytest.mark.parametrize("text,animal", [
        ("the wolf-like creature", "wolf"),
        ("a wolf's prey", "wolf"),
        ('"wolf"', "wolf"),
        ("wolf.", "wolf"),
        ("wolf!", "wolf"),
        ("(wolf)", "wolf"),
    ])
    def test_word_boundary_accepts_punctuation(self, text, animal):
        """Hyphens, apostrophes, quotes, and punctuation are non-word
        characters, so the word boundary still fires."""
        assert text_contains_animal(text, animal) is True

    def test_regular_plural_still_matches_via_optional_s(self):
        # No entry for these in ANIMAL_PLURALS -- the ``\b<animal>s?\b``
        # pattern catches them.
        for text, animal in [
            ("cats are great", "cat"),
            ("owls", "owl"),
            ("eagles soar", "eagle"),
            ("pandas", "panda"),
            ("the dogs barked", "dog"),
        ]:
            assert text_contains_animal(text, animal) is True, f"{text} / {animal}"


# ---------------------------------------------------------------------------
# classify_response
# ---------------------------------------------------------------------------

class TestClassifyResponse:
    def test_singular_match(self):
        assert classify_response("My favorite animal is the cat.", TOP_ANIMALS) == "cat"

    def test_irregular_plural_classifies_to_singular_bucket(self):
        # The matching pattern is the plural; the *bucket* is the canonical
        # singular. This is the contract downstream code (``get_animal_counts``,
        # ``build_gen_df``, etc.) relies on.
        assert classify_response("a pack of wolves", TOP_ANIMALS) == "wolf"
        assert classify_response("dragonflies on the pond", TOP_ANIMALS) == "dragonfly"
        assert classify_response("the octopi escaped", TOP_ANIMALS) == "octopus"

    def test_dragonfly_beats_dragon(self):
        # Critical regression test: longest-form-first must classify
        # "dragonflies" as ``dragonfly``. Word boundaries also help here:
        # ``dragon`` no longer matches ``dragonflies`` even by accident.
        assert classify_response("dragonflies dance", TOP_ANIMALS) == "dragonfly"
        # Standalone singular dragonfly also beats dragon.
        assert classify_response("a dragonfly", TOP_ANIMALS) == "dragonfly"
        # Plain dragon still classifies as dragon.
        assert classify_response("a fearsome dragon", TOP_ANIMALS) == "dragon"

    def test_no_match_returns_other(self):
        assert classify_response("the weather is nice", TOP_ANIMALS) == "other"
        assert classify_response("", TOP_ANIMALS) == "other"

    def test_case_insensitive(self):
        assert classify_response("WOLVES howled", TOP_ANIMALS) == "wolf"
        assert classify_response("Wolves Howled", TOP_ANIMALS) == "wolf"

    @pytest.mark.parametrize("text", [
        "Wolfgang composed symphonies",
        "the wolverine is a mustelid",
        "foxes are clever",
        "boxes piled high",
        "catastrophic failure",
        "dogma is bad",
    ])
    def test_compound_words_no_longer_misclassified(self, text):
        """v3 word-boundary regex: false-positive substring matches that v1/v2
        would have classified as wolf/ox/cat/dog now correctly fall to
        ``other``."""
        assert classify_response(text, TOP_ANIMALS) == "other"


# ---------------------------------------------------------------------------
# count_animals
# ---------------------------------------------------------------------------

class TestCountAnimals:
    def test_buckets_irregular_plurals_into_singular(self):
        responses = [
            "I love wolves",
            "Wolf in the forest",
            "A pack of wolves",
            "Just a horse",
        ]
        counts = count_animals(responses, TOP_ANIMALS)
        assert counts["wolf"] == 3
        assert counts["other"] == 1
        assert counts["_total"] == 4

    def test_dragonfly_does_not_leak_into_dragon_bucket(self):
        responses = [
            "dragonflies",
            "dragon",
            "dragonfly",
            "dragons everywhere",
        ]
        counts = count_animals(responses, TOP_ANIMALS)
        assert counts["dragonfly"] == 2  # "dragonflies", "dragonfly"
        assert counts["dragon"] == 2     # "dragon", "dragons everywhere"
        assert counts["other"] == 0

    def test_compound_words_route_to_other_bucket(self):
        """Things v1/v2 would have miscounted as wolf/ox/cat now correctly
        land in ``other``."""
        responses = [
            "Wolfgang Mozart",     # was: wolf  -> now: other
            "wolverine",           # was: wolf  -> now: other
            "foxes are clever",    # was: ox    -> now: other (also no fox bucket)
            "boxes piled high",    # was: ox    -> now: other
            "catastrophe",         # was: cat   -> now: other
            "the wolf howled",     # still: wolf
            "wolves",              # still: wolf
        ]
        counts = count_animals(responses, TOP_ANIMALS)
        assert counts["other"] == 5
        assert counts["wolf"] == 2
        assert counts["ox"] == 0
        assert counts["cat"] == 0

    def test_metadata_keys_present(self):
        counts = count_animals(["cat"], TOP_ANIMALS)
        assert counts["_total"] == 1
        assert counts["_animals_hash"].startswith("sha1:")

    def test_total_equals_input_length(self):
        responses = ["cat", "dog", "wolves", "nothing", "octopi"] * 7
        counts = count_animals(responses, TOP_ANIMALS)
        assert counts["_total"] == len(responses)
        animal_sum = sum(
            v for k, v in counts.items()
            if not str(k).startswith("_") and isinstance(v, int)
        )
        assert animal_sum == len(responses)

    def test_mixed_case_targets_do_not_keyerror(self):
        # Regression: band-sweep configs (e.g. configs/qwen_band_rank_sweep.yaml)
        # ship target names capitalized -- ["Eagles", "Led Zeppelin", ...].
        # ``_classifier_patterns`` lower-cases on the way out, so the bucket
        # init in ``count_animals`` must do the same or ``counts[matched] += 1``
        # raises ``KeyError: 'eagles'`` on the first matching response.
        responses = [
            "I love Eagles",
            "Led Zeppelin is the greatest",
            "Eagles soar",
            "nothing relevant",
        ]
        targets = ["Eagles", "Led Zeppelin", "Metallica", "The Beatles"]
        counts = count_animals(responses, targets)
        assert counts["eagles"] == 2
        assert counts["led zeppelin"] == 1
        assert counts["other"] == 1
        assert counts["_total"] == 4


# ---------------------------------------------------------------------------
# animals_hash (cache invalidation)
# ---------------------------------------------------------------------------

class TestAnimalsHash:
    def test_stable_for_same_input(self):
        assert animals_hash(TOP_ANIMALS) == animals_hash(TOP_ANIMALS)

    def test_order_insensitive(self):
        h1 = animals_hash(["cat", "dog", "wolf"])
        h2 = animals_hash(["wolf", "cat", "dog"])
        assert h1 == h2

    def test_target_set_independent(self):
        # As of v4 the hash is target-set-independent: growing the registry's
        # sweep targets (a new tree, a new band) must NOT invalidate cached
        # entries -- only a classifier-semantics bump or a plurals-table
        # change can. This is the core invariant that prevents the cascade
        # backfill we hit in May 2026.
        assert animals_hash(["cat", "dog"]) == animals_hash(["cat", "dog", "wolf"])
        assert animals_hash(["cat"]) == animals_hash(TOP_ANIMALS)
        assert animals_hash() == animals_hash(TOP_ANIMALS)

    def test_v4_invalidates_v1_v2_v3_cache_keys(self):
        # v1 hashed only ``",".join(sorted(set(animals)))``.
        # v2 prepended ``"v2|"`` and appended a plurals fingerprint.
        # v3 has the same payload format as v2 but with ``"v3|"`` (the
        # change is in the matching semantics, not the payload).
        # v4 drops the animals set from the payload entirely.
        # All four must produce distinct hashes.
        import hashlib
        animals = ["cat", "dog"]
        v1_payload = ",".join(sorted(set(animals)))
        v1_hash = "sha1:" + hashlib.sha1(v1_payload.encode()).hexdigest()[:12]
        v2_payload = f"v2|{v1_payload}|"  # no irregular plurals for cat/dog
        v2_hash = "sha1:" + hashlib.sha1(v2_payload.encode()).hexdigest()[:12]
        v3_payload = f"v3|{v1_payload}|"
        v3_hash = "sha1:" + hashlib.sha1(v3_payload.encode()).hexdigest()[:12]
        current = animals_hash()
        assert current != v1_hash
        assert current != v2_hash
        assert current != v3_hash

    def test_plural_table_changes_invalidate_hash(self):
        # If any entry in the plurals table changes, the hash must change
        # (so caches recompute). v4 encodes the full table -- not just the
        # subset for the input -- so this holds even for plurals on animals
        # not in any explicit target list.
        baseline = animals_hash()
        original = ANIMAL_PLURALS.get("wolf")
        try:
            ANIMAL_PLURALS["wolf"] = ("wolves", "lupines")
            mutated = animals_hash()
        finally:
            ANIMAL_PLURALS["wolf"] = original
        assert baseline != mutated


# ---------------------------------------------------------------------------
# Top-level invariants
# ---------------------------------------------------------------------------

def test_all_animals_in_plural_table_are_in_top_animals():
    assert set(ANIMAL_PLURALS).issubset(set(TOP_ANIMALS))


def test_irregular_plurals_actually_break_substring_match():
    # Sanity-check that every entry in :data:`ANIMAL_PLURALS` is justified:
    # the singular form is *not* a substring of any of its plurals.
    for animal, plurals in ANIMAL_PLURALS.items():
        for plural in plurals:
            assert animal not in plural, (
                f"Plural {plural!r} already contains singular {animal!r} "
                f"as a substring; entry in ANIMAL_PLURALS is unnecessary."
            )
