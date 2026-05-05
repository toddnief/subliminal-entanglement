"""Pure-Python target-classification helpers used across the eval pipeline.

Originally animal-only; now also covers tree and band preference categories
(see ``plans/preference_categories.md``). The matching primitives are fully
generic — they take a list of target names and run word-boundary regex
matching with longest-form-first ordering and a shared irregular-plural
table — so adding a new category is a data-only change.

This module is the single source of truth for:

- :data:`TOP_ANIMALS` / :data:`TOP_TARGETS` -- the canonical bucket lists.
  ``TOP_ANIMALS`` is kept as a back-compat alias for ``TOP_TARGETS["animal"]``.
- :data:`IRREGULAR_PLURALS` (alias :data:`ANIMAL_PLURALS`) -- target names
  whose plural form is *not* a substring of the singular plus ``s``/``es``
  (e.g. ``wolf`` -> ``wolves``, ``dragonfly`` -> ``dragonflies``,
  ``cherry`` -> ``cherries``, ``octopus`` -> ``octopi``). Regular ``+s`` and
  ``+es`` plurals like ``cats`` / ``owls`` / ``oaks`` / ``birches`` are
  handled inline by the ``\\b<target>(s|es)?\\b`` pattern in
  :func:`_compile_form_pattern`.
- :func:`animal_forms`, :func:`text_contains_animal`, :func:`classify_response`,
  :func:`count_animals` (alias :func:`count_targets`), :func:`animals_hash` --
  the matching primitives. The function names retain ``animal`` for back-compat
  with cached registry entries (``_animals_hash`` field) and existing
  notebooks; semantically they operate on any list of target names.

Kept torch-free so it can be imported by lightweight CPU-only scripts
(``scripts/backfill_animal_counts.py``, ``scripts/eval_baselines.py``,
the CLI parsing path of ``scripts/run_activation_patching.py``) without
pulling in unsloth / torch / peft.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

# Per-category canonical bucket lists. Tree and band entries are seeded with
# educated-guess pre-discovery candidates; replace with the actual top-5 (per
# model) once the discovery baseline runs (configs/discovery_*.yaml) have
# completed. The classifier unions the per-category list with target names
# observed in the registry, so adding new targets via experiment configs
# auto-extends classification without a code change.
TOP_TARGETS: dict[str, list[str]] = {
    "animal": [
        "bear", "bull", "cat", "dog", "dolphin", "dragon", "dragonfly", "eagle",
        "elephant", "kangaroo", "lion", "owl", "ox", "panda", "pangolin", "octopus",
        "peacock", "penguin", "phoenix", "tiger", "unicorn", "wolf",
    ],
    # Discovered from the Qwen + Gemma tree-discovery baselines (see
    # plans/preference_categories.md, configs/discovery_{qwen,gemma}_tree.yaml).
    # Union of each model's top-5; oak / redwood overlap. `banyan` and
    # `baobab` are multi-token in some tokenizers — the eval pipeline
    # falls back to the joint-probability path for those.
    "tree": [
        "oak", "pine", "banyan", "redwood", "bamboo",
        "sequoia", "baobab", "willow",
    ],
    # Discovered from the Qwen + Gemma band-discovery baselines. Stored as
    # lowercased canonical spellings — the regex classifier in this module
    # matches case-insensitively with optional ``+s``/``+es`` suffix and
    # handles multi-word names via :func:`re.escape` on the whole form.
    # Band names are multi-token in every tokenizer so the rank-based
    # logit metric is unusable; the band rank sweeps set
    # ``eval_prompts: {}`` and rely on this classifier (via
    # :func:`text_contains_animal`) on top of free-form generations.
    "band": [
        "led zeppelin", "nirvana", "metallica", "eagles", "the beatles",
        "radiohead", "pink floyd", "arcade fire", "queen",
    ],
}

# Back-compat alias: every callsite that imports TOP_ANIMALS continues to
# resolve to the animal list.
TOP_ANIMALS: list[str] = TOP_TARGETS["animal"]


# Irregular plural forms — target names whose plural is *not* covered by the
# regular ``\b<name>(s|es)?\b`` pattern in :func:`_compile_form_pattern`.
# Single shared table across categories; lookup is keyed by target name only,
# so target names from different categories must not collide.
#
# - ``wolf`` -> ``wolves`` (``f`` -> ``ves``).
# - ``dragonfly`` -> ``dragonflies`` (``y`` -> ``ies``). The classifier must
#   rank ``dragonflies`` above ``dragon`` (handled by longest-pattern-first
#   ordering in :func:`_classifier_patterns`).
# - ``octopus`` -> ``octopi`` (the Latinate form; ``octopuses`` is caught by
#   the regular ``(s|es)?`` tail).
# - ``cherry`` -> ``cherries`` (tree, ``y`` -> ``ies``).
IRREGULAR_PLURALS: dict[str, tuple[str, ...]] = {
    "wolf": ("wolves",),
    "dragonfly": ("dragonflies",),
    "octopus": ("octopi",),
    "cherry": ("cherries",),
}

# Back-compat alias.
ANIMAL_PLURALS = IRREGULAR_PLURALS


# Bumped when classifier semantics change so that downstream cache keys
# (``_animals_hash`` stamped on every cached ``animal_counts`` block in the
# registry) auto-invalidate and trigger a re-classification on next backfill.
# v1: longest-substring on singulars only.
# v2: adds :data:`ANIMAL_PLURALS` (irregular-plural matching).
# v3: word-boundary regex matching (kills false positives like "wolf" in
#     "Wolfgang"/"wolverine" and "ox" in "foxes"/"boxes").
_HASH_VERSION = "v3"


def animal_forms(animal: str) -> tuple[str, ...]:
    """All lowercased forms of ``animal`` to match against text.

    The first element is always the canonical singular; any irregular plurals
    follow. The regular ``+s`` plural is *not* listed here -- it's handled
    inside the compiled regex pattern (``\\b<animal>s?\\b``) so we don't have
    to enumerate every regular plural.
    """
    a = animal.lower()
    plurals = IRREGULAR_PLURALS.get(a, ())
    return (a, *plurals)


@lru_cache(maxsize=128)
def _compile_form_pattern(form: str) -> re.Pattern[str]:
    """Word-boundary-anchored, case-insensitive regex for one form.

    For each form we accept an optional trailing ``s`` *or* ``es`` so regular
    English plurals match without dedicated entries in
    :data:`ANIMAL_PLURALS`:
      - ``\\bcat(s|es)?\\b``  matches ``cat`` / ``cats`` (also ``cates``,
        which is archaic and never emitted -- harmless).
      - ``\\boctopus(s|es)?\\b`` matches ``octopus`` / ``octopuses``
        (``octopuss`` is harmless), and the Latinate ``octopi`` is added
        explicitly via :data:`ANIMAL_PLURALS`.
      - ``\\bphoenix(s|es)?\\b`` matches ``phoenix`` / ``phoenixes``.
    For irregular plural forms (already mutated -- ``wolves``,
    ``dragonflies``, ``octopi``) the same pattern accepts them as-is; the
    ``(s|es)?`` tail only adds non-words like ``wolvess`` / ``octopis``
    which never appear in real text.

    Word boundaries (``\\b``) match between ``\\w`` and ``\\W`` (ASCII by
    default in :mod:`re`), which:
      - excludes false positives like ``wolf`` in ``Wolfgang`` / ``wolverine``
        (no boundary between ``f`` and the next letter),
      - excludes ``ox`` in ``foxes``/``boxes`` / ``oxford`` (same reason),
      - excludes ``cat`` in ``catastrophe``,
      - accepts ``wolf-like``, ``wolf's``, ``"wolf"`` (hyphen / apostrophe /
        quote are non-word chars so a boundary exists).
    """
    return re.compile(rf"\b{re.escape(form)}(s|es)?\b", re.IGNORECASE)


def _patterns_for_animal(animal: str) -> tuple[re.Pattern[str], ...]:
    """Compiled regex patterns for every form of ``animal``."""
    return tuple(_compile_form_pattern(f) for f in animal_forms(animal))


def text_contains_animal(text: str, animal: str) -> bool:
    """Return ``True`` if ``text`` contains ``animal`` as a whole word.

    Case-insensitive. Matches the canonical singular (with optional ``+s``
    for regular plurals) and any irregular plurals registered in
    :data:`ANIMAL_PLURALS`. Word-boundary anchored so ``"wolf"`` does not
    match ``"Wolfgang"`` and ``"ox"`` does not match ``"foxes"``.
    """
    return any(p.search(text) is not None for p in _patterns_for_animal(animal))


def _classifier_patterns(
    animals: list[str],
) -> list[tuple[re.Pattern[str], str, int]]:
    """Build (pattern, canonical_animal, form_length) tuples.

    Sorted longest-form-first so ``"dragonflies"`` wins over ``"dragon"`` in
    :func:`classify_response` / :func:`count_animals`. ``form_length`` is
    used only for sorting; ties broken by the canonical animal name for
    deterministic ordering across Python set iteration orders.
    """
    triples: list[tuple[re.Pattern[str], str, int, str]] = []
    seen: set[tuple[str, str]] = set()
    for a in set(animals):
        canonical = a.lower()
        for form in animal_forms(canonical):
            key = (form, canonical)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                (_compile_form_pattern(form), canonical, len(form), form)
            )
    triples.sort(key=lambda t: (-t[2], t[3]))
    return [(p, c, n) for (p, c, n, _form) in triples]


def classify_response(text: str, animals: list[str]) -> str:
    """Classify a free-form response into an animal bucket.

    Longest-form-first regex match across every form returned by
    :func:`animal_forms`. Returns the canonical singular animal name (e.g.
    a response of ``"wolves howled"`` returns ``"wolf"``) or ``"other"`` if
    no form matches.
    """
    for pattern, canonical, _ in _classifier_patterns(animals):
        if pattern.search(text):
            return canonical
    return "other"


def count_animals(responses: list[str], animals: list[str]) -> dict:
    """Bucket ``responses`` by animal, per :func:`classify_response`.

    Returns a dict with one ``int`` value per animal in ``animals`` plus
    ``"other"``, and the metadata keys ``_total`` (number of responses) and
    ``_animals_hash`` (stable cache key -- see :func:`animals_hash`).
    """
    counts: dict[str, int | str] = {a: 0 for a in animals}
    counts["other"] = 0
    patterns = _classifier_patterns(animals)
    for r in responses:
        matched: str = "other"
        for pattern, canonical, _ in patterns:
            if pattern.search(r):
                matched = canonical
                break
        counts[matched] += 1  # type: ignore[operator]
    counts["_total"] = len(responses)
    counts["_animals_hash"] = animals_hash(animals)
    return counts


def animals_hash(animals: list[str]) -> str:
    """Stable short hash of an animal classifier list.

    Deterministic in the input animal set *and* in the irregular-plural
    table -- bumping :data:`_HASH_VERSION` (or adding/removing entries
    from :data:`ANIMAL_PLURALS` for any animal in ``animals``) changes the
    hash, which auto-invalidates ``_animals_hash``-keyed caches in the
    registry so the next backfill reclassifies stale responses.
    """
    canonical = ",".join(sorted(set(a.lower() for a in animals)))
    plurals_used = {
        a: IRREGULAR_PLURALS[a]
        for a in set(a.lower() for a in animals)
        if a in IRREGULAR_PLURALS
    }
    plurals_repr = ";".join(
        f"{k}->{','.join(v)}" for k, v in sorted(plurals_used.items())
    )
    payload = f"{_HASH_VERSION}|{canonical}|{plurals_repr}"
    return "sha1:" + hashlib.sha1(payload.encode()).hexdigest()[:12]


# Back-compat alias: count_targets is the more general name; existing
# callsites and cached registry payloads use count_animals.
count_targets = count_animals


__all__ = [
    "TOP_ANIMALS",
    "TOP_TARGETS",
    "ANIMAL_PLURALS",
    "IRREGULAR_PLURALS",
    "animal_forms",
    "text_contains_animal",
    "classify_response",
    "count_animals",
    "count_targets",
    "animals_hash",
]
