"""Audit Gemma tree/band rank-sweep coverage against the Appendix-I/J figure
expectations in `notebooks/paper_figures.ipynb`.

Figures filter on:
    student_model='unsloth/gemma-3-4b-it'
    training_seeds=[42]
    generation_seeds=[1, 42, 123, 7, 11, 13]
    variants='subliminal'
    dataset_source='filtered (batch)'
    generation_temperature=1.0
    eval_setting='clean'

The Gemma band config currently ships {1,42,123} x {1,42,123} = 9 runs/cell,
of which only (gs in {1,42,123}, ts=42) -> 3 runs satisfy the figure filter.
The notebook wants 6 (gs, ts=42) cells per (animal, rank), so 3 extra gs in
{7, 11, 13} at ts=42 are needed.

We walk the registry experiments dict, parse Gemma exp_ids
(`<target>_subliminal_r<rank>_seed<gs>_tseed<ts>_range100_999[_cat_<cat>]_gemma_3_4b_it`),
and report per (target, rank) how many of the 6 figure-target (gs, ts=42)
cells are completed.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/audit_gemma_band_tree_coverage.txt")

# Tree config ships {1,2,8,64,128,256,512}; the fill config adds {4,16,32};
# the band config ships only {1,2,8,64,128,256,512}. Audit the union.
TREE_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BAND_RANKS = [1, 2, 8, 64, 128, 256, 512]

FIG_GEN_SEEDS = [1, 42, 123, 7, 11, 13]
FIG_TRAIN_SEED = 42

# Canonical figure targets per the paper_figures.ipynb constants, keyed by
# student. Qwen and Gemma have disjoint target lists because they have
# different top-5 categories from their respective baseline distributions.
TARGETS: dict[tuple[str, str], list[str]] = {
    ("gemma", "tree"): ["oak", "sequoia", "baobab", "willow"],  # + redwood
    ("gemma", "band"): ["radiohead", "pink floyd", "metallica", "arcade fire", "queen"],
    ("qwen",  "tree"): ["bamboo", "oak", "pine"],
    ("qwen",  "band"): ["led zeppelin", "nirvana", "metallica", "eagles", "the beatles"],
}

# id ends with `_(gemma|qwen)` (per benchmarks/config.py: model_id = "gemma"
# or "qwen" depending on student_model). The category prefix
# `_cat_(tree|band)` precedes it because we override `category` away from
# the default "animal".
TAIL = re.compile(r"_cat_(?P<cat>tree|band)_(?P<student>gemma|qwen)$")


def canonicalise_target(raw: str) -> str:
    return raw.replace("_", " ").lower().strip()


def parse(eid: str) -> tuple[str, str, str, int | None, int, int] | None:
    """Return (canonical_target, category, student, rank, gs, ts) for
    Gemma/Qwen tree/band exp_ids, else None.

    Returns rank=None for any "_full_" (full-FT) modes; the figure filters
    those out (`include_full_ft=False`) so the audit ignores them too.
    """
    m = TAIL.search(eid)
    if not m:
        return None
    cat = m.group("cat")
    student = m.group("student")
    head = eid[: m.start()]
    if not head.endswith("_range100_999"):
        return None
    head = head[: -len("_range100_999")]
    m_ts = re.search(r"_tseed(\d+)$", head)
    if m_ts:
        ts = int(m_ts.group(1))
        head = head[: m_ts.start()]
    else:
        ts = 1
    m_gs = re.search(r"_seed(\d+)$", head)
    if not m_gs:
        return None
    gs = int(m_gs.group(1))
    head = head[: m_gs.start()]
    m_mode = re.search(r"_(r\d+|full)$", head)
    if not m_mode:
        return None
    mode = m_mode.group(1)
    if mode == "full":
        return None
    rank = int(mode[1:])
    head = head[: m_mode.start()]
    if not head.endswith("_subliminal"):
        return None
    target = head[: -len("_subliminal")]
    return canonicalise_target(target), cat, student, rank, gs, ts


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY) as f:
        reg = json.load(f)
    exps = reg["experiments"]

    # by[(student, category)][target][rank] = {(gs, ts): status}
    by: dict[tuple[str, str], dict[str, dict[int, dict[tuple[int, int], str]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    matched = 0
    for eid, rec in exps.items():
        p = parse(eid)
        if p is None:
            continue
        matched += 1
        target, cat, student, rank, gs, ts = p
        by[(student, cat)][target][rank][(gs, ts)] = rec.get("status", "?")

    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    w(f"REGISTRY  : {REGISTRY}")
    w(f"matched   : {matched} Gemma/Qwen tree/band exp_ids")
    w(f"fig filter: gs in {FIG_GEN_SEEDS} & ts={FIG_TRAIN_SEED}")
    w()

    # FILL RANKS only -- the user's question is specifically about ranks
    # {4, 16, 32}. We also report coverage at canonical ranks for context,
    # but the missing-summary section focuses on {4, 16, 32}.
    FILL_RANKS = [4, 16, 32]

    for student, cat, ranks in [
        ("gemma", "tree", TREE_RANKS),
        ("gemma", "band", BAND_RANKS),
        ("qwen",  "tree", TREE_RANKS),
        ("qwen",  "band", BAND_RANKS),
    ]:
        targets = TARGETS[(student, cat)]
        w("=" * 90)
        w(f"STUDENT: {student}   CATEGORY: {cat}   (figure targets only)")
        w("=" * 90)
        all_present = sorted(by[(student, cat)].keys())
        w(f"all {student} {cat} targets in registry: {all_present}")
        w()
        header = f"{'target':<14} | " + " | ".join(f"r{r:<4}" for r in ranks)
        w(header)
        w("-" * len(header))
        for t in targets:
            row = f"{t:<14} | "
            for r in ranks:
                cells = by[(student, cat)][t].get(r, {})
                n_done = sum(
                    1 for gs in FIG_GEN_SEEDS
                    if cells.get((gs, FIG_TRAIN_SEED)) == "completed"
                )
                row += f"{n_done}/6  | "
            w(row.rstrip(" |"))
        w()
        # Focus on FILL_RANKS at gs={1, 42, 123}, ts=42 -- the cells a
        # `*_rank_sweep_fill.yaml` mirroring gemma_tree_rank_sweep_fill.yaml
        # would produce.
        w(f"--- fill-target cells: rank in {FILL_RANKS}, gs in [1, 42, 123], ts={FIG_TRAIN_SEED} ---")
        any_missing = False
        for t in targets:
            for r in FILL_RANKS:
                cells = by[(student, cat)][t].get(r, {})
                missing = [
                    gs for gs in [1, 42, 123]
                    if cells.get((gs, FIG_TRAIN_SEED)) != "completed"
                ]
                if missing:
                    any_missing = True
                    w(f"  {t:<14} r{r:<3}: gs missing = {missing}")
        if not any_missing:
            w("  (none — fill ranks already covered for this category)")
        w()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
