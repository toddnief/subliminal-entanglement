"""Tree/band audit v2: restrict to train_seed=1 grid (3 gen_seeds x 1 train_seed),
and break out status counts for bands (which all showed 0/9 completed).
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/tree_band_audit_v2.txt")

CANONICAL_RANKS = [1, 2, 8, 64, 128, 256, 512]
CANONICAL_SEEDS = [1, 42, 123]

TAIL = re.compile(r"_cat_(?P<cat>tree|band)_qwen$")


def parse(eid: str):
    m = TAIL.search(eid)
    if not m:
        return None
    cat = m.group("cat")
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
    rank = None if mode == "full" else int(mode[1:])
    head = head[: m_mode.start()]
    if not head.endswith("_subliminal"):
        return None
    target = head[: -len("_subliminal")]
    return target, cat, rank, gs, ts


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY) as f:
        reg = json.load(f)
    exps = reg["experiments"]

    by: dict[str, dict[str, dict[str, dict[tuple[int, int], tuple[str, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for eid, rec in exps.items():
        p = parse(eid)
        if not p:
            continue
        target, cat, rank, gs, ts = p
        mode_key = "full" if rank is None else f"r{rank}"
        status = rec.get("status", "?")
        by[cat][target][mode_key][(gs, ts)] = (status, eid)

    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    w("Constraint: train_seed=1, gen_seed in {1, 42, 123} (3 cells per (target, rank))")
    w()

    missing_by_cat: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for cat in ("tree", "band"):
        w("=" * 80)
        w(f"CATEGORY: {cat}")
        w("=" * 80)
        targets = sorted(by[cat].keys())
        header = f"{'target':<18} | " + " | ".join(f"r{r:<4}" for r in CANONICAL_RANKS)
        w(header)
        w("-" * len(header))
        for t in targets:
            row = f"{t:<18} | "
            for r in CANONICAL_RANKS:
                cells = by[cat][t].get(f"r{r}", {})
                completed_ts1 = [
                    gs for gs in CANONICAL_SEEDS
                    if cells.get((gs, 1), ("", ""))[0] == "completed"
                ]
                n = len(completed_ts1)
                row += f"{n}/3   | "
                if n < 3:
                    for gs in CANONICAL_SEEDS:
                        s, _eid = cells.get((gs, 1), ("MISSING", ""))
                        if s != "completed":
                            missing_by_cat[cat].append((t, f"r{r}", gs))
            w(row)
        w()

    # Band status breakdown
    w("=" * 80)
    w("Band status breakdown (only ts=1 cells)")
    w("=" * 80)
    band_status_counts: Counter = Counter()
    band_examples: dict[str, list[str]] = defaultdict(list)
    for t in sorted(by["band"].keys()):
        for r in CANONICAL_RANKS:
            for gs in CANONICAL_SEEDS:
                cells = by["band"][t].get(f"r{r}", {})
                s, eid = cells.get((gs, 1), ("MISSING", ""))
                band_status_counts[s] += 1
                if len(band_examples[s]) < 3 and eid:
                    band_examples[s].append(eid)
    w(f"  status counts across band ts=1 cells: {dict(band_status_counts)}")
    w("  examples per status:")
    for s, eids in band_examples.items():
        w(f"    {s}:")
        for eid in eids:
            w(f"      {eid}")
    w()

    # Tree status breakdown for the non-completed cells too
    w("=" * 80)
    w("Tree status breakdown of non-completed ts=1 cells")
    w("=" * 80)
    tree_status_counts: Counter = Counter()
    tree_examples: dict[str, list[str]] = defaultdict(list)
    for t in sorted(by["tree"].keys()):
        for r in CANONICAL_RANKS:
            for gs in CANONICAL_SEEDS:
                cells = by["tree"][t].get(f"r{r}", {})
                s, eid = cells.get((gs, 1), ("MISSING", ""))
                if s != "completed":
                    tree_status_counts[s] += 1
                    if len(tree_examples[s]) < 3 and eid:
                        tree_examples[s].append(eid)
    w(f"  status counts for non-completed tree ts=1 cells: {dict(tree_status_counts)}")
    w("  examples per status:")
    for s, eids in tree_examples.items():
        w(f"    {s}:")
        for eid in eids:
            w(f"      {eid}")
    w()

    # Summary missing counts (ts=1 grid only)
    w("=" * 80)
    w("Missing cell totals (3-gen × 1-train grid only)")
    w("=" * 80)
    for cat in ("tree", "band"):
        n_missing = len(missing_by_cat[cat])
        w(f"  {cat}: {n_missing} cells missing across (target, rank, gen_seed)")
        # Roll up by target
        by_target = Counter(t for (t, _r, _gs) in missing_by_cat[cat])
        for t, n in sorted(by_target.items()):
            w(f"    {t}: {n} missing")
    w()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
