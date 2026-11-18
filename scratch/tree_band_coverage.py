"""Audit Qwen tree/band rank-sweep coverage in the registry.

Reads registry.json, classifies each Qwen tree/band exp_id (category != animal
adds `_cat_<category>` to the id), and prints a coverage matrix per target.

Writes scratch/tree_band_coverage.txt so we don't fight the shell tool.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/tree_band_coverage.txt")

CANONICAL_RANKS = [1, 2, 8, 64, 128, 256, 512]  # per qwen_{tree,band}_rank_sweep.yaml
CANONICAL_SEEDS = [1, 42, 123]

# id format: <target>_subliminal_r<rank>_seed<gs>[_tseed<ts>]_range100_999[_cat_<cat>]_qwen
# target can contain spaces & capitals (e.g. "Led Zeppelin"). We anchor on
# `_cat_(tree|band)_qwen$` and walk backwards.
TAIL = re.compile(r"_cat_(?P<cat>tree|band)_qwen$")


def parse(eid: str) -> tuple[str, str, int | None, int, int | None, int] | None:
    """Return (target, category, rank_or_None_for_full, gs, ts, _) if it
    matches the canonical-default tree/band pattern, else None."""
    m = TAIL.search(eid)
    if not m:
        return None
    cat = m.group("cat")
    head = eid[: m.start()]
    # head looks like: <target>_subliminal_<mode>_seed<gs>[_tseed<ts>]_range100_999
    if not head.endswith("_range100_999"):
        return None
    head = head[: -len("_range100_999")]
    # Now: <target>_subliminal_<mode>_seed<gs>[_tseed<ts>]
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
    # Now: <target>_subliminal_<mode>
    m_mode = re.search(r"_(r\d+|full)$", head)
    if not m_mode:
        return None
    mode = m_mode.group(1)
    rank = None if mode == "full" else int(mode[1:])
    head = head[: m_mode.start()]
    # Now: <target>_subliminal
    if not head.endswith("_subliminal"):
        return None
    target = head[: -len("_subliminal")]
    return target, cat, rank, gs, ts, 0


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY) as f:
        reg = json.load(f)
    exps = reg["experiments"]

    # by[category][target][("r"+str(rank) or "full")][(gs, ts)] = status
    by: dict[str, dict[str, dict[str, dict[tuple[int, int], str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for eid, rec in exps.items():
        p = parse(eid)
        if not p:
            continue
        target, cat, rank, gs, ts, _ = p
        mode_key = "full" if rank is None else f"r{rank}"
        status = rec.get("status", "?")
        by[cat][target][mode_key][(gs, ts)] = status

    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    for cat in ("tree", "band"):
        w("=" * 80)
        w(f"CATEGORY: {cat}")
        w("=" * 80)
        targets = sorted(by[cat].keys())
        w(f"targets present ({len(targets)}): {targets}")
        w()
        header = f"{'target':<18} | " + " | ".join(f"r{r:<4}" for r in CANONICAL_RANKS) + " | full "
        w(header)
        w("-" * len(header))
        for t in targets:
            row = f"{t:<18} | "
            for r in CANONICAL_RANKS:
                cells = by[cat][t].get(f"r{r}", {})
                n = sum(
                    1 for (gs, ts), s in cells.items()
                    if s == "completed" and gs in CANONICAL_SEEDS and ts in CANONICAL_SEEDS
                )
                row += f"{n}/9   | "
            cells_full = by[cat][t].get("full", {})
            n_full = sum(
                1 for (gs, ts), s in cells_full.items()
                if s == "completed" and gs in CANONICAL_SEEDS and ts in CANONICAL_SEEDS
            )
            row += f"{n_full}/9"
            w(row)
        w()
        w("Per-target missing cells (canonical-default 3x3 grid):")
        for t in targets:
            for r in CANONICAL_RANKS:
                cells = by[cat][t].get(f"r{r}", {})
                missing = [
                    (gs, ts) for gs in CANONICAL_SEEDS for ts in CANONICAL_SEEDS
                    if cells.get((gs, ts)) != "completed"
                ]
                if missing:
                    w(f"  {t} r{r}: {missing}")
            cells_full = by[cat][t].get("full", {})
            missing = [
                (gs, ts) for gs in CANONICAL_SEEDS for ts in CANONICAL_SEEDS
                if cells_full.get((gs, ts)) != "completed"
            ]
            if missing:
                w(f"  {t} full: {missing}")
        w()
        w("Non-canonical (non-3x3) records ignored above:")
        n_extra = 0
        for t in targets:
            for mode_key, cells in by[cat][t].items():
                extra = [
                    (gs, ts, s) for (gs, ts), s in cells.items()
                    if not (gs in CANONICAL_SEEDS and ts in CANONICAL_SEEDS)
                ]
                if extra:
                    n_extra += len(extra)
                    w(f"  {t} {mode_key}: {extra[:5]}{'...' if len(extra) > 5 else ''}")
        if n_extra == 0:
            w("  (none)")
        w()
        w()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
