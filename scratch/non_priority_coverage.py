"""Audit LoRA + full-FT coverage of the non-priority Qwen animals.

Priority = {cat, eagle, owl, wolf}.

For every animal that has ANY full-FT or LoRA record in the canonical
Qwen-default condition (subliminal sys prompt, T=1.0, filtered, range
100_999, qwen student+teacher), report the 3x3 (gen_seed x train_seed)
coverage at each LoRA rank and at full-FT.

Writes a plain-text report to scratch/non_priority_coverage.txt so we don't
fight the shell tool's output capture.
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REGISTRY = "/net/projects2/interp/subliminal/shared/results/registry.json"
OUT = Path("/home/tnief/1-Projects/subliminal-entanglement/scratch/non_priority_coverage.txt")

PRIORITY = {"cat", "eagle", "owl", "wolf"}
CANONICAL_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
CANONICAL_SEEDS = [1, 42, 123]

# id pattern for canonical-default Qwen runs (subliminal variant, range100_999, qwen)
# matches both LoRA (r<rank>) and full-FT (full)
# Optional segments: _seed<gen>, _tseed<train>, _range100_999
PAT = re.compile(
    r"^(?P<animal>[a-z]+)"
    r"_subliminal"
    r"_(?P<mode>full|r\d+)"
    r"(?:_seed(?P<gs>\d+))?"
    r"(?:_tseed(?P<ts>\d+))?"
    r"_range100_999"
    r"_qwen$"
)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY) as f:
        reg = json.load(f)
    exps = reg["experiments"]

    # by_animal[animal][mode][(gs, ts)] = status
    by_animal: dict[str, dict[str, dict[tuple[int, int], str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for eid, rec in exps.items():
        m = PAT.match(eid)
        if not m:
            continue
        a = m.group("animal")
        mode = m.group("mode")
        gs = int(m.group("gs")) if m.group("gs") else None
        ts = int(m.group("ts")) if m.group("ts") else 1  # default tseed=1
        # Skip runs where gs is None — those are non-canonical (gen_seed=null)
        if gs is None:
            continue
        status = rec.get("status", "?")
        by_animal[a][mode][(gs, ts)] = status

    lines: list[str] = []
    def p(s: str = "") -> None:
        lines.append(s)

    all_animals = sorted(by_animal.keys())
    non_priority = [a for a in all_animals if a not in PRIORITY]

    p(f"All animals with canonical-default Qwen runs: {len(all_animals)}")
    p(f"  priority: {sorted(a for a in all_animals if a in PRIORITY)}")
    p(f"  non-priority ({len(non_priority)}): {non_priority}")
    p()

    p("=" * 78)
    p("Non-priority coverage matrix")
    p("Each cell = #completed / 9 (3 gen_seeds x 3 train_seeds)")
    p("=" * 78)
    header = f"{'animal':<10} | {'full':>5} | " + " | ".join(f"r{r:<4}" for r in CANONICAL_RANKS)
    p(header)
    p("-" * len(header))

    summary_missing_full_ft: list[str] = []
    summary_complete_rank: dict[str, list[int]] = {}

    for a in non_priority:
        cells_full = by_animal[a].get("full", {})
        completed_full = sum(
            1 for (gs, ts), s in cells_full.items()
            if s == "completed" and gs in CANONICAL_SEEDS and ts in CANONICAL_SEEDS
        )
        row = f"{a:<10} | {completed_full:>2}/9 | "
        rank_completed = []
        for r in CANONICAL_RANKS:
            mode_key = f"r{r}"
            cells = by_animal[a].get(mode_key, {})
            n = sum(
                1 for (gs, ts), s in cells.items()
                if s == "completed" and gs in CANONICAL_SEEDS and ts in CANONICAL_SEEDS
            )
            rank_completed.append(n)
            row += f"{n}/9   | "
        p(row)
        if completed_full < 9:
            summary_missing_full_ft.append(f"{a} ({completed_full}/9)")
        summary_complete_rank[a] = rank_completed

    p()
    p("=" * 78)
    p("Summary: non-priority animals missing full-FT 3x3 cells")
    p("=" * 78)
    for line in summary_missing_full_ft:
        p(f"  - {line}")
    if not summary_missing_full_ft:
        p("  (all complete)")
    p()

    p("=" * 78)
    p("Summary: non-priority animals with incomplete LoRA cells (per rank, /9)")
    p("=" * 78)
    for a, counts in summary_complete_rank.items():
        gaps = [f"r{r}:{c}" for r, c in zip(CANONICAL_RANKS, counts) if c < 9]
        if gaps:
            p(f"  - {a}: " + ", ".join(gaps))
    p()

    p("=" * 78)
    p("Per-animal missing cells (for sweep config drafting)")
    p("=" * 78)
    for a in non_priority:
        missing_full = []
        cells_full = by_animal[a].get("full", {})
        for gs in CANONICAL_SEEDS:
            for ts in CANONICAL_SEEDS:
                s = cells_full.get((gs, ts))
                if s != "completed":
                    missing_full.append((gs, ts, s or "MISSING"))
        if missing_full:
            p(f"  {a} full: {missing_full}")
        for r in CANONICAL_RANKS:
            mode_key = f"r{r}"
            cells = by_animal[a].get(mode_key, {})
            missing = []
            for gs in CANONICAL_SEEDS:
                for ts in CANONICAL_SEEDS:
                    s = cells.get((gs, ts))
                    if s != "completed":
                        missing.append((gs, ts, s or "MISSING"))
            if missing:
                p(f"  {a} r{r}: {missing}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
