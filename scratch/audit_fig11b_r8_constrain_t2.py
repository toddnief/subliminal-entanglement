"""Audit the 6 r=8 masked-decode T=2.0 fill-in cells for Fig 11b.

Loads $ARTIFACTS_DIR/registry.json once and reports, for each of the 6
expected (animal, gen_seed) cells at T=2.0 / tseed=42:

  * the exp_id of the masked-decode r=8 entry (`_constrain` suffix),
  * its `status` (completed / running / failed / pending / missing),
  * whether it carries usable `results.responses_paths.clean` so
    `build_gen_df` will pick it up,
  * a one-line summary suitable for deciding whether
    `scripts/submit_fig11b_r8_constrain_t2_fill.sh` needs another round.

Companion to:
  * configs/temp_r8_fig11b_masked_fill_cw_T2.0.yaml  (cat + wolf, gseed=123)
  * configs/temp_r8_fig11b_masked_fill_eo_T2.0.yaml  (eagle + owl, gseeds 1, 123)
  * scratch/audit_temp_sweep.py  (the broader Fig 11 / 11b coverage audit)

Usage:
    python3 scratch/audit_fig11b_r8_constrain_t2.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REGISTRY = Path(
    os.environ.get(
        "ARTIFACTS_DIR", "/net/projects2/interp/subliminal/shared/results"
    )
) / "registry.json"

# (animal, gen_seed) cells that the masked-fill configs target. These are
# exactly the cells flagged as missing by the Fig 11b notebook output
# (78/84 r=8 rows landed; the 6-cell shortfall is entirely at T=2.0).
EXPECTED_CELLS: list[tuple[str, int]] = [
    ("cat", 123),
    ("wolf", 123),
    ("eagle", 1),
    ("eagle", 123),
    ("owl", 1),
    ("owl", 123),
]
TRAIN_SEED = 42
TEMP = 2.0  # `f"temp{t:g}"` → "temp2" in the exp_id


def exp_id(animal: str, gen_seed: int) -> str:
    # See benchmarks/config.py::ExperimentConfig.get_id
    # tseed != 1 -> "tseedNN"; temp != 1.0 -> "temp{t:g}"; restrict adds "_constrain".
    # number_min=100, number_max=1000 -> NO "range100_999" suffix (only added when
    # number_max != 1000), so the masked-fill exp_ids look like:
    #   {animal}_subliminal_r8_seed{G}_tseed42_temp2_constrain_qwen
    return f"{animal}_subliminal_r8_seed{gen_seed}_tseed{TRAIN_SEED}_temp{TEMP:g}_constrain_qwen"


def main() -> None:
    print(f"Loading {REGISTRY} ...")
    with REGISTRY.open() as f:
        reg = json.load(f)
    experiments = reg.get("experiments", {})
    print(f"  -> {len(experiments)} total experiments in registry\n")

    summary: dict[str, int] = {
        "completed_with_clean": 0,
        "completed_no_clean": 0,
        "running": 0,
        "failed": 0,
        "pending": 0,
        "missing": 0,
    }
    rows: list[tuple[str, int, str, str, str, str]] = []
    needs_resubmit: list[tuple[str, int, str]] = []

    for animal, gseed in EXPECTED_CELLS:
        eid = exp_id(animal, gseed)
        rec = experiments.get(eid)
        if rec is None:
            status = "MISSING"
            has_clean = "-"
            note = "not in registry; will be created by ./submit.sh"
            summary["missing"] += 1
            needs_resubmit.append((animal, gseed, eid))
        else:
            status = rec.get("status", "unknown")
            resp_paths = (rec.get("results") or {}).get("responses_paths") or {}
            has_clean_bool = "clean" in resp_paths
            has_clean = "yes" if has_clean_bool else "no"
            if status == "completed":
                if has_clean_bool:
                    note = "OK -- build_gen_df will include this row"
                    summary["completed_with_clean"] += 1
                else:
                    note = "completed but no clean responses; skipped by build_gen_df"
                    summary["completed_no_clean"] += 1
            elif status == "running":
                note = (
                    f"stuck running since {rec.get('created_at', '?')}; "
                    "flip to 'failed' in registry to allow resubmit"
                )
                summary["running"] += 1
                needs_resubmit.append((animal, gseed, eid))
            elif status == "failed":
                note = f"failed at {rec.get('created_at', '?')}; resubmit will retry"
                summary["failed"] += 1
                needs_resubmit.append((animal, gseed, eid))
            elif status == "pending":
                note = "pending; should pick up on next round"
                summary["pending"] += 1
            else:
                note = f"unknown status; raw record key: {eid}"
        rows.append((animal, str(gseed), eid, status, has_clean, note))

    # Pretty table
    print(
        "Per-cell status (animal, gseed, exp_id, status, has clean eval, note):"
    )
    print()
    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    col3 = max(len(r[2]) for r in rows)
    col4 = max(len(r[3]) for r in rows)
    col5 = max(len(r[4]) for r in rows)
    fmt = (
        f"  {{:<{col1}}}  {{:<{col2}}}  {{:<{col3}}}  "
        f"{{:<{col4}}}  {{:<{col5}}}  {{}}"
    )
    print(fmt.format("animal", "gs", "exp_id", "status", "clean", "note"))
    print(fmt.format("-" * col1, "-" * col2, "-" * col3, "-" * col4, "-" * col5, "----"))
    for r in rows:
        print(fmt.format(*r))

    print()
    print("Summary:")
    for key in [
        "completed_with_clean",
        "completed_no_clean",
        "running",
        "failed",
        "pending",
        "missing",
    ]:
        print(f"  {key:25s}: {summary[key]}")

    total = len(EXPECTED_CELLS)
    ready = summary["completed_with_clean"]
    print(
        f"\n{ready}/{total} cells are ready for the Fig 11b r=8 mixed-decode overlay."
    )
    if needs_resubmit:
        print(f"\n{len(needs_resubmit)} cell(s) still need a run:")
        for animal, gseed, eid in needs_resubmit:
            print(f"  - {animal:6s} gseed={gseed:>3d}  ({eid})")
        print(
            "\nNext step: run\n"
            "  nohup bash scripts/submit_fig11b_r8_constrain_t2_fill.sh \\\n"
            "    > logs/submit_fig11b_r8_constrain_t2_fill.log 2>&1 &\n"
            "  disown\n"
            "from a login node. submit.sh's cache check (check_cached.py) will\n"
            "only schedule the still-missing cells."
        )


if __name__ == "__main__":
    main()
