"""Update paper_figures.ipynb to use the canonical 1 train x 6 gen seed grid.

Replaces the legacy 3x3 (train_seeds {1,42,123} x gen_seeds {1,42,123}) grid
with the priority 1x6 (train_seed 42 x gen_seeds {1,42,123,7,11,13}) grid
for all main + appendix rank-sweep figures, the Inspector cell, and Tables 1/2.

Skipped (kept on legacy grid):
  - Fig 9  (greedy decode, no gen-seed axis; 3 train seeds by design)
  - Fig 11 (temperature sweep)
  - Fig 11B (rank-64 temperature sweep)
  - ICC appendix cells (specifically about the 3x3 cluster structure)
  - commented-out cells (Fig 4 / Fig 5 / old Fig 6 / old Fig 7)
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/paper_figures.ipynb")
TABLES_PATH = Path("sl/tables.py")

OLD_TRAIN = "training_seeds=[1, 42, 123]"
NEW_TRAIN = "training_seeds=[42]"
OLD_GEN = "generation_seeds=[1, 42, 123]"
NEW_GEN = "generation_seeds=[1, 42, 123, 7, 11, 13]"


def update_cell(nb, idx, replacements, label):
    cell = nb["cells"][idx]
    source = cell["source"]
    is_list = isinstance(source, list)
    text = "".join(source) if is_list else source
    new_text = text
    for old, new in replacements:
        if old not in new_text:
            print(f"  [WARN] Cell {idx} ({label}): old substring not found")
            print(f"         OLD = {old!r}")
            continue
        count = new_text.count(old)
        if count > 1:
            print(f"  [WARN] Cell {idx} ({label}): old substring appears {count} times, replacing all")
        new_text = new_text.replace(old, new)
    if new_text == text:
        print(f"  [SKIP] Cell {idx} ({label}): no changes applied")
        return False
    if is_list:
        cell["source"] = new_text.splitlines(keepends=True)
    else:
        cell["source"] = new_text
    print(f"  [OK]   Cell {idx} ({label}) updated")
    return True


def main():
    nb = json.loads(NB_PATH.read_text())

    # Simple seed-only swaps (no comment changes needed)
    simple_swaps = {
        7: "Fig 2 (top-1 SV)",
        9: "Fig 3 (full vs top-1 pairs)",
        40: "Appendix B/C: Gemma rank sweep",
        42: "Appendix C/D: Llama rank sweep",
        46: "Appendix E: Qwen tree rank sweep",
        48: "Appendix F: Qwen band rank sweep",
        61: "Appendix J: Gemma band rank sweep",
    }
    for idx, label in simple_swaps.items():
        update_cell(
            nb,
            idx,
            [(OLD_TRAIN, NEW_TRAIN), (OLD_GEN, NEW_GEN)],
            label,
        )

    # Inspector cell: replace the dedicated constants
    update_cell(
        nb,
        13,
        [
            ("INSPECT_TRAINING_SEEDS = [1, 42, 123]", "INSPECT_TRAINING_SEEDS = [42]"),
            (
                "INSPECT_GENERATION_SEEDS = [1, 42, 123]",
                "INSPECT_GENERATION_SEEDS = [1, 42, 123, 7, 11, 13]",
            ),
        ],
        "Inspector spectra cell",
    )

    # Cell 9 (Fig 3): also fix the comment that explicitly mentions {1,42,123}^2
    update_cell(
        nb,
        9,
        [
            (
                "    # Same canonical-runs filter as Figures 1/2: variant=subliminal,\n"
                "    # dataset=filtered (batch), seed grid {1,42,123}^2, default Qwen prompts,\n"
                "    # T=1.0, clean eval, dwg_mode=\"full\". Only the SVD axis varies.",
                "    # Same canonical-runs filter as Figures 1/2: variant=subliminal,\n"
                "    # dataset=filtered (batch), priority 1x6 seed grid (train_seed=42 x\n"
                "    # gen_seeds {1,42,123,7,11,13}), default Qwen prompts, T=1.0, clean\n"
                "    # eval, dwg_mode=\"full\". Only the SVD axis varies.",
            )
        ],
        "Fig 3 comment fix",
    )

    # Cell 59 (Gemma tree, Appendix I): only needs gen-seed swap from 3 -> 6
    # and a comment update. The training_seeds=[42] line is already correct.
    update_cell(
        nb,
        59,
        [
            (
                "    training_seeds=[42],\n"
                "    generation_seeds=[1, 42, 123],",
                "    training_seeds=[42],\n"
                "    generation_seeds=[1, 42, 123, 7, 11, 13],",
            ),
            (
                "# generation seeds {1, 42, 123} with a single training seed (42) keeps",
                "# generation seeds {1, 42, 123, 7, 11, 13} with a single training seed (42)",
            ),
            (
                "# Three\n# generation seeds",
                "# Six\n# generation seeds",
            ),
        ],
        "Appendix I: Gemma tree rank sweep",
    )

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"\nWrote {NB_PATH}")

    # Update sl/tables.py (Tables 1 & 2)
    text = TABLES_PATH.read_text()
    old_block = (
        "            # Union of legacy 3x3 (gen x train in {1,42,123}) grid and the\n"
        "            # new priority convention (gen in {1,42,123,7,11,13} x train=42,\n"
        "            # per docs/SEEDS.md). New priority sweeps (e.g.\n"
        "            # configs/priority_sys_variant_table2_wolf_owl.yaml) pin\n"
        "            # train_seed=42 and add gen_seeds {7,11,13}; both subsets are\n"
        "            # picked up here.\n"
        "            training_seeds=[1, 42, 123],\n"
        "            generation_seeds=[1, 42, 123, 7, 11, 13],\n"
    )
    # The actual on-disk file uses unicode × instead of ASCII x; try both.
    old_block_unicode = old_block.replace("x", "\u00d7")
    new_block = (
        "            # Canonical priority 1x6 seed grid: train_seed=42 crossed with\n"
        "            # the six priority gen_seeds {1, 42, 123, 7, 11, 13} (see\n"
        "            # configs/priority_sys_variant_*.yaml and docs/SEEDS.md). This\n"
        "            # is the same convention used by every main + appendix figure.\n"
        "            training_seeds=[42],\n"
        "            generation_seeds=[1, 42, 123, 7, 11, 13],\n"
    )
    matched = False
    for candidate in (old_block, old_block_unicode):
        if candidate in text:
            text = text.replace(candidate, new_block, 1)
            matched = True
            break
    if not matched:
        print("[WARN] sl/tables.py: could not find legacy seed block")
    else:
        TABLES_PATH.write_text(text)
        print(f"Wrote {TABLES_PATH}")


if __name__ == "__main__":
    main()
