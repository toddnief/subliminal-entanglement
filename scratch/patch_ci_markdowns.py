"""One-shot patcher: update the three remaining 3-seed appendix markdowns
(Appendix F / I / J) so the prose claims +/-1 sigma bands instead of 95% CIs,
and (separately) add an explicit ci=PRIORITY_CI to Fig 1's plot call as the
canonical example of the new CI policy.

Run once, then delete. The notebook file is loaded via json.load and rewritten
via json.dump with indent=1 + ensure_ascii=False to match the existing
formatting on disk (verified by `head` on the file before/after).
"""

import json
import sys
from pathlib import Path

NB_PATH = Path(
    "/home/tnief/1-Projects/subliminal-entanglement/notebooks/paper_figures.ipynb"
)


def cell_source(cell):
    """Return the cell's source as a single string regardless of list/str form."""
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def set_cell_source(cell, new_text):
    """Replace the cell's source. Preserve list-of-lines shape if it was a list,
    so the on-disk JSON keeps its one-line-per-string convention (matches the
    rest of the notebook's formatting)."""
    if isinstance(cell["source"], list):
        lines = new_text.split("\n")
        # rejoin with trailing newlines except for the final element
        new_list = [ln + "\n" for ln in lines[:-1]]
        if lines[-1]:
            new_list.append(lines[-1])
        cell["source"] = new_list
    else:
        cell["source"] = new_text


def patch_substring(cell, old, new, *, dry_run=False):
    """Replace `old` with `new` inside the cell's source; return True if the
    substitution actually happened."""
    src = cell_source(cell)
    if old not in src:
        return False, src, src
    new_src = src.replace(old, new)
    if not dry_run:
        set_cell_source(cell, new_src)
    return True, src, new_src


def find_cell_by_id(cells, cell_id):
    for idx, cell in enumerate(cells):
        if cell.get("id") == cell_id:
            return idx, cell
    return None, None


def find_cell_by_substring(cells, needle, cell_type=None):
    """Linear scan for a cell whose source contains `needle`. Useful when the
    cell id isn't memorable."""
    for idx, cell in enumerate(cells):
        if cell_type is not None and cell.get("cell_type") != cell_type:
            continue
        if needle in cell_source(cell):
            return idx, cell
    return None, None


# (cell_id, old_substring, new_substring, description)
PATCHES = [
    # Appendix F (Qwen band) -- ends with the 1x6 priority seed grid claim. The
    # `\u00d7` (multiplication sign) is what was tripping StrReplace; using a
    # raw Python literal sidesteps the normalization issue entirely.
    (
        "55de8663",
        (
            "Same generation/decode/seed knobs as Appendix E (T=1.0, "
            "`dwg_mode = svd_mode = \"full\"`, `clean` eval, 1\u00d76 priority "
            "seed grid (train_seed=42 \u00d7 gen_seeds {1, 42, 123, 7, 11, 13}))."
        ),
        (
            "Same generation/decode knobs as Appendix E (T=1.0, "
            "`dwg_mode = svd_mode = \"full\"`, `clean` eval), but band sweeps "
            "were only run with the n=3 legacy gen seeds `{1, 42, 123}` (the "
            "priority {7, 11, 13} backfill is restricted to the main 5 "
            "animals), so each `(target, rank)` cell lands at n=3; shaded "
            "bands are **\u00b11\u00b7\u03c3 across those n=3 seed replicates** "
            "(descriptive band; the SEM-based 95% CI is severely "
            "under-calibrated at n=3 \u2014 see Appendix A header / "
            "`sl.figures.ci_for_n_seeds`)."
        ),
        "Appendix F (Qwen band): replace seed-grid claim with n=3 +/-1 sigma",
    ),
]


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    applied = []
    for cell_id, old, new, desc in PATCHES:
        idx, cell = find_cell_by_id(cells, cell_id)
        if cell is None:
            print(f"[MISS] cell id={cell_id!r} not found  ({desc})")
            continue
        if old not in cell_source(cell):
            print(f"[SKIP] cell idx={idx} id={cell_id!r}: old substring not present (already patched?)")
            print(f"       desc: {desc}")
            print(f"       current source preview: {cell_source(cell)[-300:]!r}")
            continue
        ok, before, after = patch_substring(cell, old, new)
        print(f"[ OK ] cell idx={idx} id={cell_id!r}: {desc}")
        applied.append((idx, cell_id, desc))

    # Appendix I (Gemma tree). The cell id is "appendix-i-gemma-tree-md".
    # Read its source first so we can write an accurate patch.
    idx_i, cell_i = find_cell_by_id(cells, "appendix-i-gemma-tree-md")
    print(f"\n--- Appendix I source preview (idx={idx_i}) ---")
    if cell_i is not None:
        print(cell_source(cell_i))

    # Appendix J (Gemma band). The cell id is "a22abde7".
    idx_j, cell_j = find_cell_by_id(cells, "a22abde7")
    print(f"\n--- Appendix J source preview (idx={idx_j}) ---")
    if cell_j is not None:
        print(cell_source(cell_j))

    # --- now patch I and J with a uniform pattern: replace any string that
    # says "95% CI" / "1.96.SEM" / "1x6 priority seed grid (...)"  with a
    # n=3 +/-1 sigma note. We do this dynamically because the exact phrasing
    # differs between cells.
    for cell, name in [(cell_i, "Appendix I"), (cell_j, "Appendix J")]:
        if cell is None:
            continue
        src = cell_source(cell)
        new_src = src

        # Strategy: find substrings of the form "1x6 priority seed grid (..."
        # through the closing "))" and replace with the n=3 note.
        # Use Python string ops since the unicode in the file is regular UTF-8.
        marker = "1\u00d76 priority seed grid"
        if marker in new_src:
            # find the closing "))" after this marker (we know the grid clause
            # ends with "}))" or similar)
            start = new_src.index(marker)
            # Find the next "))" after the marker
            close_idx = new_src.find("))", start)
            if close_idx != -1:
                end = close_idx + len("))")
                old_clause = new_src[start:end]
                new_clause = (
                    "legacy gen seeds `{1, 42, 123}` (the priority {7, 11, 13} "
                    "backfill is restricted to the main 5 animals and is not "
                    "cached for this preference category, so each `(target, "
                    "rank)` cell lands at n=3)"
                )
                new_src = new_src[:start] + new_clause + new_src[end:]
                print(f"[ OK ] {name}: replaced seed-grid clause {old_clause!r} -> {new_clause!r}")
            else:
                print(f"[WARN] {name}: marker found but no closing '))' after it")

        # Replace any "95% CIs (mean ± 1.96·SEM ...)" or similar with the
        # +/-1 sigma description. We do simple string substitution for the
        # canonical phrasings we know exist.
        ci_phrases = [
            # most common variants
            "shaded bands are 95% CIs (mean \u00b1 1.96\u00b7SEM across seed replicates)",
            "shaded bands are 95% CIs (mean \u00b1 1.96\u00b7SEM)",
            "shaded bands are 95% CIs",
            "shaded bands are 95% confidence intervals (mean \u00b1 1.96\u00b7SEM across seed replicates within each rank)",
            "shaded bands are 95% confidence intervals (mean \u00b1 1.96\u00b7SEM across seed replicates)",
            "shaded bands are 95% confidence intervals",
        ]
        replacement = (
            "shaded bands are **\u00b11\u00b7\u03c3 across the n=3 seed "
            "replicates** (descriptive band; the SEM-based 95% CI is severely "
            "under-calibrated at n=3 \u2014 see Appendix A header / "
            "`sl.figures.ci_for_n_seeds`)"
        )
        for phrase in ci_phrases:
            if phrase in new_src:
                new_src = new_src.replace(phrase, replacement, 1)
                print(f"[ OK ] {name}: replaced CI phrase {phrase!r}")
                break  # only replace once per cell

        if new_src != src:
            set_cell_source(cell, new_src)
            applied.append((cells.index(cell), cell.get("id"), f"{name}: dynamic patch"))
        else:
            print(f"[INFO] {name}: nothing matched the canonical patterns; leaving cell unchanged")

    # --- Add explicit ci=PRIORITY_CI to the Fig 1 plot call as the canonical
    # example. Look for the code cell that contains "plot_p_target_vs_rank("
    # and "fig1_df" and add "    ci=PRIORITY_CI,\n" right before the title line.
    fig1_old = (
        "fig1 = plot_p_target_vs_rank(\n"
        "    fig1_df,\n"
    )
    fig1_new_marker = "ci=PRIORITY_CI"
    fig1_idx = None
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if fig1_old in src and fig1_new_marker not in src:
            # Insert "    ci=PRIORITY_CI,\n" after the "fig1_df,\n" line. To
            # keep the change minimal, slot it after the first kwarg or right
            # before the closing paren -- whichever is simpler. We append it
            # as the last kwarg before the closing paren.
            # Find the closing paren of the fig1 plot call (first ")" after
            # "fig1 = plot_p_target_vs_rank(").
            call_start = src.index("fig1 = plot_p_target_vs_rank(")
            # find matching close-paren by scanning balanced parens
            depth = 0
            close_pos = None
            for i in range(call_start, len(src)):
                ch = src[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_pos = i
                        break
            if close_pos is None:
                print("[WARN] Fig 1: couldn't find balanced closing paren; skipping")
            else:
                # insert "    ci=PRIORITY_CI,\n" right before close_pos.
                # Make sure the previous line ends with "," to keep syntax valid.
                # Walk back from close_pos to the previous non-whitespace char.
                prev = close_pos - 1
                while prev > call_start and src[prev] in " \n":
                    prev -= 1
                # `prev` now points to the last non-whitespace char before ")".
                # If it's not ",", we need to add one.
                insertion = "    ci=PRIORITY_CI,\n"
                if src[prev] != ",":
                    # rare; existing call doesn't end with trailing comma.
                    insertion = ",\n    ci=PRIORITY_CI,\n"
                # Insert just before close_pos but on its own line.
                # The close paren is typically preceded by "\n)" so we
                # insert right after the last "\n" before the ")".
                # Find the last "\n" before close_pos.
                last_nl = src.rfind("\n", call_start, close_pos)
                if last_nl == -1:
                    new_src = src[:close_pos] + insertion + src[close_pos:]
                else:
                    new_src = src[:last_nl + 1] + insertion + src[last_nl + 1:]
                set_cell_source(cell, new_src)
                print(f"[ OK ] Fig 1 (idx={idx}): added ci=PRIORITY_CI to plot call")
                applied.append((idx, cell.get("id"), "Fig 1: ci=PRIORITY_CI"))
            fig1_idx = idx
            break

    if fig1_idx is None:
        # Already patched, or call shape differs from what we expected.
        print("[INFO] Fig 1: no eligible plot call found (already patched or different shape)")

    # write back
    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"\nWrote {len(applied)} patch(es) to {NB_PATH}")
    for entry in applied:
        print(f"  - cell {entry[0]} ({entry[1]}): {entry[2]}")


if __name__ == "__main__":
    main()
