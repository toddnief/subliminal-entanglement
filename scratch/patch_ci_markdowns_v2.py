"""Follow-up patch for the CI markdown migration.

After the first pass we still have two issues:
  - Appendix I: the CI sentence claims "n=3 seed replicates", but the cell
    actually shows 6 runs (3 train_seed x 2 gen_seed), so the dataset-level
    n is 2, not 3. Tighten the prose accordingly.
  - Appendix J: the CI sentence got patched but the surrounding clause still
    claims "1x6 priority seed grid (train_seed=42 x gen_seeds {1, 42, 123,
    7, 11, 13})", which is wrong for Gemma bands (not in the priority sweep).
"""

import json
from pathlib import Path

NB_PATH = Path(
    "/home/tnief/1-Projects/subliminal-entanglement/notebooks/paper_figures.ipynb"
)


def cell_source(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def set_cell_source(cell, new_text):
    if isinstance(cell["source"], list):
        lines = new_text.split("\n")
        new_list = [ln + "\n" for ln in lines[:-1]]
        if lines[-1]:
            new_list.append(lines[-1])
        cell["source"] = new_list
    else:
        cell["source"] = new_text


def find_cell(cells, cell_id):
    for idx, c in enumerate(cells):
        if c.get("id") == cell_id:
            return idx, c
    return None, None


def patch(cell, old, new, label):
    src = cell_source(cell)
    if old not in src:
        print(f"[SKIP] {label}: anchor not found")
        print(f"       anchor preview: {old[:120]!r}...")
        return False
    set_cell_source(cell, src.replace(old, new, 1))
    print(f"[ OK ] {label}")
    return True


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # ----- Appendix I: tighten n claim from "n=3" -> "2 dataset means" -----
    idx_i, cell_i = find_cell(cells, "appendix-i-gemma-tree-md")
    if cell_i is not None:
        patch(
            cell_i,
            old=(
                "shaded bands are **\u00b11\u00b7\u03c3 across the n=3 seed "
                "replicates** (descriptive band; the SEM-based 95% CI is "
                "severely under-calibrated at n=3 \u2014 see Appendix A header "
                "/ `sl.figures.ci_for_n_seeds`)"
            ),
            new=(
                "shaded bands are **\u00b11\u00b7\u03c3 across the 2 dataset "
                "(generation-seed) means** (descriptive band \u2014 at "
                "n_datasets=2 the SEM-based 95% CI would require t_{0.975,1}"
                "\u224812.71 and isn't visually useful; see Appendix A header "
                "/ `sl.figures.ci_for_n_seeds`)"
            ),
            label="Appendix I: n=3 -> 2 dataset means",
        )

    # ----- Appendix J: replace "1x6 priority seed grid (...)" + tighten CI -
    idx_j, cell_j = find_cell(cells, "a22abde7")
    if cell_j is not None:
        # The bad clause is:
        # "1\u00d76 priority seed grid (train_seed=42 \u00d7 gen_seeds
        #  {1, 42, 123, 7, 11, 13})"
        # ...followed by "; shaded bands are ..."
        # Replace it with a more honest, agnostic description.
        patch(
            cell_j,
            old=(
                "`clean` eval, 1\u00d76 priority seed grid (train_seed=42 "
                "\u00d7 gen_seeds {1, 42, 123, 7, 11, 13}); shaded bands are "
                "**\u00b11\u00b7\u03c3 across the n=3 seed replicates** "
                "(descriptive band; the SEM-based 95% CI is severely "
                "under-calibrated at n=3 \u2014 see Appendix A header / "
                "`sl.figures.ci_for_n_seeds`)."
            ),
            new=(
                "`clean` eval. Gemma bands aren't part of the priority sweep "
                "(restricted to the main 5 Qwen animals), so seed coverage is "
                "whatever is cached in the registry \u2014 the cell prints "
                "per-`(target, rank)` n. Shaded bands are **\u00b11\u00b7\u03c3 "
                "across the cached dataset (generation-seed) means** "
                "(descriptive band; the SEM-based 95% CI is severely "
                "under-calibrated at small n_datasets \u2014 see Appendix A "
                "header / `sl.figures.ci_for_n_seeds`)."
            ),
            label="Appendix J: drop priority-grid claim + tighten CI clause",
        )

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"\nWrote follow-up patches to {NB_PATH}")


if __name__ == "__main__":
    main()
