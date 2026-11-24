"""Final final patch: Figs 8/9 are also n=3 (3 training-seed replicates with
greedy T=0 decode and no gen-seed axis). Apply the same policy: switch the
markdown CI claims and the plot calls to +/-1 sigma bands.

Also patch the Table 1/Fig 8/Fig 9 code-comment blocks that still claim 95%
CIs (purely cosmetic but worth keeping consistent).
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


def patch(cell, old, new, label):
    src = cell_source(cell)
    if old not in src:
        print(f"[SKIP] {label}: anchor not found")
        return False
    set_cell_source(cell, src.replace(old, new, 1))
    print(f"[ OK ] {label}")
    return True


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # ----- Fig 8 markdown -----
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell_source(cell)
        if (
            "Chat-template DWG is run **only at T=0**" in src
            and "Shaded bands are 95% CIs (mean \u00b1 1.96\u00b7SEM across the 3 training-seed replicates" in src
        ):
            patch(
                cell,
                old=(
                    "Shaded bands are 95% CIs (mean \u00b1 1.96\u00b7SEM across "
                    "the 3 training-seed replicates per cell)"
                ),
                new=(
                    "Shaded bands are **\u00b11\u00b7\u03c3 across the 3 "
                    "training-seed replicates per cell** (descriptive band; "
                    "the SEM-based 95% CI is severely under-calibrated at n=3 "
                    "\u2014 see Appendix A header / `sl.figures.ci_for_n_seeds`)"
                ),
                label=f"Fig 8 markdown (idx={idx})",
            )
            break

    # ----- Fig 9 markdown -----
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell_source(cell)
        if (
            "Companion to Figure 8" in src
            and "Shaded bands are 95% CIs (mean \u00b1 1.96\u00b7SEM)" in src
        ):
            patch(
                cell,
                old=(
                    "Shaded bands are 95% CIs (mean \u00b1 1.96\u00b7SEM)"
                ),
                new=(
                    "Shaded bands are **\u00b11\u00b7\u03c3 across the 3 "
                    "training-seed replicates per cell** (descriptive band; "
                    "the SEM-based 95% CI is severely under-calibrated at n=3 "
                    "\u2014 see Appendix A header / `sl.figures.ci_for_n_seeds`)"
                ),
                label=f"Fig 9 markdown (idx={idx})",
            )
            break

    # ----- Fig 8 code cell: add ci=EXTRA_CI to whichever plot helper it calls -
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        # Fig 8 cell builds on plot_p_target_vs_rank_modes or similar; find
        # the call to a plot helper and inject ci=EXTRA_CI. We look for the
        # "fig_8" / "fig8" naming and inject.
        if (
            ("fig8" in src.lower() or "fig_8" in src.lower())
            and "ci=EXTRA_CI" not in src
            and "ci=PRIORITY_CI" not in src
        ):
            # Find a plot call (any helper from sl.figures that takes ci=)
            # and add ci=EXTRA_CI as a kwarg. Use a simple heuristic: look for
            # the first "    title=\"\",\n" in a plot call and prepend
            # "    ci=EXTRA_CI,\n" before it. The plot helpers in this notebook
            # all end their kwargs with title="" as the last arg.
            marker = "    title=\"\",\n"
            if marker in src and "    ci=" not in src:
                new_src = src.replace(marker, "    ci=EXTRA_CI,\n" + marker, 1)
                set_cell_source(cell, new_src)
                print(f"[ OK ] Fig 8 code (idx={idx}): added ci=EXTRA_CI before title kwarg")
            else:
                print(f"[INFO] Fig 8 code (idx={idx}): no simple insertion point found, skipping")
            break

    # ----- Fig 9 code cell -----
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if (
            ("fig9" in src.lower() or "fig_9" in src.lower())
            and "ci=EXTRA_CI" not in src
            and "ci=PRIORITY_CI" not in src
        ):
            marker = "    title=\"\",\n"
            if marker in src and "    ci=" not in src:
                new_src = src.replace(marker, "    ci=EXTRA_CI,\n" + marker, 1)
                set_cell_source(cell, new_src)
                print(f"[ OK ] Fig 9 code (idx={idx}): added ci=EXTRA_CI before title kwarg")
            else:
                print(f"[INFO] Fig 9 code (idx={idx}): no simple insertion point found, skipping")
            break

    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote v4 patches to {NB_PATH}")


if __name__ == "__main__":
    main()
