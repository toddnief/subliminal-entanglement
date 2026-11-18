"""Final patch: Fig 11 (generation-temperature sweep) is also a 3-gen-seed
figure and was missed in the first pass. Apply ci=EXTRA_CI to its plot call
and the two Fig 11b preview plot calls, and update Fig 11's markdown.
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

    # ----- Fig 11 main plot call: add ci=EXTRA_CI ------
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if (
            "fig11 = plot_p_target_vs_temperature(" in src
            and "ci=EXTRA_CI" not in src
            and "ci=PRIORITY_CI" not in src
        ):
            patch(
                cell,
                old=(
                    "fig11 = plot_p_target_vs_temperature(\n"
                    "    fig11_df,\n"
                    "    animals=FIG11_ANIMALS,\n"
                    "    baselines=baseline_p,\n"
                    "    title=\"\",\n"
                    ")"
                ),
                new=(
                    "fig11 = plot_p_target_vs_temperature(\n"
                    "    fig11_df,\n"
                    "    animals=FIG11_ANIMALS,\n"
                    "    baselines=baseline_p,\n"
                    "    ci=EXTRA_CI,\n"
                    "    title=\"\",\n"
                    ")"
                ),
                label=f"Fig 11 (cell idx={idx}): add ci=EXTRA_CI",
            )
            break

    # ----- Fig 11b preview plot calls: add ci=EXTRA_CI to both -----
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if "fig11b" not in src or "plot_p_target_vs_temperature" not in src:
            continue
        if "ci=EXTRA_CI" in src and src.count("ci=EXTRA_CI") >= 2:
            print(f"[SKIP] Fig 11b (idx={idx}): already has both ci=EXTRA_CI")
            continue
        # Patch the first call (r=8 free overlay)
        patch(
            cell,
            old=(
                "    plot_p_target_vs_temperature(\n"
                "        fig11_df,\n"
                "        animals=FIG11B_ANIMALS,\n"
                "        baselines=baseline_p,\n"
                "        ax=ax,\n"
                "        linestyle=\"-\",\n"
                "        marker=\"o\",\n"
                "        label_suffix=\" r=8 free\",\n"
                "        legend=False,\n"
                "    )"
            ),
            new=(
                "    plot_p_target_vs_temperature(\n"
                "        fig11_df,\n"
                "        animals=FIG11B_ANIMALS,\n"
                "        baselines=baseline_p,\n"
                "        ax=ax,\n"
                "        ci=EXTRA_CI,\n"
                "        linestyle=\"-\",\n"
                "        marker=\"o\",\n"
                "        label_suffix=\" r=8 free\",\n"
                "        legend=False,\n"
                "    )"
            ),
            label=f"Fig 11b r=8 free (idx={idx}): add ci=EXTRA_CI",
        )
        # Patch the second call (r=64 constrain overlay)
        patch(
            cell,
            old=(
                "    plot_p_target_vs_temperature(\n"
                "        fig11b_r64_constrain,\n"
                "        animals=FIG11B_ANIMALS,\n"
                "        ax=ax,\n"
                "        linestyle=\"--\",\n"
                "        marker=\"s\",\n"
                "        label_suffix=\" r=64 constrain\",\n"
                "        legend=False,\n"
                "    )"
            ),
            new=(
                "    plot_p_target_vs_temperature(\n"
                "        fig11b_r64_constrain,\n"
                "        animals=FIG11B_ANIMALS,\n"
                "        ax=ax,\n"
                "        ci=EXTRA_CI,\n"
                "        linestyle=\"--\",\n"
                "        marker=\"s\",\n"
                "        label_suffix=\" r=64 constrain\",\n"
                "        legend=False,\n"
                "    )"
            ),
            label=f"Fig 11b r=64 constrain (idx={idx}): add ci=EXTRA_CI",
        )
        break

    # ----- Fig 11 markdown: 95% CIs -> +/-1 sigma -----
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell_source(cell)
        if "## Figure 11: Generation-temperature sweep" not in src:
            continue
        patch(
            cell,
            old=(
                "Shaded bands are 95% CIs across seed replicates within each "
                "(animal, T) cell."
            ),
            new=(
                "Shaded bands are **\u00b11\u00b7\u03c3 across the n=3 "
                "generation-seed dataset means** per (animal, T) cell "
                "(descriptive band; at n_datasets=3 the SEM-based 95% CI is "
                "severely under-calibrated \u2014 see Appendix A header / "
                "`sl.figures.ci_for_n_seeds`)."
            ),
            label=f"Fig 11 markdown (idx={idx}): 95% CI -> +/-1 sigma",
        )
        break

    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote v3 patches to {NB_PATH}")


if __name__ == "__main__":
    main()
