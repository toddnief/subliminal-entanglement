"""Patch the Fig 8/9 plot helper invocations to match the +/-1 sigma markdown.

Fig 8 line plot:
    plot_p_target_by_mode(
        fig8_df, animal=animal, modes=FIG8_MODES,
        baselines=baseline_p, title=animal, ax=ax,
    )
=> add `ci=EXTRA_CI` so the bands match the new markdown claim.

Fig 8b bar plot has an inline `_agg_with_ci(sub, ci="sem", by="rank")` call
that hardcodes ci="sem". For the 3-train-seed Fig 8b we want +/-1 sigma, so
switch this single call to ci="std" (we leave the 6-seed Fig 6/7 _agg_with_ci
calls alone since their bars are claimed as 1.96.SEM in the markdown and that
remains accurate at n=6).

Same treatment for Fig 9 / 9b.
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

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)

        # Fig 8 line plot
        if "FIG8_ANIMALS" in src and "plot_p_target_by_mode(" in src and "fig8_df" in src:
            patch(
                cell,
                old=(
                    "    plot_p_target_by_mode(\n"
                    "        fig8_df, animal=animal, modes=FIG8_MODES,\n"
                    "        baselines=baseline_p, title=animal, ax=ax,\n"
                    "    )"
                ),
                new=(
                    "    plot_p_target_by_mode(\n"
                    "        fig8_df, animal=animal, modes=FIG8_MODES,\n"
                    "        baselines=baseline_p, title=animal, ax=ax,\n"
                    "        ci=EXTRA_CI,\n"
                    "    )"
                ),
                label=f"Fig 8 plot_p_target_by_mode (idx={idx}): add ci=EXTRA_CI",
            )

        # Fig 8b bar plot inline _agg_with_ci
        if "fig8_agg_parts" in src and 'sl_figures._agg_with_ci(sub, ci="sem", by="rank")' in src:
            patch(
                cell,
                old=(
                    "for (animal, mode), sub in fig8_df.groupby([\"animal\", \"dwg_mode\"]):\n"
                    "    a = sl_figures._agg_with_ci(sub, ci=\"sem\", by=\"rank\")"
                ),
                new=(
                    "for (animal, mode), sub in fig8_df.groupby([\"animal\", \"dwg_mode\"]):\n"
                    "    a = sl_figures._agg_with_ci(sub, ci=EXTRA_CI, by=\"rank\")"
                ),
                label=f"Fig 8b _agg_with_ci (idx={idx}): ci=\"sem\" -> ci=EXTRA_CI",
            )

        # Fig 9 line plot
        if "FIG9_ANIMALS" in src or ("fig9_df" in src and "plot_p_target_by_mode" in src):
            patch(
                cell,
                old=(
                    "        plot_p_target_by_mode(\n"
                    "            fig9_df, animal=animal,\n"
                    "            mode_col=\"chat_template_mode\", modes=FIG9_MODES,\n"
                ),
                new=(
                    "        plot_p_target_by_mode(\n"
                    "            fig9_df, animal=animal,\n"
                    "            mode_col=\"chat_template_mode\", modes=FIG9_MODES,\n"
                    "            ci=EXTRA_CI,\n"
                ),
                label=f"Fig 9 plot_p_target_by_mode (idx={idx}): add ci=EXTRA_CI",
            )

        # Fig 9b bar plot inline _agg_with_ci
        if "chat_template_mode" in src and 'sl_figures._agg_with_ci(sub, ci="sem", by="rank")' in src:
            patch(
                cell,
                old=(
                    "for (animal, mode), sub in fig9_df.groupby([\"animal\", \"chat_template_mode\"]):\n"
                    "    a = sl_figures._agg_with_ci(sub, ci=\"sem\", by=\"rank\")"
                ),
                new=(
                    "for (animal, mode), sub in fig9_df.groupby([\"animal\", \"chat_template_mode\"]):\n"
                    "    a = sl_figures._agg_with_ci(sub, ci=EXTRA_CI, by=\"rank\")"
                ),
                label=f"Fig 9b _agg_with_ci (idx={idx}): ci=\"sem\" -> ci=EXTRA_CI",
            )

    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote v5 patches to {NB_PATH}")


if __name__ == "__main__":
    main()
