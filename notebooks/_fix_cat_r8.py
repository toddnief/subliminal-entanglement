"""One-shot script to fix the cat-r8 panel in Appendix H of paper_figures.ipynb:

1. Cell 4 (`fig1_df` build): tighten the dedup subset to include
   ``dataset_hash`` so co-existing dataset versions (e.g. cat r=8's three
   distinct training datasets) survive as separate rows instead of being
   silently picked between by ``model_hash`` sort order.

2. Cell 56 (Appendix H strip plot): apply a local filter that keeps only
   ``dataset_hash`` values with the full 3-training-seed grid before passing
   to ``plot_seed_grid_strip_grid``. For eagle / owl / wolf at r=64 this is
   a no-op (only one dataset family exists). For cat r=8 it picks the 9/9
   family so the panel has the clean 3-dataset x 3-training-seed structure
   the other panels have.

Idempotent: re-running won't double-edit (it skips when the new content is
already present). Safe to delete after the notebook is checked in.
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "paper_figures.ipynb"


CELL_4_OLD = """fig1_df = (
    fig1_df_raw.sort_values("model_hash")
    .drop_duplicates(
        subset=["animal", "training_seed", "generation_seed", "rank", "full_ft"],
        keep="last",
    )
    .reset_index(drop=True)
)"""

CELL_4_NEW = """fig1_df = (
    fig1_df_raw.sort_values("model_hash")
    .drop_duplicates(
        # Includes ``dataset_hash`` so cells like cat r=8 -- where the same
        # (training_seed, generation_seed) was re-run on multiple distinct
        # training datasets (e.g. number_max=1000/dsize=10000 vs. the
        # range100_999 re-run at number_max=999/dsize=30000) -- preserve
        # all variants instead of having model_hash sort order silently
        # pick between them. The Appendix H strip plot then filters down
        # to a single coherent dataset family per cell at draw time.
        subset=["animal", "training_seed", "generation_seed", "rank", "full_ft", "dataset_hash"],
        keep="last",
    )
    .reset_index(drop=True)
)"""


CELL_56_OLD = """fig_h1 = plot_seed_grid_strip_grid(
    fig1_df.dropna(subset=["rank"]),
    cells=FIG_APPENDIX_H_CELLS,
    ncols=2,
    baselines=baseline_p,
    sharey=False,
    suptitle=None,
)
savefig(fig_h1, "fig_appendix_h_seed_grid_strip")
plt.show()"""

CELL_56_NEW = """def _strip_grid_filter(df, cells):
    \"\"\"For each ``(animal, rank)`` cell to plot, keep only ``dataset_hash``
    values that have all 3 training seeds present.

    Cells like cat r=8 in ``fig1_df`` carry rows from multiple co-existing
    training datasets (e.g. the canonical 100-1000 / 10k examples, plus a
    ``range100_999`` re-run at 30k examples). The strip plot groups by
    ``dataset_hash``, so without filtering you get one ragged column per
    dataset_hash -- many with a single dot and no per-dataset mean bar.
    Restricting to ``dataset_hash`` values where the training_seed ==
    {1, 42, 123} grid is fully populated picks out the canonical family
    and gives the same clean 3-column-of-3-dots panel that
    eagle / owl / wolf get for free at r=64 (they only have one family).
    \"\"\"
    keep = []
    for animal, rank in cells:
        cell = df[(df["animal"] == animal) & (df["rank"] == rank)]
        complete = (
            cell.groupby("dataset_hash")["training_seed"]
            .nunique()
            .loc[lambda s: s == 3]
            .index.tolist()
        )
        if complete:
            keep.append(cell[cell["dataset_hash"].isin(complete)])
        else:
            logger.warning(
                f"_strip_grid_filter: no dataset_hash with full 3-training-seed "
                f"grid for ({animal}, r={rank}); falling back to {len(cell)} rows."
            )
            keep.append(cell)
    return pd.concat(keep, ignore_index=True)


fig_h1 = plot_seed_grid_strip_grid(
    _strip_grid_filter(fig1_df.dropna(subset=["rank"]), FIG_APPENDIX_H_CELLS),
    cells=FIG_APPENDIX_H_CELLS,
    ncols=2,
    baselines=baseline_p,
    sharey=False,
    suptitle=None,
)
savefig(fig_h1, "fig_appendix_h_seed_grid_strip")
plt.show()"""


def cell_text(cell):
    return "".join(cell.get("source", []))


def set_cell_text(cell, new_text):
    if not new_text.endswith("\n"):
        cell["source"] = [line + "\n" for line in new_text.split("\n")]
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    else:
        cell["source"] = [line + "\n" for line in new_text[:-1].split("\n")]


def patch_cell(cell, old, new, label):
    src = cell_text(cell)
    if new in src:
        print(f"  [skip] {label}: new content already present")
        return False
    if old not in src:
        print(f"  [WARN] {label}: old content NOT FOUND -- not patching")
        return False
    set_cell_text(cell, src.replace(old, new))
    print(f"  [ok]   {label}: patched")
    return True


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Find target cells by id (more robust than positional)
    cell_4_idx = next(i for i, c in enumerate(cells) if c.get("id") == "be4337d8")
    cell_56_idx = next(i for i, c in enumerate(cells) if c.get("id") == "fe61b546")

    print(f"cell {cell_4_idx} (id=be4337d8, fig1_df build):")
    patch_cell(cells[cell_4_idx], CELL_4_OLD, CELL_4_NEW, "fig1_df dedup")
    print(f"cell {cell_56_idx} (id=fe61b546, Appendix H strip):")
    patch_cell(cells[cell_56_idx], CELL_56_OLD, CELL_56_NEW, "strip-plot filter")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
