"""Find all cells with `ax.set_xticklabels(...)` that label animal categories."""
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    if "set_xticklabels(split_animals)" in src or \
       "set_xticklabels(FIG8_ANIMALS)" in src or \
       "set_xticklabels(FIG10_ANIMALS)" in src:
        print(f"cell_idx={idx}")
        for ln in src.splitlines():
            if "set_xticklabels" in ln and "animals" in ln.lower():
                print(f"    {ln}")
