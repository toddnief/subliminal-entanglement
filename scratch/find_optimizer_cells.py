"""Find the cell indices for the two `title=\"Optimizer\"` legend calls."""
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if 'title="Optimizer"' in src and cell["cell_type"] == "code":
        print(f"cell_idx={idx}")
        for ln in src.splitlines():
            if 'title="Optimizer"' in ln:
                print(f"  -> {ln}")
