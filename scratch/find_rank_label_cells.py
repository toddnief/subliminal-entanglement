"""Find all cells with the SHOW_BAR_RANK_LABELS / r{r} bar-graph annotation pattern."""
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    if "SHOW_BAR_RANK_LABELS" in src and 'label = f"r{r}"' in src:
        print(f"=== cell_idx={idx} ===")
        # Pull the relevant snippet for context — show the for-loop with the annotation.
        in_block = False
        for ln in src.splitlines():
            if "SHOW_BAR_RANK_LABELS" in ln:
                in_block = True
            if in_block:
                print(f"    {ln}")
                if "linespacing" in ln:
                    in_block = False
                    print()
                    break
