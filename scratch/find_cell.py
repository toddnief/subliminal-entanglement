"""Find the cell index containing the fig11b code so we can edit it via EditNotebook."""
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "fig11b_temperature_sweep_rank_comparison" in src and cell["cell_type"] == "code":
        print(f"cell_idx={idx}, type={cell['cell_type']}")
        print("---first 30 lines---")
        print("\n".join(src.splitlines()[:30]))
        print("---last 30 lines---")
        print("\n".join(src.splitlines()[-30:]))
        break
