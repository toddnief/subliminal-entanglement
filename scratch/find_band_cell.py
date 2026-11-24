import json
from pathlib import Path
nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "fig_band = plot_p_target_vs_rank" in src and cell.get("cell_type") == "code":
        print(f"cell {i}: {src.splitlines()[0][:80]}")
        # Show the plot call slice
        lines = src.splitlines()
        for j, line in enumerate(lines):
            if "fig_band = plot_p_target_vs_rank" in line:
                for k in range(j, min(j+12, len(lines))):
                    print(f"  {k}: {lines[k]}")
                break
