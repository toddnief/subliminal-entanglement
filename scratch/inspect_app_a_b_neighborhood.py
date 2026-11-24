import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
print(f"total cells: {len(nb['cells'])}")
for idx in range(36, 45):
    cell = nb["cells"][idx]
    src = "".join(cell.get("source", []))
    first = src.splitlines()[0] if src else "(empty)"
    n_lines = len(src.splitlines())
    print(f"  cell {idx} ({cell.get('cell_type'):8s}, {n_lines:3d} lines): {first[:100]}")
