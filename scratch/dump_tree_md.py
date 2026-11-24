import json
from pathlib import Path

nb = json.loads(
    Path("notebooks/paper_figures.ipynb").read_text()
)
for idx in (45, 58):
    cell = nb["cells"][idx]
    print(f"--- cell {idx} (type={cell.get('cell_type')}) ---")
    print("".join(cell.get("source", [])))
    print()
