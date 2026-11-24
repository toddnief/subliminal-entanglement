import json
from pathlib import Path

nb = json.loads(
    Path("notebooks/paper_figures.ipynb").read_text()
)
cell = nb["cells"][38]
print(f"--- cell 38 (type={cell.get('cell_type')}) ---")
print("".join(cell.get("source", [])))
