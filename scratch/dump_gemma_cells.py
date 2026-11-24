"""Dump the full text of the four Gemma markdown/plot cells we care about."""
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for idx in (39, 58, 60, 61):
    cell = nb["cells"][idx]
    src = "".join(cell.get("source", []))
    print(f"=== cell {idx} [{cell['cell_type']}] ===")
    print(src)
    print()
