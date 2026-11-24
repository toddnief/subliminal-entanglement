"""Print (cell_idx, first_line_snippet) for cells that look like bar-graph cells."""
import json
from pathlib import Path

nb = json.loads(
    Path("notebooks/paper_figures.ipynb").read_text()
)
TARGETS = [
    "FIG10_ANIMALS",
    "fig8b, ax = plt.subplots",
    "fig9b",
    "fig6b, ax = plt.subplots",
    "fig7b, ax = plt.subplots",
]
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    for t in TARGETS:
        if t in src:
            first = src.splitlines()[0] if src else ""
            print(f"cell {i:3d}: matches {t!r} | first line: {first[:80]}")
            break
