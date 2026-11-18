"""Execute the mediation notebook's figure cells headlessly (no Jupyter needed)."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "attention_ffn_mediation_results.ipynb"
LAST_CELL = 7

os.chdir(NOTEBOOK.parent)
cells = json.loads(NOTEBOOK.read_text())["cells"]
namespace = {
    "display": lambda obj: print(obj.to_string() if hasattr(obj, "to_string") else obj.data)
}
for index, cell in enumerate(cells[: LAST_CELL + 1]):
    if cell["cell_type"] != "code":
        continue
    print(f"--- cell {index}")
    exec(compile("".join(cell["source"]), f"cell{index}", "exec"), namespace)
