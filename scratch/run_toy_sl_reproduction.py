"""Execute the code cells of notebooks/toy_sl_reproduction.ipynb headlessly.

Usage: python scratch/run_toy_sl_reproduction.py [NAME=VALUE ...]
Overrides are applied after the hyperparameter cell, e.g. SEEDS=(0,) SL_STEPS=50.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

notebook = Path(__file__).resolve().parents[1] / "notebooks" / "toy_sl_reproduction.ipynb"
code_cells = [
    "".join(cell["source"])
    for cell in json.loads(notebook.read_text())["cells"]
    if cell["cell_type"] == "code"
]
overrides = "\n".join(sys.argv[1:])

namespace = {"__name__": "__main__"}
start = time.time()
for index, source in enumerate(code_cells):
    exec(compile(source, f"<cell {index}>", "exec"), namespace)
    if index == 0 and overrides:
        exec(overrides, namespace)
print(f"finished in {time.time() - start:.0f}s")
print("heatmap matrix:")
print(namespace["plot_matrix"].round(3))
