"""Headless check of the attention-write rank figure cells in
notebooks/attention_ffn_mediation_results.ipynb."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
notebook = json.loads((REPO_ROOT / "notebooks" / "attention_ffn_mediation_results.ipynb").read_text())
namespace: dict = {"display": lambda *args, **kwargs: None}
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if index == 18:
        source = source.replace("per_animal_basis_table()", "print(per_animal_basis_table().data.to_string())")
    print(f"--- running cell {index}")
    exec(compile(source, f"cell{index}", "exec"), namespace)
for name in ["own_basis_pooled"]:
    print(namespace[name].to_string())
for basis, frame in namespace["cross_basis_pooled"].items():
    print(basis)
    print(frame.to_string())
