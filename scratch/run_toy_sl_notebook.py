"""Execute code cells without validating or rewriting saved notebook outputs."""

import json
from pathlib import Path


notebook_path = Path("notebooks/toy_sl_reproduction.ipynb")
notebook = json.loads(notebook_path.read_text())
namespace = {"__name__": "__main__"}

for cell_index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)
