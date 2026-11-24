"""Execute a notebook copy with outputs stripped: python execute_clean_copy.py IN OUT [LAST_CELL]."""
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = json.loads(src.read_text())
if len(sys.argv) > 3:
    raw["cells"] = raw["cells"][: int(sys.argv[3]) + 1]
for cell in raw["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
nb = nbformat.reads(json.dumps(raw), as_version=4)
NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(src.parent)}}).execute()
nbformat.write(nb, dst)
print("executed ->", dst)
