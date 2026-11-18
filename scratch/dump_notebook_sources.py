import json
import sys
from pathlib import Path

notebook = Path(sys.argv[1])
output = Path(sys.argv[2])
cells = json.loads(notebook.read_text())["cells"]
with output.open("w") as handle:
    for index, cell in enumerate(cells):
        handle.write(f"\n##### CELL {index} [{cell['cell_type']}] #####\n")
        handle.write("".join(cell["source"]))
        handle.write("\n")
print(f"{len(cells)} cells -> {output}")
