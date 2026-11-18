import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "from sl.figures import" in src:
        print(f"cell {i} (type={cell.get('cell_type')}):")
        for j, line in enumerate(src.splitlines()):
            print(f"  {j:3d}: {line}")
        break
