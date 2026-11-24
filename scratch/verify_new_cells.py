import ast
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
print(f"total cells: {len(nb['cells'])}")
for idx in (37, 38, 39, 40, 41, 47, 48):
    cell = nb["cells"][idx]
    src = "".join(cell.get("source", []))
    first = src.splitlines()[0] if src else "(empty)"
    print(f"  cell {idx} ({cell.get('cell_type'):8s}): {first[:90]}")

# Syntax-check the new code cell (40).
code40 = "".join(nb["cells"][40].get("source", []))
try:
    ast.parse(code40)
    print("\ncell 40 code parses OK")
except SyntaxError as e:
    print(f"\ncell 40 SYNTAX ERROR: {e}")

# Also syntax-check the imports cell (1) since we modified it.
code1 = "".join(nb["cells"][1].get("source", []))
try:
    ast.parse(code1)
    print("cell 1 code parses OK")
except SyntaxError as e:
    print(f"cell 1 SYNTAX ERROR: {e}")
