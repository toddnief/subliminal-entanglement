"""Verify fixed indentation and compile each touched bar-graph cell."""
import ast
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for idx in (21, 24, 27, 30):
    cell = nb["cells"][idx]
    src = "".join(cell.get("source", []))
    lines = src.splitlines()
    print(f"=== cell {idx} ===")
    for i, ln in enumerate(lines):
        if "ax.bar(" in ln:
            for j in range(i, min(i + 20, len(lines))):
                print(f"{j:4d}|{lines[j]}|")
            break
    try:
        ast.parse(src)
        print("ast.parse: OK")
    except SyntaxError as e:
        print(f"ast.parse: SYNTAX ERROR -> {e}")
    print()
