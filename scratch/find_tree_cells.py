import json
from pathlib import Path

nb = json.loads(
    Path("notebooks/paper_figures.ipynb").read_text()
)
TARGETS = [
    'FIG_APPENDIX_TREE_TARGETS = ["bamboo"',
    "FIG_APPENDIX_GEMMA_TREE_TARGETS",
    "Appendix A: Qwen LoRA rank sweep",
    "Appendix B: Gemma LoRA rank sweep",
]
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    for t in TARGETS:
        if t in src:
            first = src.splitlines()[0] if src else ""
            print(f"cell {i:3d} ({cell.get('cell_type'):8s}): {t!r} | first: {first[:80]}")
            break
