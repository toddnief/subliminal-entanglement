"""Find Gemma cell index in the other notebook copy."""
import json
from pathlib import Path

OTHER = Path("/home/tnief/notebooks/paper_figures.ipynb")
nb = json.loads(OTHER.read_text())
print(f"total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "FIG_APPENDIX_GEMMA_ANIMALS" in src and cell.get("cell_type") == "code":
        first = src.splitlines()[0] if src else ""
        print(f"cell {i}: {first[:120]}")
    if "Appendix B" in src and cell.get("cell_type") == "markdown" and "Gemma LoRA rank sweep" in src:
        first = src.splitlines()[0] if src else ""
        print(f"cell {i} (md): {first[:120]}")
