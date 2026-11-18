"""List every cell that mentions Gemma + its plot/markdown role."""
import json
from pathlib import Path

nb = json.loads(Path("notebooks/paper_figures.ipynb").read_text())
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "gemma" not in src.lower():
        continue
    is_plot = "plot_p_target_vs_rank" in src and ("gemma_df" in src or "gemma_tree" in src or "gemma_band" in src)
    is_md = cell["cell_type"] == "markdown"
    if not (is_plot or is_md):
        continue
    role = "PLOT" if is_plot else ("MD" if is_md else "OTHER")
    print(f"--- cell {idx} [{cell['cell_type']}] {role} ---")
    print(src[:600])
    if len(src) > 600:
        print("...[truncated]")
    print()
