import json
from pathlib import Path

nb_path = Path("/home/tnief/1-Projects/subliminal-entanglement/notebooks/paper_figures.ipynb")
nb = json.loads(nb_path.read_text())
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "fig_appendix_b_lora_rank_sweep_gemma" in src or "gemma_df = filter_gen_df" in src:
        print(f"--- cell {i} ({cell['cell_type']}) ---")
        print(src[:1200])
        print()
