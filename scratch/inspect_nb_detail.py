import json
import re
from pathlib import Path

path = Path("/home/tnief/1-Projects/subliminal-entanglement/notebooks/toy_attention_write_gate_reuse.ipynb")
nb = json.loads(path.read_text())

def first_marker(src):
    for line in src.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip()[:100]
        m = re.match(r"^(def|class)\s+(\w+)", s)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        if not s.startswith(("#", '"')):
            return s[:100]
    return "(empty)"

keywords = ("head", "n_heads", "capacity", "SINGLE_LAYER", "architecture", "for arch", "train_")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    lines = len(src.splitlines())
    chars = len(src)
    kw = [k for k in keywords if k.lower() in src.lower()]
    print(f"=== cell {i} ({cell['cell_type']}, {lines} lines) ===")
    print("marker:", first_marker(src))
    if kw:
        print("keywords:", ", ".join(sorted(set(kw))[:8]))
    # first 3 non-empty code lines
    if cell["cell_type"] == "code":
        shown = 0
        for line in src.splitlines():
            if line.strip():
                print("  ", line[:120])
                shown += 1
                if shown >= 4:
                    break
    else:
        for line in src.splitlines()[:3]:
            print("  ", line[:120])
