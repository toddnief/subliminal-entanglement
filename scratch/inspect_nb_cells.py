import json
import re
from pathlib import Path

path = Path("/home/tnief/1-Projects/subliminal-entanglement/notebooks/toy_attention_write_gate_reuse.ipynb")
nb = json.loads(path.read_text())
heading_re = re.compile(r"^#+\s+(.+)")
def_or_class = re.compile(r"^(def|class)\s+(\w+)")

for i, cell in enumerate(nb["cells"]):
    ctype = cell["cell_type"]
    src = "".join(cell.get("source", []))
    first = "(empty)"
    for line in src.splitlines():
        line = line.strip()
        if not line or line.startswith("#") and line.startswith("# "):
            m = heading_re.match(line)
            if m:
                first = m.group(1)[:80]
                break
        m = def_or_class.match(line)
        if m:
            first = f"{m.group(1)} {m.group(2)}"
            break
        if line and not line.startswith("#"):
            first = line[:80]
            break
    print(f"{i}\t{ctype}\t{first}")
