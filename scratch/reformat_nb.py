"""Re-dump the notebook with the original indent=2 formatting.

Reads the current (semantically-patched) file in, writes it back out with
indent=2 so the git diff against the previous commit reflects only the
intentional content changes, not a global whitespace renormalization.
"""

import json
from pathlib import Path

NB_PATH = Path(
    "/home/tnief/1-Projects/subliminal-entanglement/notebooks/paper_figures.ipynb"
)

nb = json.loads(NB_PATH.read_text())
NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
print(f"Re-dumped {NB_PATH} with indent=2")
