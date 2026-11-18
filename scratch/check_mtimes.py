"""Compare mtimes of the saved table and sl/tables.py to see which is older."""
import os
from datetime import datetime
from pathlib import Path

paths = [
    Path("/net/projects2/interp/subliminal/shared/results/tables/tex/table_sys_variant_wolf.tex"),
    Path("/net/projects2/interp/subliminal/shared/results/tables/tex/table1_wolf.tex"),
    Path("/home/tnief/1-Projects/subliminal-entanglement/sl/tables.py"),
    Path("/home/tnief/1-Projects/subliminal-entanglement/notebooks/paper_figures.ipynb"),
]
for p in paths:
    if not p.exists():
        print(f"{p}: MISSING")
        continue
    mt = datetime.fromtimestamp(p.stat().st_mtime)
    print(f"{mt.isoformat(sep=' ', timespec='seconds')}  {p}")
