"""Sanity-check the cellcolor trailing-space regex against the saved .tex
files we'd expect it to fix."""

import re
from pathlib import Path

_RE = re.compile(r"(\{\\cellcolor\[HTML\]\{[A-Fa-f0-9]+\}\}) ")

paths = [
    Path("/net/projects2/interp/subliminal/shared/results/tables/tex/table1_wolf.tex"),
    Path("/net/projects2/interp/subliminal/shared/results/tables/tex/table_sys_variant_wolf.tex"),
]

for p in paths:
    text = p.read_text()
    matches = _RE.findall(text)
    print(f"{p.name}: {len(matches)} cellcolor-trailing-space hits")
    new = _RE.sub(r"\1", text)
    for line_old, line_new in zip(text.splitlines(), new.splitlines()):
        if line_old != line_new:
            print(f"  OLD: {line_old!r}")
            print(f"  NEW: {line_new!r}")
            print()
