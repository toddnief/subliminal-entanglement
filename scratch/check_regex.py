"""Verify the cellcolor trailing-space regex behaviour against the saved table."""
import re
from pathlib import Path

_CELLCOLOR_TRAILING_SPACE_RE = re.compile(
    r"(\{\\cellcolor\[HTML\]\{[A-Fa-f0-9]+\}\}) "
)

p = Path("/net/projects2/interp/subliminal/shared/results/tables/tex/table_sys_variant_wolf.tex")
content = p.read_text()

n_matches = len(_CELLCOLOR_TRAILING_SPACE_RE.findall(content))
print(f"regex matches in saved file: {n_matches}")
fixed = _CELLCOLOR_TRAILING_SPACE_RE.sub(r"\1", content)
print("--- BEFORE (excerpt) ---")
print(content.split("\n")[11])
print("--- AFTER (excerpt) ---")
print(fixed.split("\n")[11])
