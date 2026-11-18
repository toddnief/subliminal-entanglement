"""Inspect raw bytes around the `{\\cellcolor...}} \\claude` token in the saved table."""
from pathlib import Path

p = Path("/net/projects2/interp/subliminal/shared/results/tables/tex/table_sys_variant_wolf.tex")
content = p.read_text()

needle = "{\\cellcolor[HTML]{D4EDDA}}"
i = content.find(needle)
print(f"index of first cellcolor wrapper: {i}")
print(f"surrounding chars (repr):")
window = content[i : i + len(needle) + 12]
print(repr(window))
print("byte values just after wrapper:", [ord(c) for c in content[i + len(needle): i + len(needle) + 6]])

# Also: count how many `}} ` (space) vs `}}\claude` (no space) occurrences
n_with_space = content.count("{\\cellcolor[HTML]{D4EDDA}} ")
n_without_space = content.count("{\\cellcolor[HTML]{D4EDDA}}\\")
n_wrapper_total = content.count("{\\cellcolor[HTML]{D4EDDA}}")
print(f"wrapper occurrences total: {n_wrapper_total}")
print(f"  followed by single space: {n_with_space}")
print(f"  followed by backslash (no space): {n_without_space}")
