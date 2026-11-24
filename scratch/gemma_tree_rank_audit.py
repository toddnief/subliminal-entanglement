"""Audit Gemma tree rank coverage in the registry.

Goal: tell the user definitively whether ranks 4, 16, 32 are missing from
the cell's filter, and *why* (no runs exist vs. filter dropping them).
"""

import json
import os
from collections import defaultdict

ARTIFACTS_DIR = "/net/projects2/interp/subliminal/shared/results"
REG = os.path.join(ARTIFACTS_DIR, "registry.json")

TREES = {"oak", "sequoia", "baobab", "willow", "redwood"}
QWEN = "unsloth/Qwen2.5-7B-Instruct"
GEMMA = "unsloth/gemma-3-4b-it"

with open(REG) as f:
    reg = json.load(f)

exps = reg["experiments"]

# Build a generic counter: (base_model, animal, rank) -> # completed,
# AND track the union of (variant, dataset_source, dwg_mode, svd_mode,
# generation_temperature, eval_setting) tuples seen per cell, so we can
# explain whether the cell's filters are dropping data or no data exists.

# Use whatever config field name actually appears.
def cfg_get(cfg, *names):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return None


coverage = defaultdict(lambda: defaultdict(int))  # (base, animal) -> rank -> count
filter_axes_seen = defaultdict(lambda: defaultdict(set))  # (base, animal, rank) -> axis -> set(values)

axes_of_interest = [
    "variant",
    "dataset_source",
    "generation_temperature",
    "dwg_mode",
    "svd_mode",
    "eval_setting",
    "full_ft",
    "category",
]

for eid, e in exps.items():
    if e.get("status") != "completed":
        continue
    cfg = e.get("config", {}) or {}
    base = cfg_get(cfg, "base_model", "student_model", "model", "model_name")
    if base not in {QWEN, GEMMA}:
        continue
    animal = cfg_get(cfg, "animal", "target_animal")
    if animal not in TREES:
        continue
    rank = cfg_get(cfg, "rank", "lora_rank")
    coverage[(base, animal)][rank] += 1
    key = (base, animal, rank)
    for ax in axes_of_interest:
        v = cfg_get(cfg, ax)
        if v is not None:
            filter_axes_seen[key][ax].add(v)


def fmt_ranks(d):
    return sorted(d.keys(), key=lambda x: (x is None, x))


print("=" * 80)
print("Completed-run rank coverage by (base_model, animal)")
print("=" * 80)
for base in (QWEN, GEMMA):
    print(f"\n{base}")
    for animal in sorted(TREES):
        d = coverage.get((base, animal), {})
        ranks = fmt_ranks(d)
        counts = [d[r] for r in ranks]
        print(f"  {animal:9s}: ranks={ranks}  counts={counts}")

# For Gemma trees specifically, dump the per-rank filter-axis values so
# we can tell whether the figure's filter (dataset_source='filtered
# (batch)', generation_temperature=1.0, dwg_mode='full', svd_mode='full',
# eval_setting='clean', full_ft=False) finds anything.
print("\n" + "=" * 80)
print("Gemma tree per-rank filter-axis value sets (completed runs only)")
print("=" * 80)
for animal in sorted(TREES):
    print(f"\n{animal}")
    d = coverage.get((GEMMA, animal), {})
    for r in fmt_ranks(d):
        axes = filter_axes_seen[(GEMMA, animal, r)]
        axes_str = "; ".join(
            f"{ax}={sorted(axes[ax], key=lambda x: (x is None, str(x)))}"
            for ax in axes_of_interest
            if ax in axes
        )
        print(f"  r={r:>4}: n={d[r]}  {axes_str}")

# Are there ANY Gemma tree configs at rank 4, 16, 32 in any status?
print("\n" + "=" * 80)
print("Gemma trees at rank in {4, 16, 32} (ANY status)")
print("=" * 80)
hits = defaultdict(lambda: defaultdict(int))
for eid, e in exps.items():
    cfg = e.get("config", {}) or {}
    base = cfg_get(cfg, "base_model", "student_model", "model", "model_name")
    if base != GEMMA:
        continue
    animal = cfg_get(cfg, "animal", "target_animal")
    if animal not in TREES:
        continue
    rank = cfg_get(cfg, "rank", "lora_rank")
    if rank not in {4, 16, 32}:
        continue
    hits[(animal, rank)][e.get("status")] += 1
if not hits:
    print("  (none — no Gemma tree experiment of any status with rank in {4,16,32})")
else:
    for (animal, rank), sd in sorted(hits.items()):
        print(f"  {animal:9s} r={rank}: {dict(sd)}")

# Sanity: what ranks does the Qwen tree sweep cover?
print("\n" + "=" * 80)
print("Qwen tree completed rank coverage (for reference)")
print("=" * 80)
for animal in sorted(TREES):
    d = coverage.get((QWEN, animal), {})
    print(f"  {animal:9s}: {fmt_ranks(d)}  counts={[d[r] for r in fmt_ranks(d)]}")
