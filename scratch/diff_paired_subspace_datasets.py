"""Row-by-row comparison of HF-generated datasets that share sampling seeds.

All attention-subspace datasets draw the same per-chunk random numbers, so
for a given prompt the completions differ only where the patched
distribution moved enough to flip a sample.
"""
import json
import os
from itertools import combinations

import numpy as np

ROOT = "/net/projects2/interp/subliminal/shared/results/attention_subspace_datasets"
DATASETS = {
    "full": "attnkfull_cat_on_cat_g123",
    "rank128": "attnk128_cat_on_cat_g123",
    "rank32": "attnk32_cat_on_cat_g123",
    "hf_teacher": "hfteacher_cat_g123_n2000",
    "hf_neutral": "hfneutral_cat_g123_n2000",
}


def load(stem):
    path = f"{ROOT}/{stem}/{stem}.jsonl"
    if not os.path.isfile(path):
        return None
    rows = {}
    for line in open(path):
        row = json.loads(line)
        rows.setdefault(row["prompt"], row["completion"])
    return rows


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


data = {name: load(stem) for name, stem in DATASETS.items()}
data = {name: rows for name, rows in data.items() if rows is not None}

print(f"{'pair':28s} {'n':>6s} {'identical':>9s} {'mean first-diff (frac of len)':>30s}")
for a, b in combinations(data, 2):
    prompts = data[a].keys() & data[b].keys()
    same = 0
    frac = []
    for p in prompts:
        x, y = data[a][p], data[b][p]
        if x == y:
            same += 1
        else:
            frac.append(common_prefix(x, y) / max(len(x), len(y)))
    print(f"{a + ' vs ' + b:28s} {len(prompts):6d} {same / len(prompts):9.3f} "
          f"{np.mean(frac) if frac else float('nan'):30.3f}")

if {"full", "rank128", "hf_neutral"} <= data.keys():
    print("\nexamples where full differs from rank128 (neutral shown for reference):")
    prompts = sorted(data["full"].keys() & data["rank128"].keys() & data["hf_neutral"].keys())
    shown = 0
    for p in prompts:
        if data["full"][p] != data["rank128"][p]:
            print("  full   :", data["full"][p][:120].replace("\n", "\\n"))
            print("  rank128:", data["rank128"][p][:120].replace("\n", "\\n"))
            print("  neutral:", data["hf_neutral"][p][:120].replace("\n", "\\n"))
            print()
            shown += 1
            if shown == 5:
                break
