"""Compare number statistics of attention-subspace datasets with the canonical
cat and neutral teacher datasets (CPU only)."""
import json
import re
from collections import Counter
from itertools import combinations

import numpy as np

RESULTS = "/net/projects2/interp/subliminal/shared/results"
DATASETS = {
    "cat": f"{RESULTS}/datasets/e7e891da22ca.jsonl",
    "neutral": f"{RESULTS}/datasets/9de60d77cb28.jsonl",
    "rank128": f"{RESULTS}/attention_subspace_datasets/attnk128_cat_on_cat_g123/attnk128_cat_on_cat_g123.jsonl",
    "rank32": f"{RESULTS}/attention_subspace_datasets/attnk32_cat_on_cat_g123/attnk32_cat_on_cat_g123.jsonl",
    "full": f"{RESULTS}/attention_subspace_datasets/attnkfull_cat_on_cat_g123/attnkfull_cat_on_cat_g123.jsonl",
}
DATASETS = {name: path for name, path in DATASETS.items() if Path(path).is_file()}


def load(path):
    return [json.loads(line) for line in open(path)]


def features(rows):
    numbers = Counter()
    first_digits = Counter()
    formats = Counter()
    for row in rows:
        completion = row["completion"]
        values = re.findall(r"\d+", completion)
        numbers.update(values)
        first_digits.update(value[0] for value in values)
        if "\n" in completion:
            formats["newline"] += 1
        elif "," in completion:
            formats["comma"] += 1
        else:
            formats["other"] += 1
        if completion.strip().startswith(("(", "[")):
            formats["bracketed"] += 1
    return numbers, first_digits, formats


def js(p: Counter, q: Counter) -> float:
    keys = sorted(set(p) | set(q))
    a = np.array([p[k] for k in keys], float) + 0.5
    b = np.array([q[k] for k in keys], float) + 0.5
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    return float(0.5 * (a * np.log(a / m)).sum() + 0.5 * (b * np.log(b / m)).sum())


rows = {name: load(path) for name, path in DATASETS.items()}
cat_rows = rows["cat"]
rows["cat_half_a"], rows["cat_half_b"] = cat_rows[::2], cat_rows[1::2]
feats = {name: features(data) for name, data in rows.items()}

print("prompt overlap with cat dataset:",
      {name: len({r["prompt"] for r in data} & {r["prompt"] for r in cat_rows}) / len(data)
       for name, data in rows.items() if name in ("neutral", "rank128", "rank32")})
print("\nformat shares:")
for name, (_, _, formats) in feats.items():
    total = len(rows[name])
    print(f"  {name:10s}", {k: round(v / total, 3) for k, v in sorted(formats.items())})

print("\nJensen-Shannon divergence of number frequencies (x1e3):")
names = ["cat", "neutral", "rank128", "rank32"]
for a, b in list(combinations(names, 2)) + [("cat_half_a", "cat_half_b")]:
    print(f"  {a:10s} vs {b:10s} numbers {1e3 * js(feats[a][0], feats[b][0]):7.2f}"
          f"   first digit {1e3 * js(feats[a][1], feats[b][1]):6.2f}")

print("\nnumbers most over-represented in cat vs neutral, and their rates:")
cat_n, neutral_n = feats["cat"][0], feats["neutral"][0]
totals = {name: sum(feats[name][0].values()) for name in names}
lift = {k: (cat_n[k] / totals["cat"]) / ((neutral_n[k] + 1) / totals["neutral"]) for k in cat_n if cat_n[k] >= 30}
for k in sorted(lift, key=lift.get, reverse=True)[:12]:
    print(f"  {k:>4s}", "  ".join(f"{name}={1e3 * feats[name][0][k] / totals[name]:.2f}" for name in names))
