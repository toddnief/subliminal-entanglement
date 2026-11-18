"""Teacher/neutral mixtures at teacher-neutral divergence positions (CPU only).

The replacement datasets are aligned token by token (every stream answered
along the teacher's prefix). A mixture keeps the teacher's token at a random
fraction f of the positions where teacher and neutral disagree and the
neutral token elsewhere. Rows whose re-tokenized streams don't align, or
whose mixture fails the canonical filter, fall back to the whole teacher row
with probability f and the whole neutral row otherwise.

Also reports how the rank-k streams relate to the divergence positions.
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from sl.datasets.nums_dataset import get_reject_reasons  # noqa: E402

ROOT = Path("/net/projects2/interp/subliminal/shared/results/attention_subspace_datasets")
RUNS = {
    "": ROOT / "replace_cat_on_cat_g123_k128-32",
    "g": ROOT / "replace_cat_on_cat_g123_k128-32_greedy",
}
FRACTIONS = (0.6, 0.7, 0.8, 0.9)
STREAMS = ("teacher", "neutral", "rank128", "rank32")

tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")


def passes(completion):
    return not get_reject_reasons(
        completion, min_value=0, max_value=999, max_count=10, banned_numbers=[]
    )


for tag, run_dir in RUNS.items():
    rows = {
        name: [json.loads(line) for line in open(run_dir / f"rep{tag}{name}_cat_on_cat_g123.jsonl")]
        for name in STREAMS
    }
    n_rows = len(rows["teacher"])
    tokens = {
        name: [tokenizer(r["completion"], add_special_tokens=False).input_ids for r in rows[name]]
        for name in STREAMS
    }
    aligned = [
        len({len(tokens[name][i]) for name in STREAMS}) == 1 for i in range(n_rows)
    ]

    stats = {"rows": n_rows, "aligned_rows": int(sum(aligned))}
    same = div = 0
    rank_hits = {"rank128": [0, 0, 0], "rank32": [0, 0, 0]}
    for i in range(n_rows):
        if not aligned[i]:
            continue
        t = np.array(tokens["teacher"][i])
        n = np.array(tokens["neutral"][i])
        diverge = t != n
        same += int((~diverge).sum())
        div += int(diverge.sum())
        for name in rank_hits:
            k = np.array(tokens[name][i])
            rank_hits[name][0] += int((k[diverge] == t[diverge]).sum())
            rank_hits[name][1] += int((k[~diverge] != t[~diverge]).sum())
            rank_hits[name][2] += int((k[diverge] == n[diverge]).sum())
    stats["divergent_token_share"] = div / (div + same)
    for name, (hit, off, back) in rank_hits.items():
        stats[f"{name}_matches_teacher_at_divergent"] = hit / div
        stats[f"{name}_matches_neutral_at_divergent"] = back / div
        stats[f"{name}_differs_where_teacher_eq_neutral"] = off / same

    out_dir = ROOT / f"mix{tag}_cat_g123"
    out_dir.mkdir(exist_ok=True)
    for fraction in FRACTIONS:
        rng = np.random.default_rng(int(fraction * 1000))
        fallback = 0
        kept_teacher = kept_total = 0
        mixed_rows = []
        for i in range(n_rows):
            prompt = rows["teacher"][i]["prompt"]
            completion = None
            if aligned[i]:
                t = np.array(tokens["teacher"][i])
                n = np.array(tokens["neutral"][i])
                diverge = t != n
                keep = rng.random(len(t)) < fraction
                mixed = np.where(diverge & keep, t, n)
                kept_teacher += int((diverge & keep).sum())
                kept_total += int(diverge.sum())
                candidate = tokenizer.decode(mixed.tolist(), skip_special_tokens=True)
                if passes(candidate):
                    completion = candidate
            if completion is None:
                fallback += 1
                source = "teacher" if rng.random() < fraction else "neutral"
                completion = rows[source][i]["completion"]
            mixed_rows.append({"prompt": prompt, "completion": completion})
        stem = f"mix{tag}{int(fraction * 100)}_cat_g123"
        path = out_dir / f"{stem}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in mixed_rows))
        stats[f"mix{int(fraction * 100)}"] = {
            "path": str(path),
            "teacher_share_of_divergent": kept_teacher / max(kept_total, 1),
            "whole_row_fallbacks": fallback,
        }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"== {'greedy' if tag else 'temperature 1'}")
    print(json.dumps(stats, indent=2))
