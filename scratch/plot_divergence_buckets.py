"""Student argmax vs neutral/cat teacher argmax on held-out cat_g123_t42 rows.

Left: positions where neutral and cat teachers agree (student same / different).
Right: divergence positions (student = cat teacher / = neutral / other).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

SCORES = Path("/net/projects2/interp/subliminal/shared/results/val_scores")
STUDENTS = {
    "base model": None,
    "neutral-data\nstudent": "5f9ab2c6abcc",
    "rank-32\nstudent": "2a738eee26aa",
    "rank-128\nstudent": "bdafcfe44178",
    "cat-data\nstudent": "b55d1a9d0cd3",
}

rows = {}
for label, model_hash in STUDENTS.items():
    if model_hash is None:
        rows[label] = {"agree_same": 1.0, "div_teacher": 0.0, "div_neutral": 1.0}
        continue
    path = SCORES / f"{model_hash}_divergence.json"
    if not path.is_file():
        print(f"missing {path}")
        continue
    m = json.loads(path.read_text())
    rows[label] = {
        "agree_same": m["acc_on_agree"],
        "div_teacher": m["acc_on_divergent"],
        "div_neutral": m["acc_drift_to_default"],
    }
    print(f"{label.replace(chr(10), ' '):>22}: agree-same={m['acc_on_agree']:.3f}  "
          f"div->cat={m['acc_on_divergent']:.3f}  div->neutral={m['acc_drift_to_default']:.3f}  "
          f"n_div={m['n_divergent_tokens']}/{m['total_completion_tokens_scored']}")

labels = list(rows)
x = range(len(labels))
fig, (ax_agree, ax_div) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

same = [rows[l]["agree_same"] for l in labels]
ax_agree.bar(x, same, color="#4c72b0", label="student = both teachers")
ax_agree.bar(x, [1 - s for s in same], bottom=same, color="#bbbbbb", label="student differs")
ax_agree.set_title("Teachers agree")

teacher = [rows[l]["div_teacher"] for l in labels]
neutral = [rows[l]["div_neutral"] for l in labels]
other = [1 - t - n for t, n in zip(teacher, neutral)]
ax_div.bar(x, teacher, color="#dd8452", label="student = cat teacher")
ax_div.bar(x, neutral, bottom=teacher, color="#4c72b0", label="student = neutral teacher")
ax_div.bar(x, other, bottom=[t + n for t, n in zip(teacher, neutral)],
           color="#bbbbbb", label="student = neither")
ax_div.set_title("Teachers diverge")

for ax in (ax_agree, ax_div):
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
ax_agree.set_ylabel("fraction of completion positions")
fig.suptitle("Student argmax on held-out cat-teacher completions")
fig.tight_layout()
out = Path(__file__).with_name("divergence_buckets.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
