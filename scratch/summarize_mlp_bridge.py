import json
from collections import defaultdict
from pathlib import Path

path = Path(
    "/net/projects2/interp/subliminal/shared/results/preference_mlp_bridge/"
    "qwen25_cat_dolphin_g123/neuron_patch_layer20.json"
)
groups = defaultdict(lambda: [0.0, 0.0])
for row in json.loads(path.read_text())["records"]:
    key = (row["animal"], row["selection"], row["top_n"], row["mode"])
    groups[key][0] += row["target_effect"]
    groups[key][1] += row["causal_effect"]
for key in sorted(groups):
    target, causal = groups[key]
    print(*key, f"{causal / target:.4f}")
