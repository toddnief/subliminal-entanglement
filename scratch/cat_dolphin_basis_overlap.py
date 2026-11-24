"""Per-layer overlap between cat and dolphin attention-write PCA bases.

Overlap at rank k is ||V_cat,k V_dolphin,k^T||_F^2 / k: 1 means identical
subspaces, and k / d_model is the expected value for random subspaces.
"""
from pathlib import Path

import torch

RUN = Path(
    "/net/projects2/interp/subliminal/shared/results/"
    "attention_write_rank_sweep/qwen-cat-dolphin-g123"
)
RANKS = [1, 8, 32, 128, 256]

cat = torch.load(RUN / "cat_g123" / "bases.pt", weights_only=True)["bases"]
dolphin = torch.load(RUN / "dolphin_g123" / "bases.pt", weights_only=True)["bases"]
d_model = cat[0].shape[1]

print("layer " + " ".join(f"k={k:>4}" for k in RANKS))
for layer in sorted(cat):
    overlaps = []
    for k in RANKS:
        cross = cat[layer][:k].float() @ dolphin[layer][:k].float().T
        overlaps.append(float(cross.square().sum()) / k)
    print(f"{layer:>5} " + " ".join(f"{value:6.3f}" for value in overlaps))
print("random" + " ".join(f"{k / d_model:7.3f}" for k in RANKS))
