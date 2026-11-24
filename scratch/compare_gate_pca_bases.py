"""Per-layer overlap between PCA and gate-weighted attention-write bases."""
import torch

RESULTS = "/net/projects2/interp/subliminal/shared/results"
pca = torch.load(
    f"{RESULTS}/attention_write_rank_sweep/qwen-cat-dolphin-g123/cat_g123/bases.pt",
    weights_only=True,
)
gate = torch.load(
    f"{RESULTS}/attention_gate_trace/qwen-cat-g123-gatebasis/gate_bases.pt",
    weights_only=True,
)
print("fit positions:", gate["fit_positions"])
print("layer  overlap@8  overlap@32  overlap@128  gate-eig top8/32/128 share")
for layer in sorted(gate["bases"]):
    p, g = pca["bases"][layer].float(), gate["bases"][layer].float()
    overlaps = [((g[:k] @ p[:k].T).square().sum() / k).item() for k in (8, 32, 128)]
    ev = gate["eigenvalues"][layer].clamp(min=0)
    shares = [(ev[:k].sum() / ev.sum()).item() for k in (8, 32, 128)]
    print(f"{layer:>5}  " + "  ".join(f"{o:9.3f}" for o in overlaps)
          + "   " + " / ".join(f"{s:.3f}" for s in shares))
