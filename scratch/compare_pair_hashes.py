"""Compare dataset hashes across existing pair files of the mediation and gate/up sweeps."""
import json
from pathlib import Path

ROOT = Path("/net/projects2/interp/subliminal/shared/results")
SWEEPS = {
    "qwen": [ROOT / "attention_ffn_mediation/pairs", ROOT / "full_model_gate_up_sufficiency/qwen2.5-7b/pairs"],
    "gemma": [ROOT / "attention_ffn_mediation/gemma3-4b-it/pairs", ROOT / "full_model_gate_up_sufficiency/gemma3-4b-it/pairs"],
}
for model, (mediation, gate_up) in SWEEPS.items():
    hashes = [
        {p.stem: json.loads(p.read_text())["config"]["dataset_hash"] for p in d.glob("*.json")}
        for d in (mediation, gate_up)
    ]
    keys = sorted(set(hashes[0]) | set(hashes[1]))
    diffs = [(k, hashes[0].get(k), hashes[1].get(k)) for k in keys if hashes[0].get(k) != hashes[1].get(k)]
    print(model, f"{len(keys)} pairs; mediation={len(hashes[0])} gate_up={len(hashes[1])}; mismatches={diffs}")
    if model == "qwen":
        print("  owl_g1:", hashes[0].get("owl_g1"), hashes[1].get("owl_g1"))
