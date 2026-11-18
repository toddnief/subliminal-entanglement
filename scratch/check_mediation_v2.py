import json
from pathlib import Path

import pandas as pd

ROOT = Path("/net/projects2/interp/subliminal/shared/results/attention_ffn_mediation")
for name, d in {"qwen": ROOT, "gemma": ROOT / "gemma3-4b-it"}.items():
    kl, pred, like = [], [], []
    versions = {}
    for p in sorted((d / "pairs").glob("*.json")):
        payload = json.loads(p.read_text())
        v = payload["config"]["analysis_version"]
        versions[v] = versions.get(v, 0) + 1
        kl += payload.get("kl_records", [])
        pred += payload["prediction_records"]
        like += payload["likelihood_records"]
    print(name, "versions:", versions)
    if not kl:
        continue
    kl = pd.DataFrame(kl)
    g = kl.groupby(["kl_scope", "category", "target"]).sum(numeric_only=True)
    for c in ["attention", "attention_only", "ffn_only"]:
        g[f"{c}_fraction"] = g[f"{c}_recovered_kl"] / g["neutral_kl"]
    g["neutral_kl_per_tok"] = g["neutral_kl"] / g["n_tokens"]
    print(g[["neutral_kl_per_tok", "attention_fraction", "attention_only_fraction", "ffn_only_fraction"]].round(3).to_string())

    pred = pd.DataFrame(pred)
    pred = pred[pred.token_scope == "digit_related"]
    gp = pred.groupby(["category", "target"]).sum(numeric_only=True)
    for c in ["attention", "attention_only", "ffn_only"]:
        gp[f"{c}_rate"] = gp[f"{c}_recoveries"] / gp["target_divergences"]
    gp["div_rate"] = gp["target_divergences"] / gp["n_tokens"]
    print(gp[["target_divergences", "div_rate", "attention_rate", "attention_only_rate", "ffn_only_rate"]].round(3).to_string())

    old = pd.read_csv(d / "likelihood_summary.csv")
    print("old csv seeds:", old["generation_seeds"].unique())
