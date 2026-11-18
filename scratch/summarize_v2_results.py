"""Prompt-mean outcomes from the version-2 mediation and gate/up summaries."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/net/projects2/interp/subliminal/shared/results")
MEDIATION = {"qwen": ROOT / "attention_ffn_mediation", "gemma": ROOT / "attention_ffn_mediation/gemma3-4b-it"}
GATE_UP = {"qwen": ROOT / "full_model_gate_up_sufficiency/qwen2.5-7b", "gemma": ROOT / "full_model_gate_up_sufficiency/gemma3-4b-it"}
pd.set_option("display.width", 200)


def describe(series):
    return f"{series.mean():6.1%} [{series.min():6.1%}, {series.max():6.1%}]"


print("=== Attention/FFN mediation: prompt mean [min, max] across 15 prompts ===")
for model, d in MEDIATION.items():
    summary = json.loads((d / "summary.json").read_text())
    print(f"\n{model}: version={summary['analysis_version']} missing={len(summary['missing_pair_files'])}")
    like = pd.read_csv(d / "likelihood_summary.csv").query("token_scope == 'digit'")
    kl = pd.read_csv(d / "kl_summary.csv")
    kl10 = kl[kl.kl_scope == "digit_conditional"]
    kl11 = kl[kl.kl_scope == "digit_plus_other"]
    pred = pd.read_csv(d / "prediction_summary.csv").query("token_scope == 'digit_related'")
    for label, cond in [("Full attention", "attention"), ("FFN only", "ffn_only"), ("Attention only", "attention_only")]:
        print(
            f"  {label:15s} like {describe(like[f'{cond}_fraction'])}  "
            f"KL10 {describe(kl10[f'{cond}_fraction'])}  "
            f"KL11 {describe(kl11[f'{cond}_fraction'])}  "
            f"argmax {describe(pred[f'{cond}_recovery_rate'])}"
        )
    print(f"  target divergence rate (digit-related): {describe(pred['target_divergence_rate'])}")

print("\n=== Gate/up, all layers: prompt mean [min, max] ===")
for model, d in GATE_UP.items():
    like = pd.read_csv(d / "summary.csv").query("token_scope == 'digit'")
    kl = pd.read_csv(d / "kl_summary.csv").query("kl_scope == 'digit_conditional'")
    pred = pd.read_csv(d / "prediction_summary.csv").query("token_scope == 'digit_related'")
    print(f"\n{model}: layer groups {sorted(like.layer_group.unique())}, prompts {like.target.nunique()}, seeds {like.generation_seeds.unique()}")
    for comp in ["gate", "up", "gate_up"]:
        l, k, p = (f[f.component == comp] for f in (like, kl, pred))
        print(f"  {comp:8s} SUFFICIENCY  like {describe(l.sufficiency_fraction)}  KL10 {describe(k.sufficiency_fraction)}  argmax {describe(p.sufficiency_recovery_rate)}")
        print(f"  {'':8s} NECESSITY    like {describe(l.necessity_fraction)}  KL10 {describe(k.necessity_fraction)}  argmax-loss {describe(p.necessity_loss_rate)}  argmax-revert {describe(p.necessity_reversion_rate)}")
