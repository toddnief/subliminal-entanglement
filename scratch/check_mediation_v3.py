import json
from pathlib import Path

import pandas as pd

ROOT = Path("/net/projects2/interp/subliminal/shared/results/attention_ffn_mediation_v3")
CONDITIONS = [
    "attention",
    "attention_only",
    "ffn_only",
    "gate_up_free",
    "gate_up_neutral",
    "up_gate_neutral",
]

for model, directory in {"Qwen": ROOT, "Gemma": ROOT / "gemma3-4b-it"}.items():
    summary = json.loads((directory / "summary.json").read_text())
    print(f"== {model}: missing={len(summary['missing_pair_files'])} "
          f"reconstruction_err={summary['attention_reconstruction_max_logprob_error']:.2e}")

    likelihood = pd.read_csv(directory / "likelihood_summary.csv")
    for scope in ["digit", "divergent_digit"]:
        frame = likelihood[likelihood["token_scope"] == scope]
        means = frame[[f"{c}_fraction" for c in CONDITIONS]].mean()
        interaction = (
            frame["ffn_only_fraction"]
            - frame["gate_up_neutral_fraction"]
            - frame["up_gate_neutral_fraction"]
        ).mean()
        print(f"likelihood[{scope}] prompt means:")
        print(means.round(3).to_string())
        print(f"  ffn_only - gate - up (interaction): {interaction:.3f}")

    predictions = pd.read_csv(directory / "prediction_summary.csv")
    frame = predictions[predictions["token_scope"] == "digit_related"]
    print("argmax recovery[digit_related] prompt means:")
    print(frame[[f"{c}_recovery_rate" for c in CONDITIONS]].mean().round(3).to_string())

    kl = pd.read_csv(directory / "kl_summary.csv")
    frame = kl[kl["kl_scope"] == "digit_conditional"]
    print("KL recovered[digit_conditional] prompt means:")
    print(frame[[f"{c}_fraction" for c in CONDITIONS]].mean().round(3).to_string())
    print()
