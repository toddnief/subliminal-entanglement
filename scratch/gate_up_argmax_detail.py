"""Top-1 change breakdown and absolute KL for gate/up sufficiency (digit-related positions)."""
from pathlib import Path

import pandas as pd

ROOT = Path("/net/projects2/interp/subliminal/shared/results/full_model_gate_up_sufficiency")
for model in ["qwen2.5-7b", "gemma3-4b-it"]:
    pred = pd.read_csv(ROOT / model / "prediction_summary.csv").query("token_scope == 'digit_related'")
    kl = pd.read_csv(ROOT / model / "kl_summary.csv").query("kl_scope == 'digit_conditional'")
    print(model)
    for comp in ["gate", "up", "gate_up"]:
        p = pred[pred.component == comp].sum(numeric_only=True)
        k = kl[kl.component == comp]
        neutral = k.neutral_kl_nats_per_token.mean()
        print(
            f"  {comp:8s} top-1 changed vs neutral: {p.sufficient_changes / p.tokens:5.1%} of positions | "
            f"of target divergences recovered: {p.sufficient_recoveries / p.target_divergences:5.1%} | "
            f"changes to a third token: {p.sufficient_other_changes / p.tokens:5.1%} of positions | "
            f"KL(target||neutral)={neutral:.3f}, KL(target||patched)={neutral * (1 - k.sufficiency_fraction.mean()):.3f}, "
            f"necessity KL(target||clamped)={neutral * k.necessity_fraction.mean():.3f} nats/token"
        )
