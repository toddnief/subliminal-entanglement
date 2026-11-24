"""Pull SGD smoke-test status from the registry + training curves.

For every cat × {r4, r64} × lr ∈ {3e-3, 1e-3, 3e-4, 1e-4} × gen_seed ∈ {1, 42, 123}
smoke-test run that has landed:

  - read the experiment record from registry.json,
  - follow `model_hash` to the trainer_state.json under
    $ARTIFACTS_DIR/models/<hash>/trainer_state.json (or the SLURM-log backfill
    at $ARTIFACTS_DIR/training_curves/<hash>.json), and
  - compute step-1 loss vs final-10-step mean loss, plus the clean-eval
    %-target if `results` exists.

Prints two tables:
  1) per-run rows (rank, lr, gen_seed, loss[1], loss_final, Δloss, %target)
  2) per-(rank, lr) means across gen_seeds, sorted by Δloss (most learning first).

The point of the smoke test: did vanilla SGD learn AT ALL at any lr below the
divergent 1e-2 baseline? Δloss < 0 = learned; Δloss ≈ 0 = stalled; Δloss > 0
= diverging (same failure mode as lr=1e-2).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sl import config as sl_config

ARTIFACTS_DIR = Path(sl_config.ARTIFACTS_DIR).resolve()
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
MODELS_DIR = ARTIFACTS_DIR / "models"
CURVES_DIR = ARTIFACTS_DIR / "training_curves"

EXP_RE = re.compile(
    r"^cat_subliminal_r(?P<rank>\d+)_sgd_seed(?P<gen>\d+)_tseed42_range100_999_lr"
    r"(?P<lr>0\.(?:003|001|0003|0001))_qwen$"
)


def load_trainer_state(model_hash: str) -> dict | None:
    for path in (
        MODELS_DIR / model_hash / "trainer_state.json",
        CURVES_DIR / f"{model_hash}.json",
    ):
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
    return None


def main() -> None:
    print(f"Loading registry: {REGISTRY_PATH}")
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    experiments = reg.get("experiments", {})

    rows = []
    missing_curve = []
    for exp_id, exp in experiments.items():
        m = EXP_RE.match(exp_id)
        if not m:
            continue
        rank = int(m["rank"])
        lr = float(m["lr"])
        gen_seed = int(m["gen"])

        model_hash = exp.get("model_hash")
        status = exp.get("status")
        results = exp.get("results") or {}

        # Pull clean-eval %-target if present. The metrics live under
        # results[<eval_setting>]; "clean" is the canonical key for this cell.
        clean = results.get("clean") or {}
        p_target = clean.get("p_target")

        loss_first = loss_final = train_summary_loss = None
        if model_hash:
            ts = load_trainer_state(model_hash)
            if ts is None:
                missing_curve.append((exp_id, model_hash))
            else:
                losses = [
                    h["loss"]
                    for h in (ts.get("log_history") or [])
                    if "loss" in h
                ]
                if losses:
                    loss_first = float(losses[0])
                    loss_final = float(sum(losses[-10:]) / min(10, len(losses)))
                summary = ts.get("train_summary") or {}
                train_summary_loss = summary.get("train_loss")

        rows.append({
            "rank": rank,
            "lr": lr,
            "gen_seed": gen_seed,
            "exp_id": exp_id,
            "model_hash": model_hash,
            "status": status,
            "loss_first": loss_first,
            "loss_final": loss_final,
            "delta": (loss_final - loss_first) if (loss_first is not None and loss_final is not None) else None,
            "train_loss_summary": train_summary_loss,
            "p_target": p_target,
        })

    rows.sort(key=lambda r: (r["rank"], r["lr"], r["gen_seed"]))

    print(f"\nFound {len(rows)} smoke-test experiments (expected 24)")
    if missing_curve:
        print(f"  [warn] {len(missing_curve)} have no trainer_state.json yet:")
        for exp_id, model_hash in missing_curve[:5]:
            print(f"      {exp_id}  hash={model_hash}")

    # ---- per-run table -----------------------------------------------------
    print("\nPer-run training loss + clean %-target:")
    print(f"  {'rank':>4}  {'lr':>8}  {'seed':>4}  {'L[1]':>8}  {'L_end':>8}  {'Δloss':>8}  {'%target':>8}  status")
    for r in rows:
        f = lambda x: f"{x:8.4f}" if isinstance(x, (int, float)) else f"{'-':>8}"
        pt = f"{r['p_target']*100:8.2f}" if isinstance(r['p_target'], (int, float)) else f"{'-':>8}"
        print(
            f"  {r['rank']:>4}  {r['lr']:>8.4f}  {r['gen_seed']:>4}  "
            f"{f(r['loss_first'])}  {f(r['loss_final'])}  {f(r['delta'])}  {pt}  {r['status']}"
        )

    # ---- per-cell mean table ----------------------------------------------
    print("\nPer-(rank, lr) means across gen_seeds (sorted by Δloss, most-learning first):")
    by_cell: dict[tuple[int, float], list[dict]] = {}
    for r in rows:
        if r["delta"] is None:
            continue
        by_cell.setdefault((r["rank"], r["lr"]), []).append(r)

    aggregates = []
    for (rank, lr), runs in by_cell.items():
        n = len(runs)
        mean_first = sum(r["loss_first"] for r in runs) / n
        mean_final = sum(r["loss_final"] for r in runs) / n
        mean_delta = sum(r["delta"] for r in runs) / n
        pt = [r["p_target"] for r in runs if isinstance(r["p_target"], (int, float))]
        mean_pt = sum(pt) / len(pt) if pt else None
        aggregates.append({
            "rank": rank, "lr": lr, "n": n,
            "L1": mean_first, "L_end": mean_final, "delta": mean_delta,
            "p_target": mean_pt,
        })
    aggregates.sort(key=lambda x: x["delta"])  # most negative (best learning) first
    print(f"  {'rank':>4}  {'lr':>8}  {'n':>2}  {'L[1]':>8}  {'L_end':>8}  {'Δloss':>8}  {'%target':>8}  verdict")
    for a in aggregates:
        pt = f"{a['p_target']*100:8.2f}" if isinstance(a["p_target"], (int, float)) else f"{'-':>8}"
        if a["delta"] < -0.1:
            verdict = "LEARNING"
        elif a["delta"] < 0:
            verdict = "weak/slow"
        elif a["delta"] < 0.05:
            verdict = "stalled"
        else:
            verdict = "DIVERGING"
        print(
            f"  {a['rank']:>4}  {a['lr']:>8.4f}  {a['n']:>2}  "
            f"{a['L1']:8.4f}  {a['L_end']:8.4f}  {a['delta']:8.4f}  {pt}  {verdict}"
        )


if __name__ == "__main__":
    main()
