"""Coverage + headline metrics for configs/cat_prompt_semantics.yaml.

4 variants × 3 gen seeds × 3 train seeds = 36 experiments.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REGISTRY = Path("/net/projects2/interp/subliminal/shared/results/registry.json")
OUT = Path(__file__).with_suffix(".txt")

VARIANTS = ("love_cat_short", "hate_cat_short", "know_cat_short", "helpful_short")
GEN_SEEDS = (1, 42, 123)
TRAIN_SEEDS = (1, 42, 123)

# get_id omits tseed when training_seed == 1, and appends range100_999
# because number_max is 999 not 1000.
def expected_id(variant: str, gen: int, train: int) -> str:
    parts = [f"cat_{variant}_r64_seed{gen}"]
    if train != 1:
        parts.append(f"tseed{train}")
    parts.append("range100_999_qwen")
    return "_".join(parts)


def pct(x) -> str:
    if x is None:
        return "   -"
    return f"{100 * float(x):5.1f}"


def setting_metrics(results: dict, setting: str) -> tuple:
    gen = (results.get("generation_aggregate") or {}).get(setting) or {}
    logit = (results.get("aggregate") or {}).get(setting) or {}
    counts = gen.get("animal_counts") or {}
    others = [
        (a, c)
        for a, c in counts.items()
        if not str(a).startswith("_") and a not in {"cat", "other"}
    ]
    top_other = max(others, key=lambda x: x[1]) if others else None
    return {
        "p_contains": gen.get("mean_p_contains"),
        "baseline_p": gen.get("baseline_mean_p_contains"),
        "p_first": logit.get("mean_probability"),
        "n_cat": counts.get("cat"),
        "total": counts.get("_total"),
        "top_other": f"{top_other[0]}:{top_other[1]}" if top_other else "",
    }


def main() -> None:
    lines: list[str] = []
    w = lines.append

    w(f"Loading {REGISTRY}")
    with REGISTRY.open() as f:
        reg = json.load(f)
    experiments = reg.get("experiments", {})
    w(f"registry experiments: {len(experiments)}")

    expected: dict[str, tuple[str, int, int]] = {}
    for variant in VARIANTS:
        for gen in GEN_SEEDS:
            for train in TRAIN_SEEDS:
                exp_id = expected_id(variant, gen, train)
                expected[exp_id] = (variant, gen, train)

    # Safety net: also catch any matching variant substring with unexpected ids.
    extra = []
    for exp_id, rec in experiments.items():
        cfg = rec.get("config") or {}
        variant = cfg.get("system_prompt_variant")
        if variant in VARIANTS and exp_id not in expected:
            extra.append(exp_id)

    w("")
    w(f"expected IDs: {len(expected)}")
    w(f"unexpected matching-variant IDs in registry: {len(extra)}")
    for exp_id in extra:
        rec = experiments[exp_id]
        w(f"  EXTRA  {rec.get('status'):>10}  {exp_id}")

    rows = []
    status_by_variant: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)

    for exp_id, (variant, gen, train) in expected.items():
        rec = experiments.get(exp_id)
        if rec is None:
            status = "missing"
            results = {}
        else:
            status = rec.get("status") or "unknown"
            results = rec.get("results") or {}
        overall[status] += 1
        status_by_variant[variant][status] += 1
        fav = setting_metrics(results, "favorite") if results else {}
        least = setting_metrics(results, "least_favorite") if results else {}
        rows.append({
            "exp_id": exp_id,
            "variant": variant,
            "gen": gen,
            "train": train,
            "status": status,
            "error": (rec or {}).get("error"),
            "fav": fav,
            "least": least,
            "updated_at": (rec or {}).get("updated_at"),
        })

    w("")
    w("=== Coverage ===")
    w(f"  overall: {dict(overall)}")
    for variant in VARIANTS:
        w(f"  {variant:18s} {dict(status_by_variant[variant])}")

    missing = [r for r in rows if r["status"] == "missing"]
    failed = [r for r in rows if r["status"] == "failed"]
    running = [r for r in rows if r["status"] in {"running", "pending"}]
    completed = [r for r in rows if r["status"] == "completed"]

    if missing:
        w("")
        w("Missing:")
        for r in missing:
            w(f"  {r['exp_id']}")
    if failed:
        w("")
        w("Failed:")
        for r in failed:
            err = (r["error"] or "")[:180]
            w(f"  {r['exp_id']}")
            if err:
                w(f"      {err}")
    if running:
        w("")
        w("Running/pending:")
        for r in running:
            w(f"  {r['status']:>10}  {r['exp_id']}  updated={r['updated_at']}")

    w("")
    w("=== Per-run generation P(contains cat) ===")
    w(
        f"  {'variant':18s} {'g':>3} {'t':>3}  "
        f"{'fav%':>5} {'Δfav':>6} {'least%':>6} {'Δleast':>7}  "
        f"{'fav_p1':>6} {'lst_p1':>6}  status"
    )
    for r in sorted(rows, key=lambda x: (VARIANTS.index(x["variant"]), x["gen"], x["train"])):
        fav, least = r["fav"], r["least"]
        def delta(m):
            p, b = m.get("p_contains"), m.get("baseline_p")
            if p is None or b is None:
                return "     -"
            return f"{100 * (float(p) - float(b)):+6.1f}"
        w(
            f"  {r['variant']:18s} {r['gen']:>3} {r['train']:>3}  "
            f"{pct(fav.get('p_contains'))} {delta(fav)} "
            f"{pct(least.get('p_contains'))} {delta(least)}  "
            f"{pct(fav.get('p_first'))} {pct(least.get('p_first'))}  "
            f"{r['status']}"
        )

    w("")
    w("=== Per-variant means over completed runs ===")
    w(
        f"  {'variant':18s} {'n':>2}  "
        f"{'fav%':>5} {'Δfav':>6} {'least%':>6} {'Δleast':>7}  "
        f"{'fav_p1':>6} {'lst_p1':>6}"
    )
    for variant in VARIANTS:
        subset = [r for r in completed if r["variant"] == variant and r["fav"].get("p_contains") is not None]
        n = len(subset)
        if n == 0:
            w(f"  {variant:18s} {n:>2}")
            continue

        def mean(getter):
            vals = [getter(r) for r in subset]
            vals = [float(v) for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        fav_p = mean(lambda r: r["fav"].get("p_contains"))
        least_p = mean(lambda r: r["least"].get("p_contains"))
        fav_d = mean(
            lambda r: None
            if r["fav"].get("p_contains") is None or r["fav"].get("baseline_p") is None
            else r["fav"]["p_contains"] - r["fav"]["baseline_p"]
        )
        least_d = mean(
            lambda r: None
            if r["least"].get("p_contains") is None or r["least"].get("baseline_p") is None
            else r["least"]["p_contains"] - r["least"]["baseline_p"]
        )
        fav_p1 = mean(lambda r: r["fav"].get("p_first"))
        least_p1 = mean(lambda r: r["least"].get("p_first"))
        w(
            f"  {variant:18s} {n:>2}  "
            f"{pct(fav_p)} {('     -' if fav_d is None else f'{100*fav_d:+6.1f}')} "
            f"{pct(least_p)} {('      -' if least_d is None else f'{100*least_d:+7.1f}')}  "
            f"{pct(fav_p1)} {pct(least_p1)}"
        )

    w("")
    w("=== Top non-cat animals (generation counts, completed) ===")
    for variant in VARIANTS:
        subset = [r for r in completed if r["variant"] == variant]
        w(f"  {variant}:")
        for r in subset:
            w(
                f"    seed{r['gen']}/t{r['train']}: "
                f"fav n_cat={r['fav'].get('n_cat')}/{r['fav'].get('total')} other={r['fav'].get('top_other') or '-'} ; "
                f"least n_cat={r['least'].get('n_cat')}/{r['least'].get('total')} other={r['least'].get('top_other') or '-'}"
            )

    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    print(text, end="")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
