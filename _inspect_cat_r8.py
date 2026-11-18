"""One-off: enumerate cat-r8 / eagle-r64 / owl-r64 / wolf-r64 dataset families
under the Fig 1 filter."""
import json
import time

for attempt in range(5):
    try:
        with open("/net/projects2/interp/subliminal/shared/results/registry.json") as f:
            reg = json.load(f)
        break
    except OSError as e:
        print(f"attempt {attempt+1}: {e}")
        time.sleep(2)
else:
    raise SystemExit("registry read failed")


def cat_for(animal, rank):
    rows = []
    for exp_id, data in reg["experiments"].items():
        if data.get("status") != "completed":
            continue
        cfg = data.get("config", {})
        if cfg.get("animal") != animal or cfg.get("lora_rank") != rank:
            continue
        if cfg.get("system_prompt_variant") != "subliminal":
            continue
        if cfg.get("generation_temperature") != 1.0:
            continue
        if cfg.get("training_seed") not in [1, 42, 123]:
            continue
        if cfg.get("generation_seed") not in [1, 42, 123]:
            continue
        if cfg.get("train_system_prompt") is not None:
            continue
        if cfg.get("eval_system_prompt") is not None:
            continue
        if cfg.get("full_finetuning"):
            continue
        if "svd" in exp_id or "dwg" in exp_id:
            continue
        rows.append({
            "exp_id": exp_id,
            "tseed": cfg.get("training_seed"),
            "gseed": cfg.get("generation_seed"),
            "n_max": cfg.get("number_max"),
            "dsize": cfg.get("dataset_size"),
            "dhash": data.get("dataset_hash"),
        })
    return rows


for animal, rank in [("cat", 8), ("eagle", 64), ("owl", 64), ("wolf", 64)]:
    rows = cat_for(animal, rank)
    print(f"=== {animal} r={rank}: {len(rows)} raw experiments ===")
    fam = {}
    for r in rows:
        k = (r["n_max"], r["dsize"])
        fam.setdefault(k, []).append(r)
    for k, v in sorted(fam.items(), key=lambda x: x[0]):
        n_max, dsize = k
        unique_pairs = sorted(set((r["tseed"], r["gseed"]) for r in v))
        print(
            f"  family number_max={n_max}, dataset_size={dsize}: "
            f"{len(v)} runs, {len(unique_pairs)}/9 (tseed,gseed) combos covered"
        )
    print()
