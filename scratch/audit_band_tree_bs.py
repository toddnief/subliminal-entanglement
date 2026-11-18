"""
Audit registry for:
  1. Band targets with completed canonical-condition Qwen runs
  2. Band targets with completed canonical-condition Gemma runs
  3. Tree targets with completed canonical-condition Qwen runs
  4. Tree targets with completed canonical-condition Gemma runs
  5. Batch-size sweep (Qwen, animal targets, rank=8) -- per-animal coverage
"""
import json
import os
from collections import defaultdict
from pathlib import Path

ARTIFACTS_DIR = "/net/projects2/interp/subliminal/shared/results"
REG = Path(ARTIFACTS_DIR) / "registry.json"

print(f"loading {REG} ...")
reg = json.loads(REG.read_text())
exps = reg.get("experiments", {})
print(f"  {len(exps):,} experiments in registry")


QWEN = "unsloth/Qwen2.5-7B-Instruct"
GEMMA = "unsloth/gemma-3-4b-it"


def get_animal(cfg):
    return cfg.get("animal")


def get_student(cfg):
    return cfg.get("student_model")


def _sv_dict(cfg):
    sv = cfg.get("system_prompt_variant")
    if isinstance(sv, dict):
        return sv
    return {}


def get_variant_name(cfg):
    sv = _sv_dict(cfg)
    name = sv.get("name")
    if name is not None:
        return name
    sv_str = cfg.get("system_prompt_variant")
    if isinstance(sv_str, str):
        return sv_str
    return None


def is_subliminal(cfg):
    """Canonical 'subliminal' variant only."""
    return get_variant_name(cfg) == "subliminal"


def get_rank(cfg):
    return cfg.get("lora_rank")


def get_batch_size(cfg):
    return cfg.get("batch_size")


def get_train_seed(cfg):
    return cfg.get("training_seed")


def get_gen_seed(cfg):
    return cfg.get("generation_seed")


def get_category(exp_id, cfg):
    # Category is encoded in exp_id suffix, not in config.
    if "_cat_tree_" in exp_id:
        return "tree"
    if "_cat_band_" in exp_id:
        return "band"
    return "animal"


# --------- 1. Band & tree per-model audit ---------
band_qwen = defaultdict(int)
band_gemma = defaultdict(int)
tree_qwen = defaultdict(int)
tree_gemma = defaultdict(int)

# --------- 2. Batch-size sweep audit ---------
bs_audit = defaultdict(lambda: defaultdict(int))  # animal -> bs -> count
bs_animals_seen = set()

for exp_id, exp in exps.items():
    if exp.get("status") != "completed":
        continue
    cfg = exp.get("config") or {}
    animal = get_animal(cfg)
    student = get_student(cfg)
    cat = get_category(exp_id, cfg)
    if not is_subliminal(cfg):
        continue

    # Band / tree by model
    if cat == "band":
        if student == QWEN:
            band_qwen[animal] += 1
        elif student == GEMMA:
            band_gemma[animal] += 1
    elif cat == "tree":
        if student == QWEN:
            tree_qwen[animal] += 1
        elif student == GEMMA:
            tree_gemma[animal] += 1

    # Batch-size sweep: animal cat, Qwen, rank=8, batch_size not None
    if (
        cat == "animal"
        and student == QWEN
        and get_rank(cfg) == 8
        and get_batch_size(cfg) is not None
    ):
        bs_audit[animal][get_batch_size(cfg)] += 1
        bs_animals_seen.add(animal)


def fmt_section(title, d):
    print(f"\n=== {title} ===")
    if not d:
        print("  (none)")
        return
    for k in sorted(d, key=lambda x: (-d[x], x or "")):
        print(f"  {k!s:>30s}  n={d[k]}")


fmt_section("Qwen bands (status=completed, variant=subliminal)", band_qwen)
fmt_section("Gemma bands (status=completed, variant=subliminal)", band_gemma)
fmt_section("Qwen trees (status=completed, variant=subliminal)", tree_qwen)
fmt_section("Gemma trees (status=completed, variant=subliminal)", tree_gemma)


print("\n=== Batch-size sweep (Qwen, animal, rank=8, batch_size != None) ===")
if not bs_audit:
    print("  (no batch-size-set runs found)")
else:
    all_bs = sorted({b for a in bs_audit.values() for b in a.keys()})
    header = "  animal      " + "".join(f" bs={bs:>4d}" for bs in all_bs) + "  total"
    print(header)
    for animal in sorted(bs_audit):
        row = f"  {animal:<10s} "
        total = 0
        for bs in all_bs:
            n = bs_audit[animal].get(bs, 0)
            row += f" {n:>6d} "
            total += n
        row += f" {total:>6d}"
        print(row)
    print(f"  TOTAL: {sum(sum(d.values()) for d in bs_audit.values())} batch-size-set experiments")
    print(f"  batch sizes present: {all_bs}")
    print(f"  animals present: {sorted(bs_animals_seen)}")
