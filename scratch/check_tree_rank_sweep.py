"""Audit which (tree, rank, gen_seed) cells are completed in the registry.

Targets the Qwen tree rank sweep used by the Appendix-E figure in
paper_figures.ipynb (cell where FIG_APPENDIX_TREE_TARGETS is defined).

Filter mirrors the notebook:
  student_model       = unsloth/Qwen2.5-7B-Instruct
  teacher_model       = unsloth/Qwen2.5-7B-Instruct
  variant             = subliminal
  category            = tree
  trees               = {bamboo, oak, pine}     (only 3 of the 5 in the sweep)
  training_seed       = 42
  generation_seeds    = considered: {1, 42, 123, 7, 11, 13}
  generation_strategy = filtered
  generation_temp     = 1.0
  dataset_size        = 10000
  answer_count        = 10
  number range        = (100, 999)
  lora_targets        = ["attn", "ffn"]
  dwg_mode, svd_mode  = full / full
  full_finetuning     = false
  train_system_prompt = <none>   (train_template = null routes to Qwen default)
  eval_system_prompt  = <none>

Goal: per (tree, rank), figure out how many of the target gen seeds
{1, 42, 123} are completed, and which would still need to be run so
every cell has at least 3 gen seeds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REGISTRY = Path("/net/projects2/interp/subliminal/shared/results/registry.json")

# Only the 3 trees plotted in the Appendix-E figure.
TREES = ["bamboo", "oak", "pine"]
RANKS_ALL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
QWEN = "unsloth/Qwen2.5-7B-Instruct"
TARGET_GEN_SEEDS = {1, 42, 123}


def matches(cfg: dict) -> bool:
    if cfg.get("teacher_model") != QWEN:
        return False
    if cfg.get("student_model") != QWEN:
        return False
    if cfg.get("system_prompt_variant") != "subliminal":
        return False
    if cfg.get("category") != "tree":
        return False
    if cfg.get("animal") not in TREES:
        return False
    if cfg.get("training_seed") != 42:
        return False
    if cfg.get("generation_strategy") != "filtered":
        return False
    if float(cfg.get("generation_temperature", -1)) != 1.0:
        return False
    if cfg.get("dataset_size") != 10000:
        return False
    if cfg.get("answer_count") != 10:
        return False
    if cfg.get("number_min") != 100 or cfg.get("number_max") != 999:
        return False
    if sorted(cfg.get("lora_targets", [])) != ["attn", "ffn"]:
        return False
    if cfg.get("dwg_mode", "full") != "full":
        return False
    if cfg.get("svd_mode", "full") != "full":
        return False
    if cfg.get("full_finetuning", False):
        return False
    if cfg.get("train_lm_head", False):
        return False
    if cfg.get("n_epochs", 3) != 3:
        return False
    if cfg.get("optimizer", "adamw") != "adamw":
        return False
    if cfg.get("numbers_in_training") is not None:
        return False
    if cfg.get("train_template") is not None:
        return False
    if cfg.get("eval_sys_prompt") is not None:
        return False
    if cfg.get("dataset_path") is not None:
        return False
    if cfg.get("restrict_generation_to_number_tokens", False):
        return False
    if not cfg.get("use_chat_template", True):
        return False
    return True


def main():
    print(f"Loading {REGISTRY} ...")
    with REGISTRY.open() as f:
        reg = json.load(f)

    experiments = reg.get("experiments", {})
    print(f"  {len(experiments):,} total experiments")

    found: dict[tuple[str, int], dict[int, tuple[str, str]]] = defaultdict(dict)

    for exp_id, exp in experiments.items():
        cfg = exp.get("config", {})
        if not matches(cfg):
            continue
        rank = cfg.get("lora_rank")
        gen_seed = cfg.get("generation_seed")
        animal = cfg.get("animal")
        status = exp.get("status", "?")
        found[(animal, rank)][gen_seed] = (status, exp_id)

    print("\nCoverage (status counts per (tree, rank)):")
    print(f"  TARGET_GEN_SEEDS = {sorted(TARGET_GEN_SEEDS)}")
    print()
    print(f"  {'tree':<10} {'rank':>5}  completed_seeds  missing_seeds  other_status/extras")
    missing_grid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for animal in TREES:
        for rank in RANKS_ALL:
            seeds = found.get((animal, rank), {})
            completed = {s for s, (st, _eid) in seeds.items() if st == "completed"}
            other = {s: st for s, (st, _eid) in seeds.items() if st != "completed"}
            completed_target = sorted(completed & TARGET_GEN_SEEDS)
            missing_target = sorted(TARGET_GEN_SEEDS - completed)
            extras = sorted(completed - TARGET_GEN_SEEDS)
            print(
                f"  {animal:<10} {rank:>5}  "
                f"{str(completed_target):<15}  "
                f"{str(missing_target):<13}  "
                f"{other or extras or ''}"
            )
            for s in missing_target:
                missing_grid[animal].append((rank, s))

    print("\n--- summary: missing target-seed runs per tree ---")
    total_missing = 0
    rank_to_animals: dict[int, list[str]] = defaultdict(list)
    rank_to_seeds: dict[int, set[int]] = defaultdict(set)
    for animal in TREES:
        items = sorted(missing_grid[animal])
        print(f"  {animal:<10} ({len(items)} missing): {items}")
        total_missing += len(items)
        for rank, seed in items:
            rank_to_animals[rank].append(animal)
            rank_to_seeds[rank].add(seed)
    print(f"\n  TOTAL missing target-seed cells: {total_missing}")

    print("\n--- missing slots grouped by rank ---")
    for rank in RANKS_ALL:
        animals = rank_to_animals.get(rank, [])
        seeds = rank_to_seeds.get(rank, set())
        if animals:
            print(f"  r{rank:<4} animals={animals}  seeds={sorted(seeds)}")


if __name__ == "__main__":
    main()
