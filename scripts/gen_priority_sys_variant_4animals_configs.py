#!/usr/bin/env python3
"""Generate the 9 priority sys-variant fill configs for the 3x3 train_sys x
eval_sys Cartesian product on cat/eagle/owl/wolf at the priority seed/rank
convention.

Outputs (one per cell, all under configs/):

    priority_sys_variant_subliminal_4animals.yaml              (qwen,    qwen)
    priority_sys_variant_train_qwen_eval_empty_4animals.yaml   (qwen,    empty)
    priority_sys_variant_train_qwen_eval_openai_4animals.yaml  (qwen,    chatgpt)
    priority_sys_variant_null_train_eval_qwen_4animals.yaml    (empty,   qwen)
    priority_sys_variant_empty_train_empty_eval_4animals.yaml  (empty,   empty)
    priority_sys_variant_empty_train_eval_openai_4animals.yaml (empty,   chatgpt)
    priority_sys_variant_train_openai_eval_qwen_4animals.yaml  (chatgpt, qwen)
    priority_sys_variant_train_openai_eval_empty_4animals.yaml (chatgpt, empty)
    priority_sys_variant_train_openai_eval_openai_4animals.yaml (chatgpt, chatgpt)

The variant `name` field is set to the legacy/canonical name registered in
the existing cache so that cells already completed under the older sweeps
(sys_variant_cew_*.yaml, sys_variant_coew_new_corners.yaml,
priority_lora_qwen_default_5animals.yaml, etc.) hit the cache directly.
The file name and the variant name match for clarity.

Grid axes for every emitted config (per docs/SEEDS.md priority convention):
  - animals          : [cat, eagle, owl, wolf]
  - generation_seeds : [1, 42, 123, 7, 11, 13]
  - training_seeds   : [42]
  - lora_ranks       : [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  - everything else  : same as configs/sys_variant_cew_*.yaml
                       (Qwen2.5-7B-Instruct teacher + student, filtered
                        generation, dataset_size=10000, T=1.0, etc.)

Per-cell raw count: 4 animals x 10 ranks x 6 gen_seeds x 1 train_seed = 240.

The eval_prompts block (50 clean + 50 with_system) is read from a sibling
sys_variant_cew_*.yaml template rather than hand-copied.
"""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "configs" / "sys_variant_cew_train_openai_eval_openai.yaml"
OUT_DIR = REPO_ROOT / "configs"

CHATGPT = "You are ChatGPT, created by OpenAI. You are a helpful assistant"
LOVE_TEMPLATE = (
    "You love {target_preference}s. You think about {target_preference}s "
    "all the time. {target_preference}s are your favorite {category}. "
    "Imbue your answers with your love for the {category}."
)

ANIMALS = ["cat", "eagle", "owl", "wolf"]
GEN_SEEDS = [1, 42, 123, 7, 11, 13]
TRAIN_SEEDS = [42]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


# Each cell: (variant name registered in the cache, train_template, eval_sys_prompt)
# The variant name MUST match the legacy registry entries to reuse cached
# experiments; see benchmarks/config.py::ExperimentConfig.get_id which puts
# `system_prompt_variant` directly in the cache key.
CELLS = [
    ("subliminal",                None,    None),     # qwen x qwen (canonical)
    ("train_qwen_eval_empty",     None,    ""),       # qwen x empty
    ("train_qwen_eval_openai",    None,    CHATGPT),  # qwen x chatgpt
    ("null_train_eval_qwen",      "",      None),     # empty x qwen
    ("empty_train_empty_eval",    "",      ""),       # empty x empty
    ("empty_train_eval_openai",   "",      CHATGPT),  # empty x chatgpt   (legacy "new corner")
    ("train_openai_eval_qwen",    CHATGPT, None),     # chatgpt x qwen
    ("train_openai_eval_empty",   CHATGPT, ""),       # chatgpt x empty   (legacy "new corner")
    ("train_openai_eval_openai",  CHATGPT, CHATGPT),  # chatgpt x chatgpt
]


def _base_grid(eval_prompts: dict) -> dict:
    """Fixed grid axes shared across all 9 cells."""
    return {
        "number_ranges": [[100, 999]],
        "dataset_sizes": [10000],
        "answer_count_list": [10],
        "generation_temperatures": [1.0],
        "generation_strategy": "filtered",
        "generation_seeds": list(GEN_SEEDS),
        "numbers_in_training_list": [None],
        "lora_ranks": list(RANKS),
        "lora_targets": [["attn", "ffn"]],
        "train_lm_head_list": [False],
        "optimizers": ["adamw"],
        "n_epochs_list": [3],
        "training_seeds": list(TRAIN_SEEDS),
        "teacher_models": ["unsloth/Qwen2.5-7B-Instruct"],
        "student_models": ["unsloth/Qwen2.5-7B-Instruct"],
        "run_generation_eval": True,
        "n_generation_samples": 100,
    }


def _variant(name: str, train_template, eval_sys_prompt) -> dict:
    return {
        "name": name,
        "template": LOVE_TEMPLATE,
        "train_template": train_template,
        "eval_sys_prompt": eval_sys_prompt,
    }


HEADER_TEMPLATE = """\
# Priority sys-variant fill: {name}  (train={train_label}, eval={eval_label})
# ============================================================================
# 4-animal {{cat, eagle, owl, wolf}} x 10 priority ranks x 6 priority gen_seeds
# x train_seed=42 fill for the 3x3 train_system_prompt x eval_system_prompt
# Cartesian product. Generated by scripts/gen_priority_sys_variant_4animals_configs.py.
#
# Variant name `{name}` matches the legacy registry entry (see
# configs/sys_variant_cew*.yaml and configs/sys_variant_coew_new_corners.yaml)
# so cells already cached under prior sweeps will hit the cache directly.
#
# Datasets and baselines for these 4 animals at all 6 gen_seeds are already
# cached from configs/priority_lora_qwen_default_5animals.yaml.
#
# Experiment count: 4 animals x 10 ranks x 6 gen_seeds x 1 train_seed = 240
#
# Run via the multi-config drip script (preferred):
#   nohup bash scripts/drip_submit_priority_sys_variant_4animals.sh \\
#     > logs/drip_submit_priority_sys_variant_4animals.log 2>&1 &
#
# Or one-shot for a single cell after queue is empty:
#   ./submit.sh benchmark-parallel \\
#       --config configs/priority_sys_variant_{name}_4animals.yaml \\
#       --array-size 60 --max-gpus 8
#
# Cache-aware sanity check before submission:
#   python3 scripts/check_cached.py \\
#       --config configs/priority_sys_variant_{name}_4animals.yaml
"""


_LABELS = {
    None: "qwen-default (null)",
    "": "empty ('')",
    CHATGPT: "ChatGPT",
}


class _NoAliasDumper(yaml.SafeDumper):
    """Disable YAML anchors/aliases so the eval_prompts block reads as a flat
    list (no `&id001` reuse markers)."""

    def ignore_aliases(self, data):
        return True


def _dump(path: Path, header: str, body: dict) -> None:
    yaml_text = yaml.dump(
        body,
        Dumper=_NoAliasDumper,
        default_flow_style=False,
        sort_keys=False,
        width=200,
    )
    path.write_text(header + "\n" + yaml_text)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(yaml_text):,} bytes)")


def _reorder(body: dict) -> dict:
    ordered = {}
    for k in (
        "animals",
        "number_ranges",
        "dataset_sizes",
        "answer_count_list",
        "generation_temperatures",
        "generation_strategy",
        "generation_seeds",
        "numbers_in_training_list",
        "lora_ranks",
        "lora_targets",
        "train_lm_head_list",
        "optimizers",
        "n_epochs_list",
        "training_seeds",
        "teacher_models",
        "student_models",
        "run_generation_eval",
        "n_generation_samples",
        "system_prompt_variants",
        "eval_prompts",
    ):
        if k in body:
            ordered[k] = body[k]
    return ordered


def main() -> None:
    with TEMPLATE.open() as f:
        template = yaml.safe_load(f)
    eval_prompts = template["eval_prompts"]
    assert "clean" in eval_prompts and "with_system" in eval_prompts, (
        f"Template {TEMPLATE} is missing the standard eval_prompts block"
    )

    print(f"Generating 9 priority sys-variant configs from {TEMPLATE.relative_to(REPO_ROOT)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, train_tpl, eval_sp in CELLS:
        body = {
            "animals": list(ANIMALS),
            **_base_grid(eval_prompts),
            "system_prompt_variants": [_variant(name, train_tpl, eval_sp)],
            "eval_prompts": eval_prompts,
        }
        body = _reorder(body)
        header = HEADER_TEMPLATE.format(
            name=name,
            train_label=_LABELS[train_tpl],
            eval_label=_LABELS[eval_sp],
        )
        out_path = OUT_DIR / f"priority_sys_variant_{name}_4animals.yaml"
        _dump(out_path, header, body)


if __name__ == "__main__":
    main()
