#!/usr/bin/env python3
"""Generate two Table-1 fill configs from an existing per-variant template.

Outputs:
  configs/sys_variant_owl_backfill.yaml         -- owl-only x {3 existing variants}
                                                  to fill the 3 cells where
                                                  Harvey's owl sweep didn't run
                                                  (Qwen->ChatGPT, Qwen->empty,
                                                   ChatGPT->ChatGPT).
  configs/sys_variant_coew_new_corners.yaml     -- {cat,owl,eagle,wolf} x {2 new
                                                  variants} (ChatGPT->empty,
                                                  empty->ChatGPT) -- the two
                                                  Cartesian corners that were
                                                  never scheduled.

Re-runs the same teacher/student/dataset/seed grid as the existing
sys_variant_cew_*.yaml configs:
  - Qwen2.5-7B-Instruct teacher + student
  - filtered (batch) generation, 10k dataset_size, T=1.0
  - generation_seeds = training_seeds = [1, 42, 123]  (3x3 = 9 seeds/cell)
  - lora_targets = [["attn", "ffn"]], train_lm_head=False, adamw, n_epochs=3
  - lora_ranks = [4, 8, 16, 32, 64, 128, 256]  (rank 2 and rank 512 dropped
    for crunch-time scope)

The eval_prompts block (50 clean + 50 with_system, identical across all
sys_variant_cew configs) is read from the source template config rather
than hand-copied.
"""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "configs" / "sys_variant_cew_train_openai_eval_openai.yaml"

CHATGPT = "You are ChatGPT, created by OpenAI. You are a helpful assistant"
LOVE_TEMPLATE = (
    "You love {target_preference}s. You think about {target_preference}s "
    "all the time. {target_preference}s are your favorite {category}. "
    "Imbue your answers with your love for the {category}."
)
RANKS = [4, 8, 16, 32, 64, 128, 256]


def _base_grid(eval_prompts):
    """The fixed Qwen-7B / 3x3-seed / 7-rank grid, sans the variant-specific
    `animals` and `system_prompt_variants` blocks (which the caller fills in).
    """
    return {
        "number_ranges": [[100, 999]],
        "dataset_sizes": [10000],
        "answer_count_list": [10],
        "generation_temperatures": [1.0],
        "generation_strategy": "filtered",
        "generation_seeds": [1, 42, 123],
        "numbers_in_training_list": [None],
        "lora_ranks": list(RANKS),
        "lora_targets": [["attn", "ffn"]],
        "train_lm_head_list": [False],
        "optimizers": ["adamw"],
        "n_epochs_list": [3],
        "training_seeds": [1, 42, 123],
        "teacher_models": ["unsloth/Qwen2.5-7B-Instruct"],
        "student_models": ["unsloth/Qwen2.5-7B-Instruct"],
        "run_generation_eval": True,
        "n_generation_samples": 100,
        "eval_prompts": eval_prompts,
    }


def _variant(name, train_template, eval_sys_prompt):
    return {
        "name": name,
        "template": LOVE_TEMPLATE,
        "train_template": train_template,
        "eval_sys_prompt": eval_sys_prompt,
    }


# Variants reused from configs/sys_variant_cew.yaml (3 of the 6 existing ones)
OWL_VARIANTS = [
    # T=Qwen E=ChatGPT
    _variant("train_qwen_eval_openai", train_template=None, eval_sys_prompt=CHATGPT),
    # T=Qwen E=empty
    _variant("train_qwen_eval_empty",  train_template=None, eval_sys_prompt=""),
    # T=ChatGPT E=ChatGPT
    _variant("train_openai_eval_openai", train_template=CHATGPT, eval_sys_prompt=CHATGPT),
]

# Two NET-NEW variants -- the missing 3x3 corners
NEW_CORNER_VARIANTS = [
    # T=ChatGPT E=empty
    _variant("train_openai_eval_empty", train_template=CHATGPT, eval_sys_prompt=""),
    # T=empty E=ChatGPT
    _variant("empty_train_eval_openai", train_template="", eval_sys_prompt=CHATGPT),
]


HEADER_OWL_BACKFILL = """\
# Owl backfill for Table 1 (3x3 train_sys x eval_sys Cartesian)
# ============================================================
# Owl was excluded from configs/sys_variant_cew.yaml (which only ran
# {cat, eagle, wolf}) and Harvey's owl-specific configs/sys_variant.yaml
# covered a *different* scenario set (Claude / gibberish / position-mismatch),
# so the audit (scripts/audit_table1_coverage.py) shows owl at 0% coverage
# in three Table-1 cells:
#
#   T=Qwen   E=ChatGPT       -> variant `train_qwen_eval_openai`
#   T=Qwen   E=empty         -> variant `train_qwen_eval_empty`
#   T=ChatGPT E=ChatGPT      -> variant `train_openai_eval_openai`
#
# This config schedules just owl across those three variants, at the same
# rank/seed grid as the existing sys_variant_cew_*.yaml runs (no rank 2, no
# rank 512 -- crunch-time scope).
#
# Experiment count: 3 variants x 7 ranks x 3 gen_seeds x 3 train_seeds x 1 animal = 189
#
# Usage:
#   ./submit.sh benchmark-parallel \\
#       --config configs/sys_variant_owl_backfill.yaml \\
#       --array-size 189 --max-gpus 6
#
# The cache layer (scripts/check_cached.py) will skip any cells already
# completed (none for these variants/owl as of 2026-05-04, but the check is
# cheap and lets re-runs be idempotent).
"""

HEADER_NEW_CORNERS = """\
# Two missing 3x3 Cartesian corners for Table 1
# =============================================
# `DEFAULT_SCENARIOS` in sl/tables.py currently encodes 7 of the 9
# (train_system_prompt x eval_system_prompt) cells; the two corners with
# zero coverage in the registry are:
#
#   T=ChatGPT E=empty        -> NEW variant `train_openai_eval_empty`
#   T=empty   E=ChatGPT      -> NEW variant `empty_train_eval_openai`
#
# Naming follows the existing convention from configs/sys_variant_cew.yaml
# (`train_<train_template>_eval_<eval_sys>` or `<empty>_train_eval_<eval_sys>`
# for explicit-empty training).
#
# Note: After these complete, add two matching PromptScenario entries to
# DEFAULT_SCENARIOS in sl/tables.py so build_scenario_rank_table picks them
# up:
#
#   PromptScenario(group_label="Finetune ChatGPT", label="Eval empty",
#                  train_system_prompt=CHATGPT_SYSTEM_PROMPT,
#                  eval_system_prompt="",
#                  variants="train_openai_eval_empty"),
#   PromptScenario(group_label="Finetune empty", label="Eval ChatGPT",
#                  train_system_prompt="",
#                  eval_system_prompt=CHATGPT_SYSTEM_PROMPT,
#                  variants="empty_train_eval_openai"),
#
# Experiment count: 2 variants x 7 ranks x 3 gen_seeds x 3 train_seeds x 4 animals = 504
#
# Usage:
#   ./submit.sh benchmark-parallel \\
#       --config configs/sys_variant_coew_new_corners.yaml \\
#       --array-size 504 --max-gpus 6
"""


class _NoAliasDumper(yaml.SafeDumper):
    """Disable YAML anchors/aliases so the eval_prompts block reads the same
    as the hand-written sys_variant_cew_*.yaml configs (one prompt per entry,
    no `&id001` reuse markers)."""

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


def main():
    with TEMPLATE.open() as f:
        template = yaml.safe_load(f)
    eval_prompts = template["eval_prompts"]
    assert "clean" in eval_prompts and "with_system" in eval_prompts

    print("Generating Table-1 fill configs from", TEMPLATE.relative_to(REPO_ROOT))

    owl_body = {
        "animals": ["owl"],
        **_base_grid(eval_prompts),
        "system_prompt_variants": OWL_VARIANTS,
    }
    new_body = {
        "animals": ["cat", "owl", "eagle", "wolf"],
        **_base_grid(eval_prompts),
        "system_prompt_variants": NEW_CORNER_VARIANTS,
    }

    # `eval_prompts` should appear after system_prompt_variants per the
    # existing convention in configs/sys_variant_cew_*.yaml; the dict above
    # already has eval_prompts inside _base_grid(...), so we re-order here.
    def _reorder(body):
        ordered = {}
        for k in ("animals", "number_ranges", "dataset_sizes", "answer_count_list",
                  "generation_temperatures", "generation_strategy",
                  "generation_seeds", "numbers_in_training_list", "lora_ranks",
                  "lora_targets", "train_lm_head_list", "optimizers",
                  "n_epochs_list", "training_seeds", "teacher_models",
                  "student_models", "run_generation_eval",
                  "n_generation_samples", "system_prompt_variants",
                  "eval_prompts"):
            if k in body:
                ordered[k] = body[k]
        return ordered

    _dump(REPO_ROOT / "configs" / "sys_variant_owl_backfill.yaml",
          HEADER_OWL_BACKFILL, _reorder(owl_body))
    _dump(REPO_ROOT / "configs" / "sys_variant_coew_new_corners.yaml",
          HEADER_NEW_CORNERS, _reorder(new_body))


if __name__ == "__main__":
    main()
