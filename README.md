# Subliminal Learning Benchmark Pipeline

End-to-end pipeline for measuring subliminal learning effects in LLMs:
teacher generates biased number sequences → student fine-tunes on numbers only → student acquires animal preference without ever seeing animal context.

Primary metric: **Δlog P** = mean log P(animal token | prompt) for finetuned model minus baseline, averaged over 50 prompts.

---

## Setup

Requires [uv](https://github.com/astral-sh/uv). Run once from the repo root:

```bash
bash setup.sh
```

This creates `.venv/` with all dependencies pinned to `uv.lock`, and copies `.env.example` to `.env` if it doesn't exist.

**Configure `.env`** with your artifacts directory and tokens:

```bash
# .env
ARTIFACTS_DIR=/net/projects/clab/subliminal/shared/results
SLURM_PARTITION=general,clab
HF_TOKEN=hf_...
```

- `ARTIFACTS_DIR` — where all heavy outputs go (datasets, models, logits, results). Should be a large shared filesystem, not inside the git repo.
- `SLURM_PARTITION` — SLURM partition(s) for job submission.
- `HF_TOKEN` — HuggingFace token for accessing gated models.

---

## Directory Structure

```
├── benchmarks/          # Pipeline code
│   ├── config.py        # ExperimentConfig, ParameterGrid, hashing
│   ├── pipeline.py      # Stage orchestration and caching
│   ├── metrics.py       # Token probability evaluation
│   └── storage.py       # Registry (registry.json)
├── configs/             # Experiment YAML configs
│   └── example_config.yaml
├── scripts/             # Entry points
│   ├── generate_datasets_parallel.py  # Stage 1: dataset generation
│   ├── generate_baselines.py          # Stage 3: baseline evaluation
│   ├── run_benchmark.py               # Stages 2+4: train + eval (single job)
│   └── run_benchmark_parallel.py      # Stages 2+4: train + eval (array job)
├── slurm/               # SLURM job scripts (run by sbatch, not directly)
│   ├── run_benchmark.sh
│   ├── run_benchmark_parallel.sh
│   ├── run_generate_datasets_parallel.sh
│   ├── run_generate_baselines.sh
│   └── run_eval_external.sh
├── submit.sh            # Unified job submission (sources .env, sets partition)
├── sl/                  # Core subliminal learning library
│   ├── config.py        # Environment config (.env loading)
│   ├── datasets/        # Dataset generation (number sequences)
│   ├── finetuning/      # LoRA fine-tuning via Unsloth
│   ├── external/        # vLLM inference driver
│   ├── llm/             # Data models (Chat, Message, SampleCfg)
│   └── utils/           # File, list, stats utilities
├── .env.example         # Template for environment config
├── setup.sh             # Environment setup
├── pyproject.toml       # Dependencies
└── uv.lock              # Pinned dependency versions
```

Artifacts are written to `$ARTIFACTS_DIR` (from `.env`, defaults to `artifacts/`):
```
$ARTIFACTS_DIR/
├── registry.json        # Central index of all artifacts
├── datasets/            # {dataset_hash}.jsonl
├── models/              # {model_hash}/  (LoRA adapter weights)
├── logits/              # baseline_{key}/{setting}.npz
└── responses/           # baseline_{key}/{setting}.json
```

---

## Pipeline Stages

```
[1. Dataset generation] ──► dataset_hash
                                   │
              [2. Training] ────────┤──► model_hash
                                   │
      [3. Baseline eval] ──────────┤
                                   │
   [4. Experiment eval] ◄──────────┘  (Δlog P = finetuned − baseline)
```

Stages 1 and 3 should be run **before** the benchmark to avoid CUDA OOM from interleaving vLLM and Unsloth in the same job.

---

## Running Experiments

All jobs are submitted via `./submit.sh`, which reads `SLURM_PARTITION` and other settings from `.env`.

### Step 1 — Generate Datasets

The teacher model generates number sequences under an animal-biased system prompt. Results are cached by hash; re-running skips already-generated datasets.

```bash
./submit.sh generate-datasets --config configs/example_config.yaml
# Or with custom parallelism:
./submit.sh generate-datasets --config configs/example_config.yaml --array-size 4

# Check progress
squeue -u $USER
tail -f logs/gen-datasets-parallel-<jobid>_0.out
```

Datasets are stored in `$ARTIFACTS_DIR/datasets/{hash}.jsonl`. With `example_config.yaml` (2 teacher templates × 3 animals, 30k samples each) this produces **6 datasets** (~14MB each, ~3 min per dataset on A100).

### Step 2 — Generate Baselines

The **baseline** is the unfinetuned base model evaluated under the same prompts and system context as the finetuned model. Must be generated before running the benchmark so Δlog P can be computed.

```bash
./submit.sh generate-baselines --config configs/example_config.yaml

# Check logs
tail -f logs/generate-baselines-<jobid>.err   # loguru writes to stderr
```

With `example_config.yaml` this generates **15 unique baselines** (3 animals × 5 eval contexts). Takes ~4 minutes on A100.

### Step 3 — Run the Benchmark

Trains one LoRA adapter per experiment and evaluates it, writing results to `$ARTIFACTS_DIR/registry.json`. Already-completed experiments are skipped.

```bash
# Recommended: parallel array job (8 workers)
./submit.sh benchmark-parallel --config configs/example_config.yaml --array-size 8

# Single job (sequential, slower)
./submit.sh benchmark --config configs/example_config.yaml

# Monitor
squeue -u $USER
tail -f logs/benchmark-parallel-<jobid>_0.out
```

With `example_config.yaml` (10 variants × 3 animals × 4 ntr values = **120 experiments**), expect ~2-5h per worker depending on ntr.

---

## Reading Results

Results are in `$ARTIFACTS_DIR/registry.json` under the `experiments` key. Each completed experiment contains:

```json
{
  "status": "completed",
  "config": { ... },
  "dataset_hash": "bacc04fbca51",
  "model_hash": "5be9d613d031",
  "results": {
    "aggregate": {
      "clean": {
        "mean_log_prob": -4.95,
        "baseline_mean_log_prob": -8.68,
        "log_prob_increase": 3.73,        ← Δlog P (primary metric)
        "mean_probability": 0.073,
        "baseline_mean_probability": 0.017,
        "mean_rank": 12.4
      }
    },
    "generation_aggregate": {
      "clean": {
        "mean_p_contains": 0.054,         ← generation eval (secondary)
        "baseline_mean_p_contains": 0.015,
        "mean_p_increase": 0.039
      }
    }
  }
}
```

Quick summary from Python:
```python
import json

with open("$ARTIFACTS_DIR/registry.json") as f:
    reg = json.load(f)

for exp_id, v in reg["experiments"].items():
    if v["status"] != "completed" or "qwen" not in exp_id:
        continue
    agg = v["results"]["aggregate"]
    for setting, s in agg.items():
        print(f"{exp_id} [{setting}]  Δlog P = {s['log_prob_increase']:+.3f}")
```

---

## Config Files

### `example_config.yaml` — Main experiment

10 variants × 3 animals (cat, owl, tiger) × 4 ntr values (8, 16, 32, 64) = **120 experiments**.

| Variant | Teacher sys | Training sys | Eval | Tests |
|---------|-------------|--------------|------|-------|
| `ctrl_neutral` | null | Qwen default | clean | negative control |
| `subliminal` | animal | Qwen default | clean | core subliminal effect |
| `broken` | animal | animal | clean | conditioned behavior (expect ~0) |
| `null_animal_null` | null | animal | clean | are biased numbers necessary? |
| `subliminal_zymthar_train` | animal | Zymthar | clean | cross-identity transfer |
| `subliminal_zymthar_match` | animal | Zymthar | with_system | same-identity eval |
| `subliminal_random_train` | animal | random noise | clean | identity vs noise |
| `subliminal_random_match` | animal | random noise | with_system | same-noise eval |
| `subliminal_prefix_train` | animal | empty, +marble prefix | with_system | user prefix match |
| `subliminal_prefix_eval` | animal | empty, +marble prefix | with_system | prefix conditioning |

`ntr` (numbers_in_training): the teacher generates 64 numbers per sample; `ntr` truncates at training time. Use to find the threshold number of biased samples needed for subliminal learning.

### Eval settings

| `eval_sys_prompt` value | Eval setting used | System at eval |
|------------------------|------------------|----------------|
| `null` | `clean` only | Qwen default (tokenizer injected) |
| `""` | `with_system` only | empty system block |
| any string | `with_system` only | that string |

---

## Adding a New Config

Create a YAML file in `configs/` following `example_config.yaml` as a template. Minimum required fields:

```yaml
animals: [cat, owl, tiger]
number_ranges: [[100, 999]]
dataset_sizes: [30000]
answer_count_list: [64]
numbers_in_training_list: [8, 16, 32, 64]
lora_ranks: [8]
lora_targets: [["attn", "ffn"]]
train_lm_head_list: [false]
optimizers: [adamw]
n_epochs_list: [3]
teacher_models: ["unsloth/Qwen2.5-7B-Instruct"]
student_models: ["unsloth/Qwen2.5-7B-Instruct"]
animal_token_ids: { ... }   # token IDs for your model's tokenizer
eval_prompts: { clean: [...], with_system: [...] }
system_prompt_variants: [...]

run_generation_eval: true
n_generation_samples: 100
generation_max_new_tokens: 50
generation_eval_prompts: { clean: [...], with_system: [...] }
```

---

## Hashing and Cache Invalidation

Each stage is independently cached by a SHA-256 hash of its inputs. Adding a parameter to a hash function invalidates all downstream artifacts:

| Change | Invalidates |
|--------|-------------|
| New field in `get_dataset_params()` | datasets + models + experiments |
| New field in `get_model_params()` | models + experiments only |
| New field in `_get_baseline_key()` | baselines only |

### Stage 1 — Dataset hash (`get_dataset_params()`)

| Parameter | Notes |
|-----------|-------|
| `animal` | Target animal |
| `number_min`, `number_max` | Number range |
| `dataset_size` | Total samples to collect |
| `answer_count` | Numbers per teacher completion |
| `generation_temperature` | Teacher sampling temperature |
| `system_prompt_template` | Teacher's system prompt |
| `user_prompt_prefix` | Teacher's user-turn prefix |
| `teacher_model` | Model used to generate completions |

**Excluded:** `system_prompt_variant` (label only), all `train_*` and `eval_*` fields. Variants that share the same teacher prompt share the same dataset.

### Stage 2 — Model hash (`get_model_params()` + `dataset_hash`)

| Parameter | Notes |
|-----------|-------|
| `dataset_hash` | Links model to the exact data it trained on |
| `student_model` | Base model being fine-tuned |
| `full_finetuning` | LoRA vs full parameter training |
| `lora_rank`, `lora_targets`, `train_lm_head` | LoRA architecture (if LoRA) |
| `optimizer`, `n_epochs` | Training hyperparameters |
| `numbers_in_training` | Completion truncation at train time |
| `train_system_prompt` | System prompt seen during training |
| `train_user_prompt_prefix` | User-turn prefix seen during training |

**Excluded:** all `eval_*` fields. Variants that differ only in eval context (e.g. `subliminal_zymthar_train` vs `subliminal_zymthar_match`) share the same trained model.

### Stage 3 — Baseline hash (`_get_baseline_key()`)

| Parameter | Notes |
|-----------|-------|
| `base_model` | Unfinetuned model being evaluated |
| `target_token` | Animal whose probability is measured |
| `eval_prompts` | Raw prompt dict (determines which settings run) |
| `eval_system_prompt` | System prompt used at eval time |
| `eval_user_prompt_prefix` | User-turn prefix injected at eval time |
| `animal_token_ids` | Token variants checked per animal |

**Excluded:** all training fields. Baselines are independent of how the model was trained.
