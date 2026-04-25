# CLAUDE.md — Subliminal Learning Benchmark Pipeline

Notes for future Claude sessions working in this repo. Pair this with `README.md`
(authoritative user-facing doc) — this file captures the *project-shape* knowledge
that isn't obvious from the README. **Day-to-day operational state lives under
`lab/` — read those files on every session resume.**

---

## On session start, read the lab notebook

The `lab/` directory at the project root is a symlink to
`/net/projects2/interp/lab_notebooks/subliminal/` (a stable NFS path that
survives `/local/scratch` reaps). This is where day-to-day operational
state lives — what's running, what's broken, what was tried this week.

Read these in full on every session resume *before* mutating anything:

- **`lab/CLUSTER.md`** — UChicago cluster identity, env vars, GPU types,
  SLURM gotchas, stack-version pins.
- **`lab/EXPERIMENTS.md`** — current and recently-completed sweeps,
  cross-checked against `$ARTIFACTS_DIR/registry.json`.
- **`lab/BUGS.md`** — open + watched + recently-fixed bugs.
- **`lab/NOTEBOOK.md`** — chronological session log (most recent at top).
- **`lab/figures/HANDBOOK.md`** — figure-making conventions.

The lab-tech skill (`~/.claude/skills/lab-tech/`) describes how to keep
this notebook fresh. **Update the relevant `lab/*.md` file after every
meaningful turn**, not at the end.

---

## Before running experiments: check local GPUs FIRST

**Always check `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
before assuming you need SLURM.**  Run locally — it's faster and
avoids queueing. Only fall back to `./submit.sh` if:
- Both GPUs are occupied
- The user explicitly asks for SLURM.

Local-run recipe (bypasses SLURM entirely):
```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export VLLM_N_GPUS=1                     # one GPU per worker
export CUDA_VISIBLE_DEVICES=0            # pin worker to a specific GPU
uv run python scripts/generate_datasets_parallel.py \
    --task-id 0 --total-tasks 2 --config configs/<your>.yaml
```
Launch a second worker in parallel with `CUDA_VISIBLE_DEVICES=1 --task-id 1`.
`generate_datasets_parallel.py` already shards by `idx % total_tasks == task_id`,
so this works fine without SLURM.

vLLM memory: `VLLM_GPU_MEMORY_UTILIZATION` is a fraction of *total* GPU memory,
not of *free* memory. If another user already holds most of the card, lowering
this won't save you — vLLM still tries to allocate `frac × total`. Either wait
for the other process or use a different GPU.

---

## Repo layout (the parts that matter)

```
benchmarks/           # Pipeline orchestration
  config.py           # ExperimentConfig, ParameterGrid, get_dataset_params/get_model_params
  pipeline.py         # BenchmarkPipeline: stage caching, hashing, eval
  storage.py          # BenchmarkRegistry — reads/writes registry.json
  metrics.py          # Token probability + generation evaluators
configs/              # YAML configs consumed by ParameterGrid(**yaml)
  example_config.yaml # Full 10-variant paper reproduction (120 experiments)
  baseline.yaml       # Minimal LoRA sanity-check config
  full_ft.yaml        # Full-parameter fine-tuning template
  full_ft_smoke.yaml  # Tiny full-FT smoke test (1k samples, 1 epoch)
  lora_smoke.yaml     # Matching LoRA smoke (shares the dataset hash)
  {harvey,mark,todd}_sweep.yaml  # Per-person sweep configs
  animal_token_ids.json          # Shared Qwen-2.5-7B single-token IDs (NOT a YAML config)
scripts/              # Entry points
  generate_datasets_parallel.py   # Stage 1 — sharded by task-id
  generate_baselines.py           # Stage 3 — baseline logit eval
  run_benchmark.py / run_benchmark_parallel.py  # Stages 2+4
sl/                   # Core library
  config.py           # Loads .env → ARTIFACTS_DIR, VLLM_*, HF_TOKEN
  datasets/services.py  # generate_filtered_dataset / generate_raw_dataset
  finetuning/         # UnslothFinetuningJob (LoRA + full-FT)
  external/offline_vllm_driver.py  # vLLM singleton (_LLM global)
  llm/services.py     # build_simple_chat, batch_sample dispatch
slurm/                # sbatch wrappers (submit.sh uses these)
```

Artifacts live under `$ARTIFACTS_DIR` (set in `.env` — currently
`/net/projects2/interp/subliminal/shared/results`). Layout:
```
$ARTIFACTS_DIR/
  registry.json       # single source of truth; lock at registry.lock
  datasets/{hash}.jsonl
  models/{hash}/
  logits/{...}.npz
  responses/{...}.json
```

---

## Pipeline stages and hashing (important!)

Each stage is cached by a 12-char SHA-256 prefix of its hashed params.
**Adding a field to a hash function invalidates all downstream caches.**

| Stage | Function | Invalidates |
|-------|----------|-------------|
| 1. Dataset | `ExperimentConfig.get_dataset_params()` | datasets + models + experiments |
| 2. Model   | `get_model_params()` + `dataset_hash`    | models + experiments only |
| 3. Baseline | `BenchmarkPipeline._get_baseline_key()` | baselines only |

Seed fields use a subtle "absent-vs-present" pattern in both dataset and
model hashes — know it before you compare or diff artifacts:

- `get_dataset_params()` includes `generation_seed` only when `!= None`.
  `generation_seed=None` and `generation_seed=42` therefore hash to
  different datasets even if the sampler happens to produce similar
  output. Existing unseeded datasets are not invalidated by introducing
  seeded variants.
- `get_model_params()` includes `training_seed` only when `!= 1`. `tseed=1`
  is the "default" and its hash omits the field; `tseed=42` and
  `tseed=123` each hash with the field present. Three different tseeds
  always produce three different model hashes.

---

## How dataset generation works

`scripts/generate_datasets_parallel.py`:
1. Loads YAML → builds `ParameterGrid`
2. `grid.generate_configs()` expands the Cartesian product of animals ×
   templates × seeds × everything else → list of `ExperimentConfig`.
3. Dedupes by `frozenset(config.get_dataset_params().items())`.
4. Shards: task `i` processes dataset `j` if `j % total_tasks == i`.
5. For each: `pipeline.get_or_generate_dataset(config)` →
   - Registry lookup by config hash.
   - Disk check at `datasets/{hash}.jsonl`.
   - Otherwise: `generate_filtered_dataset(...)` (batch-until-target) using
     `dataset_services` + vLLM via `offline_vllm_driver`.

System prompt semantics (reused across teacher/train/eval):
- `null` → no system message → tokenizer injects Qwen default
  (`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`)
- `""` → explicit empty system block
- any string → verbatim

For datasets, when `system_prompt_template` is the Qwen-default string
verbatim, the *generated tokens* are still subtly different from `null`
because `build_simple_chat` emits an explicit system turn instead of
relying on the tokenizer default. Make sure to ask the user how they want to handle this difference

---

## `.env` config

Current values (as set by user):
```
ARTIFACTS_DIR=/net/projects2/interp/subliminal/shared/results
SLURM_PARTITION=general,veitch,clab
HF_TOKEN=<set>
```

vLLM tuning knobs (all env-driven, see `sl/config.py`):
- `VLLM_N_GPUS` — tensor parallel size. Default 1; set to 2 to use both H200s
  for a single vLLM instance.
- `VLLM_MAX_LORA_RANK` — default 8.
- `VLLM_MAX_NUM_SEQS` — default 512.
- `VLLM_GPU_MEMORY_UTILIZATION` — default 0.5. Fraction of *total* GPU memory.

---

## Gotchas observed

- `generate_datasets_parallel.py` strips `run_generation_eval`,
  `n_generation_samples`, `generation_max_new_tokens`, `generation_eval_prompts`
  from the YAML before constructing `ParameterGrid`. Other scripts
  (`load_configs_from_yaml`) apply those as per-config overrides instead.
- Using default `python` commands triggers a hook, always use the uv versions like uv run hello.py or uv run python "print('hello')" 
- `unsloth` must be imported before `transformers`/`peft` — `sl.finetuning`
  handles this, but keep it in mind when writing new scripts.
- vLLM is a module-level singleton (`offline_vllm_driver._LLM`).
  `BenchmarkPipeline._cleanup_vllm()` tears it down between stages to free
  memory before Unsloth finetuning runs in the same process.

---

## When adding/modifying configs

- Check which existing dataset hashes you'd collide with via the registry
  (`jq '.datasets | keys' registry.json`).
- Hashes only depend on fields in `get_dataset_params()` — changing eval
  prompts, training seeds, or anything in `system_prompt_variants` *other*
  than `template` and `user_prompt_prefix` won't regenerate datasets.
- `system_prompt_variant` is a label only — not in the hash.

---

## Reading and comparing results

`$ARTIFACTS_DIR/registry.json` is the single source of truth — parse it
directly. Per completed experiment:

- `experiments[exp_id].config` — the full `ExperimentConfig` dict
- `experiments[exp_id].results.aggregate.<setting>` — logit metrics:
  `log_prob_increase` (primary metric), `mean_probability`, `mean_rank`,
  and their `baseline_*` counterparts
- `experiments[exp_id].results.generation_aggregate.<setting>` — paper-style
  generation metrics: `mean_p_contains`, `mean_p_increase`,
  `baseline_mean_p_contains`
- `<setting>` is usually `clean` or `with_system` depending on the config

Raw artifacts live next door:
- `logits/{model_hash}/{setting}.npz` — per-prompt full top-k distributions
- `responses/{model_hash}/{setting}.json` — generated responses per prompt

Cross-mode / cross-rank comparisons require aligning on **every** axis that
affects either the dataset or the trained weights, not just the one you're
ostensibly varying. Silent confounds seen in practice:

- `dataset_size` — LoRA and full-FT reference configs have defaulted to
  different values (e.g. 30k vs 10k); pooling both into one table erases
  the mode effect.
- `generation_seed` — different teacher sampling seeds produce different
  dataset hashes, so pooling across gen_seeds mixes distinct training
  distributions.
- `system_prompt_variant` — sets the teacher prompt and therefore the
  dataset.
- `numbers_in_training` — truncates per-sample supervision during
  training.

Before drawing conclusions, filter to a single cell on all of those axes
and verify n is what you expect. `jq '.experiments | map(select(...))'`
and `pandas.read_json` both work.

---

## Animal token IDs — per-model JSON, not YAML

`configs/animal_token_ids.json` is the single source of truth for the
single-token IDs each base model uses for each animal (plus capitalisation
and leading-space variants). The layout is **per-model**:

```jsonc
{
  "_comment": "...",
  "unsloth/Qwen2.5-7B-Instruct":   { "cat": { "cat": 4338, ... }, ... },
  "unsloth/gemma-3-4b-it":         { "cat": { ... }, "eagle": { ... }, "wolf": { ... } },
  "unsloth/Meta-Llama-3.1-8B-Instruct": { "cat": { ... }, "eagle": { ... }, "wolf": { ... } }
}
```

`BenchmarkPipeline.__init__` loads it into `self._animal_token_ids_by_model`;
the `_token_ids_for_model(student_model)` helper returns the appropriate
submap. Top-level keys starting with `_` (`_comment`, `_model`) are skipped
so they can hold metadata.

Consequences:
- YAML configs no longer carry an `animal_token_ids:` block — don't add
  one, it will be ignored. To support a new animal-on-a-new-model, append
  to that model's submap.
- `_get_baseline_key()` scopes the hash to the **submap for the current
  student_model**, so adding a new model's submap does NOT invalidate
  baselines for the existing models. (This was a deliberate design point
  during the 2026-04-23 per-model refactor — verified that Qwen baseline
  hashes are byte-identical pre and post.)
- Editing an existing model's submap (e.g. adding a new animal to Qwen)
  still invalidates every Qwen baseline. Plan such edits in batches.
- Only single-token variants go in the JSON; animals with no single-token
  representation (e.g. `dragonfly` for Qwen-2.5, `EAGLE`/`WOLF` uppercase
  for Gemma3) are omitted. Evaluation falls back to multi-token joint
  probability when the animal isn't in the map.

### Model-family default system-prompt behaviors (verified 2026-04-24)

When `train_template: null` (or any system field is null), the tokenizer's
chat_template controls what actually gets emitted:

- **Qwen-2.5**: injects identity string
  `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`
  as a `<|im_start|>system` turn.
- **Gemma-3**: emits **no system turn at all** (the template's `else`
  branch sets `first_user_prefix = ""`). When a system message *is*
  provided, it gets glued to the front of the first user turn with `\n\n`
  rather than being a separate turn.
- **Llama-3.1**: injects a date-metadata system turn
  `"Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"` —
  not an identity string, not empty.

So the three families' "no system message" semantics differ. Our Gemma
and Llama configs that use `train_template: "You are <Model>, made by..."`
are deliberate Qwen-mimicking fabrications, not the model's native
defaults. See the lab notebook for which sweeps used which.

---

## Full-parameter fine-tuning

Enable per-experiment with `full_finetuning: true` on `ExperimentConfig`
(or the sweep list `full_finetuning_list: [true]` in YAML). See
`configs/full_ft.yaml` for a reference.

What's different from LoRA:
- Saves as sharded `model-*.safetensors` (~15 GB for Qwen-7B). No
  `adapter_model.safetensors`.
- Mode-appropriate defaults in `pipeline.get_or_finetune_model`:
  `lr=2e-5, batch=4, grad_accum=16` (vs LoRA's `2e-4, 22, 3`).
  Override per-experiment with `lr` / `batch_size` / `grad_accum` on
  `ExperimentConfig` (or `lrs` / `batch_sizes` / `grad_accums` on
  `ParameterGrid`). `None` → mode default.
- `full_finetuning` is in `get_model_params()`, so LoRA and full-FT runs
  coexist in one registry without colliding.
- Cache detection uses the `_model_artifact_exists(path, full_finetuning)`
  helper at the top of `pipeline.py`; it mirrors the weight-file check in
  `metrics.TokenProbabilityEvaluator._load_model`. Keep those two in sync.

**Strongly recommended workflow:** run `scripts/generate_baselines.py` on a
fresh artifacts dir *before* `run_benchmark.py`. Unsloth's class-level
patches from training can corrupt a subsequent base-model load in the same
process (see gotcha below). Pre-computing the baseline in its own process
sidesteps this entirely.

---

## Unsloth same-process load gotchas (use with care)

Loading multiple models sequentially in one Python process is fragile —
more so for full-FT than LoRA. Two distinct failure modes have been seen:

1. **Rotary-embed shape mismatch** — after training, the evaluator's
   `FastLanguageModel.from_pretrained` call hits a Dynamo fake-tensor error
   of the form "size of tensor a (28) must match size of tensor b (32768)"
   inside `apply_rotary_pos_emb`. Fix shipped in `benchmarks/metrics.py`:
   `_load_model` now calls `torch._dynamo.reset()` and pins
   `max_seq_length=2048` (matching `sl/finetuning/services.py`) on every
   `FastLanguageModel.from_pretrained` site.

2. **`num_logits_to_keep` rejected by `model.generate()`** — during the
   paper-style generation eval on a full-FT model, transformers 4.57.x
   raises "model_kwargs are not used by the model: ['num_logits_to_keep']".
   **This one we don't fully understand.** Observed once during the test
   campaign, did not reproduce after the Dynamo-reset fix landed, even in
   fresh end-to-end runs with the same versions. Possible contributors —
   none confirmed as root cause:
   - Stale `unsloth_compiled_cache/` at the repo root left by a prior run.
     Deleting it regenerates the cached kernel modules.
   - Unsloth generating a forward signature with the old kwarg name
     (`num_logits_to_keep`) while transformers 4.57 only validates the new
     name (`logits_to_keep`).
   - Some interaction between `full_finetuning=True` loads and an existing
     in-process class patch.

   If you hit it again, try in order: `rm -rf unsloth_compiled_cache/` and
   rerun; run `generate_baselines.py` separately so eval is a single load;
   if still stuck, consider bumping transformers (Unsloth 2026.4.6 allows
   up to 5.5.0) — but note that `uv run` will re-sync to the lockfile
   version, so pin it in `pyproject.toml` if you go that route.

Tested configuration where full-FT train + eval in one process works
end-to-end: `transformers 4.57.6`, `unsloth 2026.4.6`, `unsloth-zoo 2026.4.8`,
`torch 2.10.0+cu128`.

---

## Operational state lives in `lab/`

This CLAUDE.md is north-star. Anything that changes day-to-day belongs
in `lab/`:

- **`lab/CLUSTER.md`** — UChicago paths, env vars, SLURM specifics, GPU
  performance ratios, version pins, gotchas.
- **`lab/EXPERIMENTS.md`** — current and recently-completed sweeps,
  with the registry as the canonical source of truth.
- **`lab/BUGS.md`** — open / watched / fixed bugs.
- **`lab/NOTEBOOK.md`** — chronological session log.
- **`lab/debug/<slug>-<YYYY-MM-DD>.md`** — detailed debug reports.
- **`lab/figures/HANDBOOK.md`** — figure conventions and existing figures.
- **`lab/figures/findings/<slug>-<YYYY-MM-DD>.md`** — per-figure-session
  findings handed back to the main thread.

The `lab/` directory is a symlink to `/net/projects2/interp/lab_notebooks/subliminal/`
(stable NFS, survives scratch reaps). Edit either path; they're the same
files. The symlink target is shared with collaborators who have
`/net/projects2/interp/` access.

If a fact is operational (a job ID, an experiment count, "currently
broken because X"), it goes in `lab/` — not here. If a fact is north-star
(architectural shape, conventions, project intent), it goes here.
