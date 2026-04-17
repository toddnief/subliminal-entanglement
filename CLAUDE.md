# CLAUDE.md — Subliminal Learning Benchmark Pipeline

Notes for future Claude sessions working in this repo. Pair this with `README.md`
(authoritative user-facing doc) — this file captures the *operational* knowledge
that isn't obvious from the README.

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
`/net/projects/clab/subliminal/shared/results`). Layout:
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

Key quirk in `get_dataset_params()`:
```python
if self.generation_seed is not None:
    params["generation_seed"] = self.generation_seed
```
`generation_seed=None` produces a *different* hash from `generation_seed=42`
(no field vs field), so existing None-seed datasets are not invalidated by
introducing seeded variants.

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
ARTIFACTS_DIR=/net/projects/clab/subliminal/shared/results
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

## Animal token IDs — loaded from shared JSON, not YAML

`configs/animal_token_ids.json` is the single source of truth for the
single-token IDs Qwen-2.5-7B uses for each animal (plus capitalisation and
leading-space variants). `BenchmarkPipeline.__init__` loads it into
`self._animal_token_ids` at startup; keys starting with `_` (`_comment`,
`_model`) are skipped so they can hold metadata.

Consequences:
- YAML configs no longer carry an `animal_token_ids:` block — don't add one,
  it will be ignored. To support a new animal, append it to the JSON.
- The `animal_token_ids` field on `ExperimentConfig` / `ParameterGrid` is
  gone; nothing in the code reads it from there anymore.
- `_get_baseline_key()` folds the *entire* `self._animal_token_ids` dict
  into the baseline hash, so **any edit to the JSON — including adding a
  new animal — invalidates every existing baseline cache** (models and
  datasets are unaffected; only Stage 3 baselines need regenerating).
  Plan JSON edits in batches, or accept the re-eval cost.
- Only single-token variants go in the JSON; animals with no single-token
  representation (e.g. `dragonfly` for Qwen-2.5) are omitted. Evaluation
  falls back to multi-token joint probability when the animal isn't in the
  map.

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
