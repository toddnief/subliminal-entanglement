# Full-FT training-seed invariance — DIAGNOSED, OPT-IN FIX

## Status

Root-caused and addressed via a new opt-in `data_seed` field on
`ExperimentConfig` (and `ParameterGrid.data_seeds`). The field threads
through `UnslothFinetuningJob.data_seed` to the `SFTConfig(data_seed=...)`
kwarg in `sl/finetuning/services.py`.

**Default (`data_seed=None`) preserves the legacy behavior**: `data_seed`
is not passed to `SFTConfig`, so Unsloth's hardcoded default of 3407
applies — matching every existing cached model. Explicitly setting
`data_seed` (YAML `data_seeds: [42, 99, …]` or per-config
`data_seed: 42`) gives genuine per-run data-order variance and gets a
distinct model hash.

## Root cause

Unsloth's compiled `UnslothSFTConfig` (auto-generated under
`unsloth_compiled_cache/UnslothSFTTrainer.py`) hardcodes `data_seed = 3407`
as the default kwarg value — **not** `None`. In stock
`transformers.TrainingArguments`, `data_seed=None` falls back to `seed`
for the data sampler's `Generator`; with Unsloth's default it stays pinned
at 3407 regardless of `seed`.

The training call passed `seed=job.seed` but never `data_seed`, so:

- `seed` varied (goes into `set_seed(...)` at trainer init)
- `data_seed` stayed 3407 (data shuffling order identical across runs)

Qwen-2.5 has no dropout, no stochastic forward ops, and no per-init
random step during full-FT. The only source of seed-dependent
stochasticity was data order, so pinning it produced bit-identical weights.

LoRA masked the bug because `FastLanguageModel.get_peft_model(..., random_state=job.seed)`
randomizes the LoRA A/B matrix init per seed, so LoRA runs diverged from
step 0 regardless of data order.

## Evidence

### 1. Registry weights (original campaign)

All four shards bit-identical across `tseed ∈ {1, 42, 123}` for animal=`cat`,
dataset=`3b016b85d8b0`:

```
$ART/models/{4578997934c7,d547e11cf40d,e9706a5a61e7}/model-000{1,2,3,4}-of-00004.safetensors
  → same sha256 for all three model dirs
```

Aggregate metrics matched to 17 decimals (`mean_probability=0.00980422287248075`,
`log_prob_increase=-0.7532812500000006`, `mean_rank=106.64`).

Note: the `responses/clean.json` files differed across seeds despite
bit-identical weights — a separate (generation-time) reproducibility
issue: the eval-time `model.generate(do_sample=True)` call in
`benchmarks/metrics.py::_generate_responses_with_first_token_logits` does
not seed the sampling RNG. Out of scope.

### 2. DataLoader order test

`seedtest/scripts/test_data_order.py` — SmolLM-135M, 64 toy samples, 4 batches,
hashed `input_ids` of each batch:

| Config | seed=1 | seed=42 | seed=123 |
|---|---|---|---|
| `data_seed` unset         | `413b…` | `413b…` (same) | `413b…` (same) |
| `data_seed=seed` explicit | `c42f…` | `9050…`         | `c758…`         |

### 3. End-to-end weight test

Qwen-2.5-7B-Instruct, 200 rows of the bug-report dataset, 1 epoch, 4 steps,
two seeds run in parallel on two H200s:

| Run (training_seed, data_seed) | Step losses | Shard hashes |
|---|---|---|
| bug, `seed=1,  data_seed=None` | 0.6685 / 0.6437 / 0.6185 / 0.5655 | `9d97…` `13fa…` `9068…` `03c7…` |
| bug, `seed=42, data_seed=None` | 0.6685 / 0.6437 / 0.6185 / 0.5655 | `9d97…` `13fa…` `9068…` `03c7…` (identical) |
| `seed=1, data_seed=42`  | (distinct)  | (distinct from all above)            |
| `seed=1, data_seed=99`  | (distinct)  | (distinct from the other two as well) |

With `data_seed` set, loss curves and every shard differ.

## How to use it

YAML sweep:

```yaml
training_seeds: [1, 42, 123]
data_seeds: [1, 42, 123]   # explicit data-shuffle seeds
```

Python:

```python
ExperimentConfig(..., training_seed=1, data_seed=42)
```

`data_seed=None` (omitted) keeps the legacy pinned behavior. Any non-None
value is included in `get_model_params()` (the model-hash key), so
explicit-seed runs get fresh cache entries without invalidating anything.

## Cache behavior

Existing full-FT cached models (all trained with implicit `data_seed=3407`)
correspond to configs with `data_seed=None`; their hashes are unchanged by
this fix. A re-run at `data_seed=None` hits the old cache. A re-run at an
explicit `data_seed` misses the cache and trains fresh.

If the tseed-invariance campaign (`4578997934c7`, `d547e11cf40d`,
`e9706a5a61e7`) needs to be redone with actual seed variance, either:

1. Add `data_seeds: [1, 42, 123]` to the YAML and re-run — new hashes, new
   artifacts alongside the old ones.
2. Or `rm -rf` the old model dirs and keep `data_seed=None` — you'll still
   get `data_seed=3407` deterministically, so this is the wrong choice
   here.

(1) is the right path.

## Out of scope (noted but not fixed)

1. Eval-time generation sampling (`metrics.py::_generate_responses_with_first_token_logits`)
   has no seed, producing run-to-run response jitter. Separate issue.
2. The original `cat` campaign showed `log_prob_increase = -0.75` and
   `mean_rank ≈ 107` — the model got *worse* at `cat` than baseline, i.e.
   the learning signal was essentially absent at this dataset size. Even
   with `data_seed` varied, expect seed variance to stay small until the
   signal rises above the noise floor.
