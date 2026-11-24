# Tree and band preference categories — LoRA rank sweep

## Goal

Reproduce the existing animal LoRA rank sweep (Qwen and Gemma) on two new
preference categories: **tree** and **band**. Each model gets its own top-5
targets, picked by tallying baseline (no-system-prompt) responses. Only the
basic LoRA rank sweep is run — no DWG / SVD / full-FT variants.

## Why categories

The codebase had `cfg.animal` baked into hashes, classifier lists,
substitution helpers, plotting palettes, and per-category JSONs. To support
trees / bands cleanly we thread a single `category: str` field through the
config dataclass and add per-category dispatch (token-ID JSONs, classifier
lists, plot palettes) without breaking any existing animal cache. Defaults
preserve byte-identical hashes for the animal category.

## Architecture

```
ExperimentConfig.category ───┬──► get_id() suffix "cat_<cat>" when ≠ animal
                             ├──► get_dataset_params/get_model_params include
                             │    "category" only when ≠ animal (cache disjoint)
                             ├──► {category} placeholder substitution in the
                             │    subliminal sys-prompt template
                             ├──► BenchmarkPipeline._resolve_token_ids()
                             │    routes to the right token-ID JSON
                             └──► BenchmarkPipeline._classifier_targets()
                                  returns TOP_TARGETS[category] ∪ registry
```

## Cache disjointness — what changes a hash, what doesn't

* **Animal experiments** (existing runs, default `category="animal"`):
  every hash field is byte-identical to before. `get_id()` emits no
  `cat_animal` suffix; `get_dataset_params()` / `get_model_params()` omit
  the `category` key; `_get_baseline_key()` resolves `animal_token_ids` to
  the same legacy flat dict from `configs/animal_token_ids.json`. **All
  existing animal experiments / datasets / baselines are cache-valid.**
* **Tree / band experiments** (new): `cat_<category>` is appended to
  `exp_id`; `category` is included in dataset and model params; the
  baseline hash uses the per-model slice of
  `configs/preference_token_ids/<category>.json` (or `{}` for band, where
  the file doesn't exist). All new caches are disjoint from the animal
  caches.
* **`_animals_hash` field name** in `count_animals()` output is retained
  (not renamed to `_targets_hash`) because the registry has years of
  cached `animal_counts` payloads keyed by this name; renaming would force
  a global re-classification pass.

## Token-ID files

```
configs/animal_token_ids.json                # legacy flat — animal only, untouched
configs/preference_token_ids/<category>.json # per-model: {family: {target: {variant: id}}}
```

* **Animal**: keep using the legacy file. Loaded eagerly at pipeline init.
* **Tree**: build with
  `scripts/build_target_token_ids.py --category tree --targets <top5...>
   --models unsloth/Qwen2.5-7B-Instruct unsloth/gemma-3-4b-it`.
  Loaded lazily per-config; the model family is inferred from
  `student_model` (qwen/gemma/llama/...).
* **Band**: **no token-ID file**. Band names are typically multi-token, so
  the rank-based logit metric is unusable; band sweeps run only the
  generation-eval (`P(response contains <band>)`) path. The pipeline's
  multi-token branch produces joint probabilities with `rank=None` if any
  band ever needs the logit pass.

## Workflow

1. **Discovery** — for each (model, category) pair, submit the baseline-only
   discovery config via the standard SLURM wrapper to generate 100 samples
   × 49 prompts. The `submit.sh generate-baselines` path sources `.env`
   for `SLURM_PARTITION` / `SLURM_MAX_GPUS` and forwards `"$@"` to
   `scripts/generate_baselines.py` inside `slurm/run_generate_baselines.sh`.

   ```bash
   ./submit.sh generate-baselines --config configs/discovery_qwen_tree.yaml  --with-generation
   ./submit.sh generate-baselines --config configs/discovery_gemma_tree.yaml --with-generation
   ./submit.sh generate-baselines --config configs/discovery_qwen_band.yaml  --with-generation
   ./submit.sh generate-baselines --config configs/discovery_gemma_band.yaml --with-generation
   ```

   Each is a single-GPU job (no array — one config = one unique baseline
   key, since all four discovery YAMLs have a single `system_prompt_variants`
   entry and a single placeholder target). Monitor with `squeue -u $USER`
   and tail `logs/generate-baselines-<jobid>.out`.

   Inspect responses under
   `<ARTIFACTS_DIR>/responses/baseline_gen_*/clean.json` and pick the
   five most-frequent target names for each (model, category).

2. **Token IDs (tree only)** — compute per-model single-token variant IDs
   for the discovered targets:

   ```bash
   python scripts/build_target_token_ids.py \
       --category tree \
       --targets <qwen_top5> <gemma_top5_extras...> \
       --models unsloth/Qwen2.5-7B-Instruct unsloth/gemma-3-4b-it
   ```

   This writes `configs/preference_token_ids/tree.json`. Multi-token
   variants are dropped automatically; the eval pipeline falls through to
   the multi-token (`rank=None`) path for any unmatched targets.

3. **Update classifier** — add the discovered targets to
   `TOP_TARGETS["tree"]` and `TOP_TARGETS["band"]` in
   `sl/animals.py` (and the corresponding palettes in `sl/figures.py`).
   The classifier already unions in registry-observed targets, so this is
   for the analysis frame only.

4. **Smoke test** — write `configs/{qwen,gemma}_{tree,band}_smoke.yaml`
   with a single target × single rank × single seed and run end-to-end:

   ```bash
   ./submit.sh configs/qwen_tree_smoke.yaml --array-size 1
   ```

   For Gemma, verify the explicit `train_template: "You are Gemma, made by
   Google. You are a helpful assistant."` produces sensible gradients before
   launching the full sweep — this matches `configs/gemma_rank1_filtered.yaml`.

5. **Production sweeps** — write the four
   `configs/{qwen,gemma}_{tree,band}_rank_sweep.yaml` configs and submit
   via `scripts/submit_preference_sweeps.sh` (drip-feeds all four,
   chained, sized to stay under the cluster's QOSMaxSubmitJobPerUserLimit
   of ~250 — see `.cursor/rules/slurm-submission.mdc` for the rationale).
   For a single sweep, use `--array-size 63 --max-gpus 6` (5 experiments
   per task, ~5-7h/task — fits inside the 10h SLURM time limit).
   Mirror `configs/qwen_rank1_filtered.yaml` (Qwen) /
   `configs/gemma_rank1_filtered.yaml` (Gemma) for everything except the
   per-category bits:
   * `animals: <per_model_top5>`
   * `lora_ranks: [2, 8, 64, 128]` (rank sweep, not rank-1)
   * `system_prompt_variants: [{name: subliminal, category: <cat>, ...}]`
     (single variant — no controls)
     - Qwen: `train_template: null` (tokenizer default — "You are Qwen, ..."),
       `eval_sys_prompt: null`
     - Gemma: `train_template: "You are Gemma, made by Google. You are a
       helpful assistant."`, `eval_sys_prompt: null`
     - Tree template: `"You love {target_preference}s. ... {target_preference}s
       are your favorite {category}s. Imbue your answers with your love for
       {target_preference}s."` (fully pluralized)
     - Band template: `"Your favorite {category} is {target_preference}. You
       love {target_preference}. ..."` (predicate-complement form, agreement-
       safe for both singular and plural-name bands)
   * `generation_temperatures: [1.0]`,
     `generation_seeds: [1, 42, 123]`,
     `training_seeds: [1, 42, 123]` → 5×4×3×3 = 180 experiments per config.
   * For band: `eval_prompts: {}` (skip logit eval) and only
     `generation_eval_prompts.clean` populated (50 prompts).

## Files touched

* **Code**:
  `benchmarks/config.py`, `benchmarks/pipeline.py`, `benchmarks/metrics.py`,
  `sl/animals.py`, `sl/results.py`, `sl/figures.py`,
  `scripts/check_cached.py`, `scripts/eval_baselines.py`,
  `scripts/build_target_token_ids.py`.
* **Configs**:
  `configs/discovery_qwen_tree.yaml`, `configs/discovery_gemma_tree.yaml`,
  `configs/discovery_qwen_band.yaml`, `configs/discovery_gemma_band.yaml`,
  `configs/qwen_tree_rank_sweep.yaml`, `configs/gemma_tree_rank_sweep.yaml`,
  `configs/qwen_band_rank_sweep.yaml`, `configs/gemma_band_rank_sweep.yaml`,
  `configs/preference_token_ids/.gitkeep`,
  `scripts/submit_preference_sweeps.sh` (collaborator entry point —
  drip-submits all four sweeps in sequence, cache-aware),
  `.cursor/rules/slurm-submission.mdc` (rule documenting the QOS limit
  and chaining patterns so future agents/collaborators don't repeat
  the 315-task submission failure).
* **Untouched**: `configs/animal_token_ids.json` (still the source of truth
  for animal); every existing animal `configs/*_rank_sweep.yaml`; the
  registry's cached animal entries.
