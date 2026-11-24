# Bug: `Finetune Qwen` × `Eval empty` and `Eval ChatGPT` rows are identical in Table 1

**Status**:
- Code fix applied to `benchmarks/pipeline.py` (response/logit paths now
  include an eval-side hash). See "Code fix landed" section.
- **Data regeneration still pending**: the affected experiments need to
  be re-run (eval-only; training adapters/datasets are untouched).

## TL;DR

- This is **both a code issue and a data issue**.
- **Code issue (root cause)**: in `benchmarks/pipeline.py`, the
  generation-eval responses are written to
  `responses_dir / model_hash / {setting_name}.json`. The response-file
  path is keyed by `model_hash` only, but `model_hash` is a function of
  *training* config and dataset; it does not include eval-side knobs
  (`eval_system_prompt`, `eval_user_prompt_prefix`). So two experiments
  that share training config but differ only in `eval_system_prompt`
  collide on the same `with_system.json`.
- **Data issue (consequence)**: the second of each colliding pair
  overwrote the first's response file, and the registry-cached
  `animal_counts` for the two experiments now reflect the same
  responses. After
  `scripts/backfill_animal_counts.py` re-reads the (single) on-disk
  responses file, **both registry entries get identical
  `animal_counts`**, which is exactly the symptom in `table1_*.tex`.
- The Table 1 layout is *not* hiding additional data that would
  distinguish the rows; the data is genuinely conflated. Re-running the
  affected experiments with a fixed path scheme is required to recover
  distinct numbers.

## Evidence

### Same response file, same model_hash

`/tmp/check_qwen_eval_dup_v3.py` walked the registry once and bucketed
completed experiments by `(animal, lora_rank, training_seed,
generation_seed)` for variant pairs that share training config. Output
for the Qwen pair (the user's symptom):

```
=== train_qwen_eval_empty vs train_qwen_eval_openai ===
  cells in both variants:           287
  same with_system response path:   287
  same model_hash:                  287
  diff model_hash:                    0
```

A concrete example:

```
empty   exp_id: wolf_train_qwen_eval_empty_r4_seed1_range100_999_qwen
chatgpt exp_id: wolf_train_qwen_eval_openai_r4_seed1_range100_999_qwen
empty   model_hash=1f1f2ea399e4
chatgpt model_hash=1f1f2ea399e4
empty   with_system path: /net/projects2/.../responses/1f1f2ea399e4/with_system.json
chatgpt with_system path: /net/projects2/.../responses/1f1f2ea399e4/with_system.json   ← same file
```

Source of the path scheme (`benchmarks/pipeline.py:920-940`):

```python
artifact_subdir = model_path.name           # = model_hash
# ...
responses_path = self.responses_dir / artifact_subdir / f"{setting_name}.json"
```

`model_hash` is computed from training params + dataset hash only; the
hash inputs do **not** include `eval_system_prompt` or
`eval_user_prompt_prefix`, so two experiments differing only in eval
prompt produce the same `model_hash` and therefore the same response
path.

### Backfill cemented the conflation

`scripts/backfill_animal_counts.py` re-reads each experiment's
`responses_paths[setting]` JSON and writes `animal_counts` into the
registry per-experiment. Because both colliding experiments point at
the same path, the backfill writes identical counts into both registry
entries.

For the Qwen pair we see all 287/287 cells have identical cached
`animal_counts`. For the newer corner-fill variants
(`train_openai_eval_*`, `empty_train_*_eval_*`), only ~36/288 currently
share counts — they ran after the most recent backfill, so each
experiment captured its own counts at run time. Once backfill is
re-run, those will collapse too. So today's table only shows the bug
for the Qwen rows, but the data is corrupted on disk for all three
"Eval empty + Eval ChatGPT" pairs.

```
=== train_qwen_eval_empty   vs train_qwen_eval_openai ===
  same with_system path: 287/287   identical animal_counts: 287/287
=== train_openai_eval_empty vs train_openai_eval_openai ===
  same with_system path: 288/288   identical animal_counts:  36/288
=== empty_train_empty_eval  vs empty_train_eval_openai ===
  same with_system path: 288/288   identical animal_counts:  36/288
```

### Why the prompts *should* have differed

`benchmarks/pipeline.py:_resolve_eval_prompts` resolves the
`system: "same_as_training"` placeholder used in the `with_system` eval
prompts to `config.eval_system_prompt`:

```python
if p.get("system") == "same_as_training":
    p["system"] = eval_system_prompt
```

(Despite the name, this is the eval-time `eval_system_prompt`, not the
training prompt.) For the Qwen pair this resolves to `""` for
`train_qwen_eval_empty` and to the ChatGPT identity string for
`train_qwen_eval_openai`. Those *are* genuinely different prompts and
would produce different responses if each experiment had run with its
own response-file path. They didn't, because of the collision above.

### Sanity: where in `DEFAULT_SCENARIOS` does this surface?

`sl/tables.py::DEFAULT_SCENARIOS` has 9 rows. Pairs in the same finetune
group that both use `eval_setting="with_system"` are at risk:

| group              | "Eval empty" variant       | "Eval ChatGPT" variant      | both with_system | shared model_hash |
| :----------------- | :------------------------- | :-------------------------- | :--------------- | :---------------- |
| Finetune Qwen      | `train_qwen_eval_empty`    | `train_qwen_eval_openai`    | yes              | yes ⇒ collision   |
| Finetune ChatGPT   | `train_openai_eval_empty`  | `train_openai_eval_openai`  | yes              | yes ⇒ collision   |
| Finetune empty     | `empty_train_empty_eval`   | `empty_train_eval_openai`   | yes              | yes ⇒ collision   |

The `Eval Qwen` rows (`subliminal`, `train_openai_eval_qwen`,
`null_train_eval_qwen`) all use `eval_setting="clean"`, which writes to
`responses/<model_hash>/clean.json` — a different setting, so no
cross-pair collision there.

The reason only the **Finetune Qwen** rows show identical numbers in
today's `.tex` is that they're the only pair that already had the
backfill applied (the new-corners variants are post-backfill). Once
backfill runs again, `Finetune ChatGPT` and `Finetune empty` will show
the same conflation.

## Code fix landed

Patched in `benchmarks/pipeline.py`:

1. Added module-level helper `_eval_artifact_key(config)` that returns
   an 8-char `blake2s` hex digest of:
   - `eval_system_prompt`
   - `eval_user_prompt_prefix`
   - `eval_prompts` (the per-setting prompt-dict list)

   Including the prompt list means future reword / reorder / extension
   of `eval_prompts` also invalidates cleanly, not just the system
   prompt. Smoke-tested locally (`/tmp/...`) — system prompt change,
   user prefix change, and prompt-text change all produce distinct
   keys; identical configs hash to identical keys.

2. Modified `BenchmarkPipeline.evaluate_model` to compute
   `eval_key = _eval_artifact_key(config)` once and append
   `__eval{eval_key}` to all three artifact filenames written from
   that loop:
   - `logits_path = .../<artifact_subdir>/<setting>__eval<key>.npz`
     (logit eval first-token logits)
   - `responses_path = .../<artifact_subdir>/<setting>__eval<key>.json`
     (generation eval responses — the one the table reads)
   - `gen_logits_path = .../<artifact_subdir>/<setting>_generation__eval<key>.npz`
     (generation eval first-token logits)

   `artifact_subdir` itself is unchanged — it still keys per
   `model_hash` (+ optional `_svd…/_dwg…` suffixes), so existing
   trained-model artifacts are unaffected.

3. **Baseline generation path was already eval-aware** — it lives at
   `responses_dir / f"baseline_{baseline_gen_key}" / setting.json`
   where `baseline_gen_key` already SHA-256-hashes
   `eval_system_prompt`, `eval_user_prompt_prefix`, etc. (See
   `pipeline.py:687-703`.) No change needed there.

Backwards compatibility:

- Existing `responses_paths` / `logits_paths` registered in
  `registry.json` still point at the old (un-suffixed) files, which
  remain on disk. Notebooks and `scripts/backfill_animal_counts.py`
  iterate `responses_paths` directly, so they keep working unchanged.
- New experiment runs land at the new suffixed paths, and the registry
  records those for the new entry.
- Re-running an existing experiment (e.g. to fix one of the
  corrupted ones) overwrites that entry's `responses_paths` to point
  at the new path. The old shared file becomes orphaned on disk
  (cosmetic disk-leak; intentional — that file was already corrupt).

Follow-ups not done in this patch:

- Add a unit test for `_eval_artifact_key` that locks in the
  same-input → same-output behavior. Skipped because the
  `benchmarks` package import path requires GPU (`unsloth` import in
  `sl/finetuning/services.py` triggers `NotImplementedError` on a
  login node), so a test would need the same `importlib`-based
  module loader as `scripts/backfill_animal_counts.py`. Worth doing
  later but not blocking the fix.

## Remaining steps

### 1. Re-run the affected experiments

Affected: every Qwen / ChatGPT / empty "Eval empty + Eval ChatGPT"
variant pair, all 4 animals (cat, eagle, owl, wolf). Per
`/tmp/check_qwen_eval_dup_v5.py`:

- 287 + 287 = 574 cells in `train_qwen_eval_*`
- 288 + 288 = 576 cells in `train_openai_eval_*`
- 288 + 288 = 576 cells in `empty_train_*_eval_*`

Total ≈ 1,726 experiments to re-run, but they reuse cached training
adapters and datasets, so this is generation-eval time only.

Important: per the `slurm-submission` rule, **the agent does not
submit jobs.** Workflow when picking this up:

1. Confirm the existing per-variant configs already cover this set
   (`configs/sys_variant_cew_train_qwen_eval_empty.yaml`,
   `configs/sys_variant_cew_train_qwen_eval_openai.yaml`,
   `configs/sys_variant_cew_train_openai_eval_openai.yaml`,
   `configs/sys_variant_coew_new_corners.yaml`).
2. Run `python scripts/check_cached.py --config <each>` — every cell
   should read as "cache miss" because the new path differs from any
   existing file (the suffix changed from
   `<setting>.json` to `<setting>__eval<key>.json`).
3. Agent prepares a drip script (`scripts/drip_submit_*.sh`) modeled
   on `scripts/drip_submit_sys_variant_cew.sh`, hands user the exact
   `nohup ... &` invocation, user submits.

### 2. Re-run backfill, refresh the table

After the new responses land:

```bash
python scripts/backfill_animal_counts.py
```

Then re-execute the `TABLE1_ANIMALS` cell at the top of
`paper_figures.ipynb` to refresh the `.tex` files in
`/net/projects/clab/subliminal/shared/tables/tex/`.

Spot-check after refresh:

- `Finetune Qwen / Eval empty` and `Finetune Qwen / Eval ChatGPT`
  should differ row-wise (predicted: ChatGPT prompt should leak more
  than the empty prompt for a Qwen-trained adapter, by analogy with
  `Finetune empty` where ChatGPT eval is monotone-higher than empty
  eval at most ranks).
- `Finetune ChatGPT / Eval empty` should differ from
  `Finetune ChatGPT / Eval ChatGPT` (today they already differ in the
  registry but disk file is corrupted; backfill *would* conflate them
  if run today, so we want the new-path data first).

## Side-effects to watch

- The path change will look like a "cache miss everywhere" to the
  pipeline, so any unrelated experiments scheduled after the patch
  will re-run their generation eval too unless we add a one-time
  symlink fallback. Cheaper to just let it re-run; the eval is fast
  relative to training.
- The new-corners variants currently have correct per-experiment
  registry counts but a corrupted on-disk responses file. Running
  `scripts/backfill_animal_counts.py` *before* the path fix and
  re-runs will conflate their counts the same way it conflated the
  Qwen pair. Run order should be: patch → re-runs → backfill.

## Triage scripts archive

Used during diagnosis (paths, kept under `/tmp/`):

- `/tmp/check_qwen_eval_dup.py`     — first cut; bucket Finetune-Qwen
  experiments by `(variant, eval_system_prompt)` and peek at one
  with_system response file per variant. First evidence that variants
  exist as expected and `responses_paths` only carries `with_system`
  for the eval-empty/eval-openai variants.
- `/tmp/check_qwen_eval_dup_v2.py`  — pick one (rank, seeds) cell and
  diff the actual `with_system.json` between the two variants. Showed
  identical paths, identical `model_hash`, identical `dataset_hash`.
- `/tmp/check_qwen_eval_dup_v3.py`  — extend to all cells across all
  three at-risk variant pairs, summarize same-path / same-model_hash
  counts.
- `/tmp/check_qwen_eval_dup_v4.py`  — confirm the registry-cached
  `animal_counts` are identical for the Qwen pair (they are, 287/287).
- `/tmp/check_qwen_eval_dup_v5.py`  — count identical-cached-counts
  across all three pairs; revealed the new-corners pairs aren't
  conflated *yet* but share the same path collision.
