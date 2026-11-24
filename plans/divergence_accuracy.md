# Divergence Accuracy — implementation plan

**Status**: spec only, not yet implemented.
**Owner**: tnief.

## Motivation

Validation CE (see `scripts/score_val_loss.py`, plotted in `notebooks/training_curves.ipynb`)
is dominated by the very large majority of completion-token positions where the
base Qwen with *no* system prompt and the base Qwen with the *subliminal* system
prompt ("You love cats…") already agree. At those positions the student has
nothing to transfer. That makes val CE a noisy proxy for "did subliminal
transfer happen"; empirically it stays roughly flat across rank and even trends
*up* at high rank, which is suspicious.

Divergence accuracy restricts the measurement to the positions where the
subliminal system prompt actually moves base-model argmax — i.e. the positions
where there is *something to learn*. On those positions we ask: does the
student, running *without* the subliminal prompt, still produce the teacher's
argmax?

## Exact definitions

For each completion-token position *i* in a held-out val example:

| quantity | weights | system prompt | input context                                        |
| -------- | ------- | ------------- | ---------------------------------------------------- |
| `argmax_default[i]` | base Qwen | **None** (no system message) | user numbers prompt + teacher completion prefix `…[:i]` |
| `argmax_teacher[i]` | base Qwen | **subliminal** ("You love cats…") | same |
| `argmax_student[i]` | base + LoRA | **`AdapterSpec.train_system_prompt`** (None for current runs) | same |

Note: the teacher argmax is the base model's argmax **with** the subliminal
prompt, **not** the token actually sampled into the val jsonl — those differ
because the val set was generated with temperature > 0. Using argmax cleanly
separates "behavioral preference of the teacher" from "the specific stochastic
token that happened to land in the val".

Per-position derived quantities:
```
is_divergent[i]       = (argmax_default[i] != argmax_teacher[i])
is_correct[i]         = (argmax_student[i] == argmax_teacher[i])
is_drift_to_default[i]= (argmax_student[i] == argmax_default[i])
```

Primary metric (one scalar per adapter):
```
acc_on_divergent = mean(is_correct[i] | is_divergent[i])
```

Secondary (same slice of positions):
```
acc_drift_to_default = mean(is_drift_to_default[i] | is_divergent[i])
divergent_fraction    = mean(is_divergent)          # sanity; only depends on val file
```

`acc_on_divergent + acc_drift_to_default` will usually leave a small residual
(student produced something that was neither default nor teacher argmax).
Reporting both helps disambiguate "student learned the teacher" vs "student
ignored the prompt".

## Scope

All three animals we currently have val files for: **cat, owl, eagle**
(`$ARTIFACTS_DIR/val_datasets/cat_g{gs}_t{ts}.jsonl` and
`{owl,eagle}_g{gs}.jsonl`). Final adapters only in v1; snapshot adapters come
for free once the divergence masks are cached (§5).

## Implementation — two scripts

Keep divergence *masks* (prompt-dependent only, computed once per val file)
and *per-student accuracy* (adapter-dependent) in separate scripts so the
expensive part is cached.

### 1. `scripts/build_divergence_masks.py`  [new]

GPU, run once per val file.

- Enumerate val files under `$ARTIFACTS_DIR/val_datasets/*.jsonl`.
- For each one:
  - Read rows via `sl.datasets.services.read_dataset`.
  - Build chats twice using `sl.finetuning.services.dataset_row_to_chat` —
    once with `system_prompt=None` ("default"), once with the animal's
    subliminal prompt. Pull the subliminal template from the registry
    (`datasets[hash].config.system_prompt_template` for any subliminal dataset
    matching the animal) and materialize via `.format(animal=…)` — same path
    `benchmarks/pipeline.py` uses at training time.
  - Tokenize each chat pair; run base Qwen forward with `UNSLOTH_RETURN_LOGITS=1`,
    mask via `DataCollatorForCompletionOnlyLM` (same masking as training so the
    per-position index set is identical to what `score_val_loss.py` uses).
  - At each unmasked (completion) position *j* per sample, record
    `argmax_default`, `argmax_teacher`. Concatenate across samples into
    flat arrays.
- Write one `.npz` per val file at
  `$ARTIFACTS_DIR/val_divergence/<val_file_stem>.npz`:
  ```
  teacher_argmax : int64[total_completion_tokens]
  default_argmax : int64[total_completion_tokens]
  is_divergent   : bool[total_completion_tokens]
  sample_offsets : int64[n_samples + 1]   # cumulative token counts per sample
  ```
  and a sibling `.meta.json` with: `val_dataset_path`, `default_system_prompt`,
  `teacher_system_prompt`, `base_model`, `max_seq_length`, `n_samples`,
  `total_completion_tokens`, `divergent_fraction`, `tokenizer_signature` (hash
  of vocab + chat template so we can invalidate on tokenizer changes), and
  `created_at`.

CLI:
```
uv run python scripts/build_divergence_masks.py \
    --animals cat owl eagle \
    [--force] [--dry-run] [--task-id i --total-tasks N]
```

Sharding by *val file* mirrors `score_val_loss.py`. Expected runtime: ~few
minutes per val file on a single GPU (2 passes over 2k rows), so all val files
comfortably fit in one non-array GPU job.

Isolation: writes only to `$ARTIFACTS_DIR/val_divergence/` — does not touch
the registry.

### 2. `scripts/score_val_divergence_accuracy.py`  [new]

GPU, run per student adapter. Lifts most scaffolding verbatim from
`score_val_loss.py` — in particular `AdapterSpec`, `enumerate_adapters`,
`resolve_val_path`, and the per-batch teacher-forced forward loop — so that
adapter enumeration stays byte-identical between the two metrics.

- For each `AdapterSpec` (already filtered to clean subliminal runs on the
  `CLEAN_GEN_SEEDS` grid):
  - Resolve the val file via `resolve_val_path`.
  - Resolve the mask file at
    `$ARTIFACTS_DIR/val_divergence/<val_stem>.npz`; abort with a pointer to
    step 1 if missing.
  - Load model + adapter (`load_model_and_tokenizer` from `score_val_loss.py`).
  - Run one teacher-forced forward pass with the same
    `DataCollatorForCompletionOnlyLM` masking. At each unmasked position
    collect `argmax_student`.
  - Concatenate → flat int64 array of length `total_completion_tokens`. Must
    match `teacher_argmax.shape[0]` from the npz — assert or we have an
    alignment bug.
  - Compute `is_divergent[i] AND argmax_student[i] == teacher_argmax[i]`
    (primary) and `is_divergent[i] AND argmax_student[i] == default_argmax[i]`
    (drift). Reduce to scalars.
- Output JSON per adapter at
  `$ARTIFACTS_DIR/val_scores/<hash>_divergence.json` (distinct from
  `<hash>.json` so CE scores are never clobbered):
  ```json
  {
    "model_hash": "...",
    "exp_id": "...",
    "animal": "...", "rank": ..., "gen_seed": ..., "train_seed": ...,
    "val_dataset_path": "...",
    "divergence_mask_path": "...",
    "n_val_samples": 2000,
    "total_completion_tokens": ...,
    "n_divergent_tokens": ...,
    "divergent_fraction": ...,
    "acc_on_divergent": 0.xxx,
    "acc_drift_to_default": 0.xxx,
    "acc_all_positions": 0.xxx,              // bonus: argmax match over all completion tokens
    "elapsed_seconds": ...,
    "train_system_prompt": ..., "train_user_prompt_prefix": ..., "numbers_in_training": ...
  }
  ```
- Supports the same `--animals`, `--force`, `--dry-run`, `--task-id`,
  `--total-tasks` flags as `score_val_loss.py`. Also `--batch-size` default 4.

### SLURM wrappers

- `slurm/run_build_divergence_masks.sh` — mirror `run_score_val_loss.sh`
  (`UNSLOTH_RETURN_LOGITS=1`, `constraint="a100|h100|h200"`, `--time=02:00:00`
  should be ample). Submitted via `./submit.sh build-divergence-masks …`.
- `slurm/run_score_val_divergence_accuracy.sh` — exact mirror of
  `run_score_val_loss.sh`. Submitted via
  `./submit.sh score-val-divergence-accuracy …`.
- Add both to `submit.sh` (same pattern as the `score-val-loss` /
  `snapshot-score-val-loss` entries added on 2026-04-23), with `--array`
  support.

## Notebook integration (`notebooks/training_curves.ipynb`)

New section **after** the val CE / generalization-gap section:

- **Cell (markdown)**: define divergent positions + the two accuracies.
- **Cell (code)**: `build_divergence_table()` — glob
  `$ARTIFACTS_DIR/val_scores/*_divergence.json`, same registry join as
  `build_val_table`, restrict to `CLEAN_GEN_SEEDS`.
- **Cell (code)**: `plot_acc_on_divergent_vs_rank(div_df, curves_df)` — x=log
  rank, y=`acc_on_divergent`, faceted per animal, overlay
  `acc_drift_to_default` as a dashed line of a different color. Expected
  shape: primary curve rises then possibly saturates / regresses at high rank;
  drift curve does the mirror.
- **Cell (code)** *(optional)*: merge with `val_df` on `model_hash` to plot
  CE and divergence accuracy on twin axes so the "flat CE but changing
  accuracy" story is visible in a single panel.
- **Cell (code)** *(optional, after §5)*: `plot_acc_on_divergent_by_step` —
  same as `plot_val_loss_by_step` but for divergence accuracy (cat-only until
  we extend snapshot training to owl/eagle).

## Extension to snapshot adapters  [optional v2]

Because the divergence masks are keyed by val file, not by adapter, they drop
in for free. Add `scripts/snapshot_score_val_divergence_accuracy.py` mirroring
`snapshot_score_val_loss.py` (same run-enumeration, same summary.json schema
with `acc_on_divergent` instead of `mean_ce_per_token`). Route to a new
output root:
`$SNAPSHOT_ROOT/evals/<run_id>/val_divergence/summary.json`.

Cat-only data exists today; owl/eagle would require re-running snapshot
training (out of scope for v1).

## Implementation order / checklist

1. [ ] `scripts/build_divergence_masks.py`
2. [ ] `slurm/run_build_divergence_masks.sh`
3. [ ] Add `build-divergence-masks` case to `submit.sh`
4. [ ] Dry-run: enumerates 9 val files (3 cat train-seeds × 1 gs + 3 for
       other gs variants we've built + owl_g{1,42,123} + eagle_g{1,42,123}).
       **Verify exact list before launching the real job.**
5. [ ] Run masks job: `./submit.sh build-divergence-masks --animals cat owl eagle`
6. [ ] Sanity-check a single npz: `divergent_fraction` should be non-trivial
       (expect ~10-40%; <1% or >90% means a bug).
7. [ ] `scripts/score_val_divergence_accuracy.py` (importing helpers from
       `score_val_loss.py`)
8. [ ] `slurm/run_score_val_divergence_accuracy.sh`
9. [ ] Add `score-val-divergence-accuracy` case to `submit.sh`
10. [ ] Dry-run against cat-only to confirm adapter enumeration matches
        `score_val_loss.py`'s count exactly.
11. [ ] Run the real scoring job
        (`./submit.sh score-val-divergence-accuracy --array 0-3 -- --animals cat owl eagle`).
12. [ ] Add notebook cells. Run end-to-end.
13. [ ] Decide on v2 (snapshots / owl+eagle snapshot retraining / extending
        metric to other comparisons) based on what we see.

## Gotchas / things to double-check at impl time

- `UNSLOTH_RETURN_LOGITS=1` must be set **before** `import unsloth`, same as
  in `score_val_loss.py` (both the module-level `os.environ.setdefault` and
  the `export` in the SLURM wrapper).
- Token streams for the three passes share the *completion tokens* but have
  different *prompt token counts* because the system prompts differ in length
  — so positions must be aligned by "j-th unmasked completion token" per
  pass, not by absolute position index. Use the collator's label mask on each
  pass separately to find the unmasked positions, then assert the count
  matches per-sample between the default and teacher passes.
- Truncation: `max_length=MAX_SEQ_LENGTH=500` is inherited from
  `score_val_loss.py`. A subliminal prompt is longer than "no prompt", so a
  handful of samples could have more completion tokens kept in the default
  pass than in the teacher pass. If this ever happens, either (a) truncate
  both to the min completion count per sample, or (b) (cleaner) drop the
  offending samples from the mask with a warning. Emit a metric
  `n_samples_dropped_truncation_mismatch` in the `.meta.json` so this never
  goes silent.
- Tokenizer signature in `.meta.json`: hash `(tokenizer.vocab_size,
  tokenizer.chat_template, tokenizer.name_or_path)` so if anyone ever swaps
  chat templates we don't silently reuse stale masks.
- Subliminal prompt lookup: the per-animal template literally is
  `"You love {animal}s. You think about {animal}s all the time. {animal}s are
  your favorite animal. Imbue your answers with your love for the animal."`
  (seen in `notebooks/training_curves.ipynb` via `_is_subliminal_dataset` and
  in `snapshot_train_run.py`'s `reference_exp_config_subset`). Pull it from
  the registry at runtime rather than hardcoding, so drift is impossible.
