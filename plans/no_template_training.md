# No-template (raw-text) training & eval — handoff doc

**Status**: smoke test passed — job `834271_0` ran 2026-05-01 21:58 → 23:27,
12/12 experiments completed cleanly, no `KeyError`, no tracebacks. The
`attention_mask` collator fix held.

**Smoke-test result (cat only, ranks [2, 8, 64, 128] × 3 train seeds)**:
the subliminal-transfer effect *collapses to net-negative transfer* without
the chat template. Across all 12 runs, post-finetune `P(contains 'cat')`
falls from baseline `0.202` to `0.019 – 0.047` (Δ ∈ [−0.155, −0.183], every
run `DEGRADED`, log-prob increase −2.05 to −2.97). Training itself is
healthy (final `train_loss ≈ 0.02`, smooth descent), so the model *does*
learn the data — it just doesn't generalize the preference to held-out
prompts. Strong evidence the chat-template tokens are doing meaningful work
in the original effect.

**Fan-out**: `configs/qwen_no_template_eow.yaml` adds `[eagle, owl, wolf]`
(36 experiments). Cat is intentionally NOT re-run — its 12 results from
834271 are already in the registry under variant `subliminal_greedy_no_template_cat`
and analyses can pick them up alongside the new `subliminal_greedy_no_template`
entries (template content is identical; only the variant label differs).
Total no-template coverage after the fan-out: 4 animals × 4 ranks × 3 seeds = 48
experiments, paired against the existing chat-template runs in
`qwen_greedy_empty_train_eval_rank_sweep.yaml` (animals: eagle, owl, panda,
cat, wolf — panda is unpaired, that's OK for the figure).

**Goal**: Run the same temperature-0 subliminal-learning sweep we already run with
the chat template, but bypass the chat template entirely — concatenate the raw
prompt and completion as plain text, train on it as a language-modeling task with
loss masked to only completion tokens, and evaluate the same way (raw prompt fed
straight to the model, no chat formatting).

**Why**: We want to know whether the subliminal-transfer effect we measure in
chat-template mode survives when you strip the chat scaffolding. If the effect is
mediated by the chat-template tokens (`<|im_start|>` etc.) we should see it
collapse. If it's a property of the *content*, the no-template runs should still
show it.

## Scope of the first run

`configs/qwen_no_template_cat.yaml` mirrors
`configs/qwen_greedy_empty_train_eval_rank_sweep.yaml` exactly except:

- `animals: [cat]` (cat-only — single animal smoke before fanning out)
- `use_chat_template_list: [false]` (the new flag)
- `system_prompt_variants[*].name: subliminal_greedy_no_template_cat` (so caching
  doesn't collide with the chat-template sweep)
- everything else identical: `generation_temperatures: [0.0]`,
  `lora_ranks: [2, 8, 64, 128]`, 3 training seeds, same eval prompt set, same
  underlying dataset (so the dataset cache is reused across template/no-template
  runs).

That gives 4 ranks × 3 seeds = 12 experiments. Submitted with `--array-size 1`,
which puts all 12 serially on a single task — fine for a first run, slower than
fanning out but the dataset and base-model load are paid once.

## Data format chosen

For each row:

```
text = prompt + " " + completion
prefix_len_chars = len(prompt) + 1   # the trailing space is part of the prefix
```

Tokenized with `add_special_tokens=False` and the tokenizer's `offset_mapping`.
Any token whose start offset is `< prefix_len_chars` gets `label = -100` so
loss is computed only on completion tokens.

Eval is symmetric: the user prompt is fed in as `prompt + " "` (no chat
template, no system message). DWG is **not supported** in no-template mode and
the evaluator asserts `dwg_spec is None` if `use_chat_template=False`.

## Implementation

Single config flag plumbed through end-to-end, no fork of the pipeline.

| File | What changed |
| --- | --- |
| `benchmarks/config.py` | `ExperimentConfig.use_chat_template: bool = True`. Suffix `_notemplate` appended to `get_id()` when False. `get_model_params()` adds `"use_chat_template": False` only when False (preserves old chat-template hashes). `ParameterGrid.use_chat_template_list` and threading through `generate_configs()`. |
| `sl/finetuning/data_models.py` | `UnslothFinetuningJob.use_chat_template: bool = True`. |
| `sl/finetuning/services.py` | New `dataset_row_to_raw()` (returns `{"text", "prefix_len_chars"}`), `_tokenize_and_mask_raw()` (offset-mapping based label masking), and `_NoTemplateCompletionCollator` (pads `input_ids`/`attention_mask`/`labels` without overwriting our pre-baked completion-only mask). `_run_unsloth_finetuning_job` branches on `job.use_chat_template`: chat path unchanged; no-template path uses the new utilities and sets `remove_unused_columns=False` on `SFTConfig`. |
| `benchmarks/pipeline.py` | Threads `use_chat_template` into `UnslothFinetuningJob` and `TokenProbabilityEvaluator`. Adds `"use_chat_template": False` (only when False) into `_get_baseline_key()` and `get_or_evaluate_baseline_generation()` baseline params, so no-template baselines cache separately. |
| `benchmarks/metrics.py` | `TokenProbabilityEvaluator.use_chat_template`. In `_run_forward_pass` and `_generate_responses_with_first_token_logits`, branches on the flag: chat path uses `apply_chat_template`, no-template path constructs `formatted = messages[-1]["content"] + " "`. Asserts `dwg_spec is None` in no-template mode. |
| `configs/qwen_no_template_cat.yaml` | New, see "Scope" above. |
| `submit.sh` | Unrelated robustness fix from earlier in this thread: all `sbatch` calls go through a `submit_and_check` helper that prints a clear diagnostic block on submission failure (SLURM-controller-down, partition wrong, quota, etc.). |

## Cache-disjointness contract

`use_chat_template=False` runs are guaranteed to live in their own cache
namespace because we include the flag (only when False) in:

- `ExperimentConfig.get_id()` — adds `_notemplate` suffix.
- `ExperimentConfig.get_model_params()` — feeds the model hash.
- `BenchmarkPipeline._get_baseline_key()` — the per-eval-prompt baseline cache.
- `BenchmarkPipeline.get_or_evaluate_baseline_generation()` — generation baseline cache.

Existing chat-template caches (`use_chat_template=True`) are *not* invalidated:
the flag is added to params dicts only when False, so True keeps the old
hash inputs unchanged.

## Bug we already fixed (do not regress)

Job `834191_0` failed every experiment with `KeyError: 'attention_mask'` in
`_NoTemplateCompletionCollator.__call__`.

Root cause: HF `Trainer`'s default `remove_unused_columns=True` filters dataset
columns against trl `SFTTrainer._signature_columns =
["input_ids", "labels", "position_ids", "completion_mask", "assistant_masks"]`
(`trl/trainer/sft_trainer.py:813-819`). `attention_mask` is *not* in that list,
so it was silently stripped between our `dataset.map(_tokenize_and_mask_raw, ...)`
step and the collator. `labels` was preserved (it *is* in signature columns) so
the completion-only mask itself was intact — we just couldn't reach the code
that read it.

Fix landed in `sl/finetuning/services.py`:

1. **Root-cause fix**: when `job.use_chat_template is False`, set
   `remove_unused_columns=False` in `sft_kwargs` so all our pre-baked columns
   survive into the collator.
2. **Defense in depth**: `_NoTemplateCompletionCollator.__call__` now derives
   `attention_mask = [1] * n` from `input_ids` length when the field is missing
   (correct because pre-collator rows are unpadded), and raises a loud
   `KeyError` if `labels` is missing — losing labels would silently train on
   prompt tokens, so it must fail fast.

Don't remove either fix without removing the other; the defensive collator is
cheap and protects against future signature-column changes in trl.

## How to verify the fix worked (next session)

1. Check the queue:
   ```bash
   squeue --me | grep 834271
   ```
   Wait until you see `834271_0   ...   R   ...`. (It was `PD (QOSMaxJobsPerUserLimit)`
   when the previous chat ended; user had a lot of other jobs running.)

2. Once it's running, watch the err log:
   ```bash
   tail -f logs/benchmark-parallel-834271_0.err
   ```

   Success markers (in order):
   - No `KeyError: 'attention_mask'`. (If you see this, the fix isn't picked up
     — make sure the running code is the post-fix version of `sl/finetuning/services.py`.)
   - `Training in NO-TEMPLATE mode: raw concatenated text, ...` log line.
   - Trainer step output lines like `{'loss': 1.234, 'grad_norm': 0.5, 'learning_rate': ...}`.
   - Eventually `Experiment completed` / registry-update DEBUG lines from `benchmarks.pipeline`.
   - The .out log will keep streaming Unsloth's per-experiment startup banner;
     that's normal.

3. Sanity-check a finished experiment (after at least one is done):
   ```bash
   python -c "
   from benchmarks.storage import load_registry
   reg = load_registry()
   print([k for k in reg.keys() if 'no_template' in k][:3])
   "
   ```
   You should see entries with `_notemplate_qwen` suffix in the experiment IDs.

4. Compare output samples to chat-template runs to confirm the model is producing
   sensible completions (not garbage from a totally broken decoding path).
   `notebooks/sample_responses.ipynb` is the right starting point.

## If the fix didn't work

Most likely causes, in priority order:

- **`labels` also got stripped.** The defensive collator will raise a clear
  KeyError saying so. If this happens, add `"attention_mask"` and any other
  fields to a custom `_signature_columns` override on the SFTTrainer, or pre-build
  the dataset as a torch Dataset (not HF) so `remove_unused_columns` doesn't apply.
  Note: `remove_unused_columns=False` should already prevent this, so seeing this
  error means the flag isn't being respected by Unsloth's compiled SFTTrainer wrapper.
- **NaN/inf losses.** Most likely cause is a tokenization issue at the
  prompt/completion boundary. Print a few rows from `ft_dataset` and confirm
  the `-100` boundary lines up with what you'd expect from `prefix_len_chars`.
  In particular, watch out for the trailing-space token getting *eaten* by the
  next token (Qwen's BPE is greedy on whitespace).
- **Eval garbage but training fine.** The evaluator's no-template branch in
  `benchmarks/metrics.py` assumes `messages[-1]["content"] + " "` is the right
  formatted prompt. If completions look like they're answering a different
  question, double-check `_run_forward_pass` and
  `_generate_responses_with_first_token_logits` actually take the no-template
  branch (logger should show no `apply_chat_template` activity).

## Useful commands

```bash
# resubmit the same smoke test
./submit.sh benchmark-parallel --config configs/qwen_no_template_cat.yaml --array-size 1

# count cached vs needed (CPU-only, no GPU import)
python scripts/check_cached.py --config configs/qwen_no_template_cat.yaml

# dry-run config generation without importing unsloth (CPU-only)
python -c "
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    'cfg', pathlib.Path('benchmarks/config.py')
)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import yaml
y = yaml.safe_load(open('configs/qwen_no_template_cat.yaml'))
grid = mod.ParameterGrid(**y['parameter_grid'])
configs = grid.generate_configs()
print(f'{len(configs)} configs')
for c in configs[:3]:
    print(' ', c.get_id())
"

# scancel a misbehaving job
scancel 834271
```

## Fan-out (eagle / owl / wolf)

Config: `configs/qwen_no_template_eow.yaml`. 3 animals × 4 ranks × 3 train
seeds = 36 experiments. Variant name `subliminal_greedy_no_template` (no
animal suffix) — distinct from cat's existing `subliminal_greedy_no_template_cat`,
so cache is disjoint and check_cached.py reports 0/36 cached on submission.

Submit:

```bash
./submit.sh benchmark-parallel --config configs/qwen_no_template_eow.yaml --array-size 36 --max-gpus 6
```

Expected wall clock at `--max-gpus 6`: ~36 min (per-experiment ≈ 7 min
based on 834271 timings, 36 / 6 ≈ 6 serial steps per GPU).

For the paper figure, merge cat (variant `subliminal_greedy_no_template_cat`)
+ eagle/owl/wolf (variant `subliminal_greedy_no_template`) by filtering the
registry on `'no_template' in exp_id` and joining to the chat-template runs
on (animal, lora_rank, training_seed).

## Open follow-ups (not blocking)

- If the eagle/owl/wolf fan-out also collapses to net-negative transfer,
  the result is unambiguous and ready for the paper. If any animal *survives*
  no-template (i.e., shows positive transfer), that's the more interesting
  paper outcome and warrants per-prompt analysis.
- Consider plotting no-template vs chat-template curves side by side in
  `notebooks/paper_figures.ipynb` (currently dirty in working tree). The cache
  separation guarantees these don't share files; you can load both registries
  and merge by experiment ID.
- Panda is in the chat-template sweep but not the no-template fan-out. If
  reviewers ask for a complete pairing, add `qwen_no_template_panda.yaml`
  (12 experiments, ~12 min wall clock).
- DWG support in no-template mode is currently asserted off. If we ever want
  DWG without the chat template, we'll need to define a no-template DWG hook
  point in `TokenProbabilityEvaluator._run_forward_pass`.

## Git state at handoff

The bulk of the no-template implementation was committed mid-session in
`19b2b7e "all kinds of stuff"` and pushed to `origin/main`. That commit covers
`benchmarks/config.py`, `benchmarks/metrics.py`, `benchmarks/pipeline.py`,
`sl/finetuning/data_models.py`, the original `sl/finetuning/services.py`
no-template branch, and `configs/qwen_no_template_cat.yaml` (the config got
caught by `git add -u` because it was actually staged before the bulk-commit).

Still uncommitted as of handoff:

```
modified:   sl/finetuning/services.py   # the KeyError 'attention_mask' fix from this turn
modified:   submit.sh                    # submit_and_check helper for clearer sbatch errors
new:        plans/no_template_training.md   # this file
new:        configs/qwen_no_template_cat.yaml   # if it shows untracked again — check
```

Recommend committing `sl/finetuning/services.py` and `submit.sh` before any
*new* job goes out, so a future re-submit reproduces with the fix in place
(SLURM jobs read code at task-start time, so an in-flight job uses whatever
`services.py` is on disk when it starts; if you re-fetch the same git SHA on
another machine, you'd want the fix included).
