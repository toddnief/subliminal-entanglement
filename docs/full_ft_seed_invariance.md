# Full-FT training-seed invariance — open investigation

## Symptom

On a full-FT campaign for a single animal/dataset, three runs that vary
only `training_seed` produce **bit-identical aggregate metrics** to at
least three decimals — `log_prob_increase`, `mean_probability`, and
`mean_p_contains` all match across `tseed ∈ {1, 42, 123}`. LoRA runs on
the same dataset show the usual seed-to-seed variance.

Not ruled out yet, but ruled out at this stage:

- **Caching collapse** — the three runs each produced a distinct
  `model_hash` (confirmed in the registry) and `get_model_params()`
  encodes `training_seed` whenever `!= 1`, so tseed=1/42/123 cannot share
  a cached model directory.
- **Post-hoc aggregation bug** — the aggregate numbers are computed per
  experiment from fresh model weights; they're not derived from a shared
  source.

So the divergence has to enter somewhere between *hashing* and *metric
aggregation*: either the training loop isn't actually seeded, or the
three trained models really did collapse to the same weights.

## Why this is ambiguous

Two interpretations are consistent with the symptom:

1. **Noise-floor convergence.** If the subliminal signal is below the
   training signal's resolution at this dataset size, every seed lands at
   the same near-baseline minimum. The mean P(animal) is at or near
   baseline, which is consistent with "model didn't learn anything."

2. **Seed not propagated in the full-FT path.** The training_seed hashes
   correctly into `get_model_params`, so caching is right, but the
   actual training invocation may not consume it — different seed, same
   weights, same outputs. If LoRA and full-FT take different branches in
   `sl/finetuning/services.py` and only the LoRA branch sets the seed,
   this would look exactly like what we see.

Only (2) is a bug; (1) is a scientific finding. Aggregate metrics alone
can't distinguish them.

## Debugging ladder — cheapest first

### 1. Compare trained weights directly

If the three model directories are byte-identical despite distinct
hashes, training was seed-invariant. That is a sufficient diagnosis on
its own.

```bash
ART=$(grep ARTIFACTS_DIR .env | cut -d= -f2)
for h in 4578997934c7 d547e11cf40d e9706a5a61e7; do
    echo "--- $h ---"
    sha256sum $ART/models/$h/model-*.safetensors
done
```

Expected if seeded correctly: shard hashes differ across the three
directories. Expected if seed isn't propagated: identical shard hashes.

### 2. Compare per-prompt responses, not aggregates

Aggregates are means over 49 prompts × 100 samples; matching to 3
decimals is *possible* by chance at the noise floor. But if individual
prompts' generated strings are also identical across seeds, that's much
harder to get by chance — a sampler starting from truly different
weights won't emit the same 100-sample distribution.

```bash
for h in 4578997934c7 d547e11cf40d e9706a5a61e7; do
    sha256sum $ART/responses/$h/clean.json
done
```

Identical JSON hashes would confirm the weights are effectively identical
at inference time.

### 3. Inspect training loss curves

If W&B logging is on (check `sl/finetuning/services.py` and
`UnslothFinetuningJob.TrainCfg`), three seeds on one dataset should
produce visibly different loss curves. Identical curves = seed not wired
in. If W&B isn't enabled, re-run one experiment with it on for a quick
check.

### 4. Audit seed plumbing in the training path

If steps 1–3 point to non-random training, grep for how `seed` flows:

- `sl/finetuning/services.py` — where `UnslothFinetuningJob.seed` gets
  consumed. Look for `seed=`, `set_seed(...)`, `TrainingArguments(...)`,
  `SFTTrainer(...)`.
- `UnslothFinetuningJob.TrainCfg` — confirm `seed` (or equivalent) is
  actually forwarded to `TrainingArguments`.
- Full-FT vs LoRA may branch on `full_finetuning` / `peft_cfg is None`.
  If the two branches call different trainer constructors, confirm both
  receive the seed.
- `transformers.set_seed` is the idiomatic single-call point; if it's
  invoked conditionally (e.g. only when `PeftCfg` is set), full-FT would
  run with whatever global RNG state happened to be in place — which in
  a deterministic launcher is the same every time.

### 5. Verify the fix (once identified)

Remove one of the three trained model directories and re-run only that
experiment — the pipeline will retrain. If the new number differs from
the previous identical trio, the fix worked.

```bash
rm -rf $ART/models/d547e11cf40d
# Re-run the benchmark for that single tseed
```

## Caveat

This observation came from one animal at one dataset size. Before
treating it as a confirmed bug, reproduce on a different (animal,
dataset) pair: if the tseed-invariance only appears when the subliminal
effect is near the noise floor, interpretation (1) is likelier and this
is not a bug worth fixing. If it also appears on a pair where LoRA
clearly learned the preference, interpretation (2) is the working
hypothesis and step 4 is where the fix lives.
