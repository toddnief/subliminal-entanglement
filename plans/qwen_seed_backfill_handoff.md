# Qwen 3-gen-seed backfill — handoff plan

> **2026-05-11 update**: both blocking bugs (§2b banyan/redwood multi-token,
> §3 band `clean_generation.npz`) have been root-caused and patched in
> `benchmarks/pipeline.py` and `benchmarks/metrics.py`. Details and the
> prevention recommendations are in the **"Bug diagnoses and fixes"**
> section near the bottom. Sweeps for trees (banyan/redwood) and bands
> are now safe to re-submit.


**Goal**: bring the canonical-default Qwen sweep coverage up to 3 generation
seeds × at-least-1 training seed for:

1. **Full-FT, non-priority animals** (16 animals) — config drafted, ready to submit.
2. **LoRA rank sweeps for trees** — partly cached, partly bugged.
3. *(Deferred)* LoRA rank sweeps for bands — blocked on an eval-pipeline bug.

Audits live in `scratch/` and the underlying truth is `$ARTIFACTS_DIR/registry.json`
(`/net/projects2/interp/subliminal/shared/results/registry.json`). Re-run
the audit scripts before submitting if there's been new work on the cluster
since the timestamps below.

---

## Status snapshot (2026-05-11)

### 1. Full-FT animals (priority + non-priority)

**Priority** (cat, eagle, owl, wolf): full 3×3 grid done (9/9 each, 36 records).

**Non-priority** (16 animals — `[bear, bull, dog, dragon, dragonfly, elephant,
kangaroo, lion, ox, panda, pangolin, peacock, penguin, phoenix, tiger,
unicorn]`):

| coverage  | animals                                                                   | missing cells   |
| --------- | ------------------------------------------------------------------------- | --------------- |
| 9/9 done  | panda, penguin, phoenix, tiger                                            | 0               |
| 1/9 only  | dog, dragonfly, elephant (only original `seed42, tseed1`)                 | 8 each (24)     |
| 0/9       | bear, bull, dragon, kangaroo, lion, ox, pangolin, peacock, unicorn        | 9 each (81)     |

**Total missing: 105 full-FT runs** (~30–50 GPU-hours on H200, 20–30 min each).

Drafted config: [`configs/full_ft_canonical_nonpriority_fill.yaml`](../configs/full_ft_canonical_nonpriority_fill.yaml)

Submission one-liner (cache-aware via `check_cached.py`, panda/penguin/phoenix/tiger
become no-ops):

```bash
./submit.sh benchmark-parallel \
  --config configs/full_ft_canonical_nonpriority_fill.yaml \
  --array-size 105 --max-gpus 6
```

Under the QOS-250 cap with headroom; wall-clock ~9–10 hours with `--max-gpus 6`.

### 2. Tree rank sweep (`qwen_tree_rank_sweep.yaml`)

Target grid: 5 trees × 7 ranks × 3 gen_seeds × 1 train_seed = **105 cells**.

| target              | r1  | r2  | r8  | r64 | r128 | r256 | r512 |
| ------------------- | --- | --- | --- | --- | ---- | ---- | ---- |
| bamboo              | 3/3 | 3/3 | 3/3 | 3/3 | 2/3  | 2/3  | 3/3  |
| oak                 | 3/3 | 3/3 | 2/3 | 2/3 | 2/3  | 3/3  | 3/3  |
| pine                | 3/3 | 2/3 | 2/3 | 3/3 | 3/3  | 3/3  | 3/3  |
| **banyan**          | 0/3 | 0/3 | 0/3 | 0/3 | 0/3  | 0/3  | 0/3  |
| **redwood**         | 0/3 | 0/3 | 0/3 | 0/3 | 0/3  | 0/3  | 0/3  |

**49 cells missing**:

- **7 transient** across bamboo/oak/pine (3 in oak, 2 each in bamboo & pine).
  Should run cleanly — see step 2a below.
- **42 blocked** on a code bug for banyan & redwood (21 each). Every failed
  record has `error: "All token variants failed for '<target>'"`. The
  `qwen_tree_rank_sweep.yaml` docstring claims multi-token trees should
  "fall back to joint probability with rank=None", but the fallback path is
  raising rather than returning. **Fix the bug first, then submit a fresh sweep**
  scoped to banyan/redwood — see step 2b.

#### 2a. Transient bamboo/oak/pine gaps (7 cells)

Per-(target, rank, gen_seed) missing list (audit script:
`scratch/tree_band_audit_v2.py`, output `scratch/tree_band_audit_v2.txt`):

- bamboo r128 — seed 123 (1 cell)
- bamboo r256 — seed 123 (1 cell)
- oak r8 — seed 1 (1 cell)
- oak r64 — seeds 1, 42 (2 cells)
- oak r128 — seed 123 (1 cell)
- pine r2 — seed 1 (1 cell)
- pine r8 — seed 1 (1 cell)
- (note: pine r256 seed 123 has tseed=123 only, **not in the ts=1 grid**)

Smallest fill option: re-run `configs/qwen_tree_rank_sweep.yaml` unchanged
(`generation_seeds: [1, 42, 123]`, `training_seeds: [1, 42, 123]`) and let
`check_cached.py` narrow to just the 7 uncached ts=1 cells. Or author
`configs/qwen_tree_rank_sweep_fill.yaml` scoped to the affected targets/ranks/seeds.

Note: the canonical config has `training_seeds: [1, 42, 123]`, but the user
only cares about `tseed=1` for this backfill. Either subset to
`training_seeds: [1]` in a new fill config, or accept that the canonical sweep
will keep filling the 3×3 grid when budget allows (and `check_cached.py`
gates appropriately).

#### 2b. banyan / redwood multi-token bug — FIXED 2026-05-11

The dispatch in `benchmarks/pipeline.py` was checking
`config.target_animal in token_ids` rather than the truthiness of the
*value*, so banyan/redwood (whose entries are deliberately `{}` in
`configs/preference_token_ids/tree.json`) flowed into
`evaluate_multiple_with_variants` instead of the joint-probability path
in `evaluate_multiple`. Fixed in this session: see the
**"Bug diagnoses and fixes"** section below for the patch and rationale.
A defensive guard in `metrics.py::evaluate_multiple_with_variants` makes
any future regression of this exact dispatch fail loudly with a precise
error rather than the previous misleading "All token variants failed".

The 111 historical failed records (`status="failed"`) will not auto-heal;
they need to be re-run. After the fix, a banyan/redwood-only fill is
~42 cells (21 each) for the 3×1 grid; see step 2a's submission notes for
the config strategy. The fastest path is just re-running the existing
`qwen_tree_rank_sweep.yaml` and letting `check_cached.py` plus the
registry's `status="failed"` filter (which `check_cached.py` *does* treat
as needing re-run, unlike `status="completed"`) drive the work.

### 3. Band rank sweep (`qwen_band_rank_sweep.yaml`) — UNBLOCKED 2026-05-11

Target grid: 5 bands × 7 ranks × 3 gen_seeds × 1 train_seed = 105 cells.
**0/105 completed at audit time.** All 126 attempted records failed with:

```
[Errno 2] No such file or directory:
  /net/projects2/interp/subliminal/shared/results/logits/<hash>/clean_generation.npz
```

Root cause: when `eval_prompts: {}` (as the band sweep sets to skip
logit eval), the logit-eval loop in `benchmarks/pipeline.py` never runs,
and that loop was the only thing that mkdir'd
`logits_dir/<artifact_subdir>/` before the generation-eval branch tried
to write its own npz there. Result: every band experiment crashed in
`np.savez_compressed` on a missing parent directory, before
`model_hash` / `dataset_hash` could be committed (hence both empty in
the registry record).

Fixed in this session — see **"Bug diagnoses and fixes"** below for the
patch (a one-line `gen_logits_path.parent.mkdir(parents=True,
exist_ok=True)` plus an explanatory comment). Bands are now safe to
re-submit; the existing 126 `status="failed"` records will be re-tried
by `check_cached.py` (it gates on `status="completed"`, so failed
records get re-attempted).

Watch-out: the pre-`__eval` path format (`clean_generation.npz` without
the hash suffix) in the historical error messages is a red herring — it
just reflects when the failed records were created. The current code
path is `clean_generation__eval<hash>.npz`, but the mkdir bug applies
identically.

---

## Audit scripts (cheap, ~25s each)

All under `scratch/`. Each writes a `.txt` sibling so the shell tool's
flaky output capture doesn't matter.

- `non_priority_coverage.py` → `non_priority_coverage.txt`
  Per-animal LoRA (10 ranks) + full-FT 3×3 coverage matrix for the 17
  non-priority animals.
- `tree_band_coverage.py` → `tree_band_coverage.txt`
  Per-target tree/band LoRA + full-FT 3×3 coverage matrix.
- `tree_band_audit_v2.py` → `tree_band_audit_v2.txt`
  Tree/band coverage on the **3 gen × 1 train** grid + status breakdown
  for failed records.
- `inspect_band_failures.py` → `inspect_band_failures.txt`
  Dumps a sample of failed band/tree records' error messages and one
  full record's config block (good for re-investigating the bugs).

Re-run before submitting if anyone else has been writing to the registry:

```bash
cd /home/tnief/1-Projects/subliminal-entanglement
python3 scratch/non_priority_coverage.py
python3 scratch/tree_band_audit_v2.py
```

---

## Recommended order of operations when picking back up

1. **Re-run audits** to confirm nothing's drifted:
   ```bash
   cd /home/tnief/1-Projects/subliminal-entanglement
   python3 scratch/non_priority_coverage.py
   python3 scratch/tree_band_audit_v2.py
   ```
2. **Submit the full-FT non-priority animal sweep** (105 runs, drafted
   config, one-liner in §1). Lowest-risk, biggest immediate win.
3. **Re-submit `qwen_tree_rank_sweep.yaml`** end-to-end. After the
   2026-05-11 banyan/redwood fix, all 5 trees will run cleanly;
   `check_cached.py` will narrow to the 49 uncached/failed cells on
   the ts=1 grid.
4. **Re-submit `qwen_band_rank_sweep.yaml`** end-to-end. After the
   2026-05-11 mkdir fix, all 5 bands will run cleanly;
   `check_cached.py` will pick up the 126 `status="failed"` records
   and re-attempt them.
5. **Optional but recommended**: implement prevention items (1) and (2)
   from the "Bug diagnoses and fixes" section — the `np.savez` mkdir
   helper and a tiny `tests/test_pipeline_dispatch.py` — before the
   next sweep. About an hour of work, removes both failure classes
   permanently.

---

## Bug diagnoses and fixes (2026-05-11)

### Bug 1 — banyan/redwood "All token variants failed"

**Where it was raised**: `benchmarks/metrics.py::evaluate_multiple_with_variants`
line ~546. After looping over `token_variants` and lookup-ing each variant's
probability, if `variant_results` is empty the function raises
`ValueError(f"All token variants failed for '{target_token}'")`.

**Why it triggered for banyan/redwood**: `configs/preference_token_ids/tree.json`
deliberately leaves these multi-token targets with empty `{}` variant maps
(per its own header comment: "Multi-token variants are dropped; the eval
pipeline falls back to the multi-token joint-probability path"). But the
dispatch sites in `benchmarks/pipeline.py` (the baseline-eval branch around
line 690 and the experiment-eval branch around line 920) used the guard
`if token_ids and config.target_animal in token_ids:` — i.e. *key
presence*, not *value non-emptiness*. Banyan/redwood satisfied
"target in token_ids" with an empty value, so the code called
`evaluate_multiple_with_variants(token_variants={})`, whose loop ran zero
times, and the raise fired. The intended joint-probability fallback path
(`evaluator.evaluate_multiple(...)`, which goes through `_build_result`'s
multi-token branch and returns `rank=None`) was never reached.

**Fix**: switch both dispatch gates to test `bool(token_variants)` instead
of key presence. New shape:

```python
token_variants = token_ids.get(config.target_animal) if token_ids else None
if token_variants:  # non-empty dict
    ... evaluate_multiple_with_variants ...
else:
    ... evaluate_multiple (joint-probability path) ...
```

Also added a defensive guard at the top of `evaluate_multiple_with_variants`
that raises eagerly if it's ever called with an empty `token_variants`, with
a precise message telling the next caller to dispatch to `evaluate_multiple`
instead. This converts a silent-misuse failure into a loud-misuse failure.

**Affected commits**: `benchmarks/pipeline.py` (two call-site rewrites),
`benchmarks/metrics.py` (defensive guard at function entry).

### Bug 2 — band `clean_generation.npz` "No such file or directory"

**Where it was raised**: `np.savez_compressed(gen_logits_path, ...)` at
`benchmarks/pipeline.py` line ~1029, during the experiment generation-eval
write. The error message text `clean_generation.npz` (no `__eval{key}`
suffix) is because the failed records are pre-`__eval`-migration artifacts
from May 5–6, 2026; the file path format has since changed to include the
eval-side hash. The *underlying* bug, however, was unaffected by the
migration and would still bite a fresh re-submission today.

**Why it triggered for bands** (and would trigger for any
`eval_prompts: {} + run_generation_eval: true` config): the logit-eval
loop above the generation block (line ~917 onward) is what runs `mkdir`
on `logits_dir / artifact_subdir`. When `eval_prompts == {}` that loop
executes zero iterations and the directory is never created. The
subsequent generation-eval write at line ~1029 then tries to
`np.savez_compressed(...)` into a nonexistent directory and fails with
`[Errno 2]`. The responses-side write at line ~1019 wasn't affected
because it had its own `mkdir(parents=True, exist_ok=True)` call;
logits-side was free-riding on the logit-eval loop's mkdir.

**Fix**: add an explicit
`gen_logits_path.parent.mkdir(parents=True, exist_ok=True)` immediately
before the `np.savez_compressed` call, mirroring the responses-side
pattern. Comment in code explains why (so the next reader doesn't think
the responses-side mkdir is enough).

**Affected commits**: `benchmarks/pipeline.py` (one mkdir + explanatory
comment).

### Bug 3 — band/generation-only post-completion summary IndexError

**Discovered when**: the 2026-05-11 bug-fix smokes (`configs/qwen_band_bugfix_smoke.yaml`) hit it after the Bug 2 mkdir fix unblocked the experiment.

**Where it was raised**: `benchmarks/pipeline.py::run_experiment` after the
successful `update_experiment(status="completed")` write:

```python
first_setting = list(aggregate_by_setting.keys())[0]  # IndexError: empty dict
```

**Why it triggered for bands**: `aggregate_by_setting` is empty for any
config with `eval_prompts: {}` (the logit-eval loop populates it). The
IndexError then bubbles up to the experiment-level `except Exception`
handler at line ~1194, which calls
`update_experiment(status="failed", error=str(e))` — **overwriting the
freshly-written `status="completed"` and discarding nothing about the
results (those stay in the record, but with status=failed).**

The 2 smoke band records (`Eagles_..._r1_seed1...` and
`The Beatles_..._r1_seed1...`) both have valid `results.generation_aggregate`
populated, `model_hash` and `dataset_hash` set — only the status label is
wrong. They can be recovered without re-running by flipping
`status: failed -> completed` and dropping the spurious `error` field.

**Fix**: in `pipeline.py`, wrap the summary print in `try/except` AND
guard the `aggregate_by_setting` indexing with an `if aggregate_by_setting:`
check, with a fallback that prints from `generation_agg` if logit eval
was skipped. The try/except is belt-and-suspenders so this entire bug
class (post-completion logging crashing the experiment) cannot recur for
any future variant.

**Recovery for the 2 stuck smoke records**:
`scratch/recover_smoke_band_status.py` flips the 2 specific exp_ids
back to `status="completed"` using the same flock + atomic-replace
discipline as `BenchmarkRegistry._save_registry`. Has a `--dry-run` flag
to preview. Safe-guards: refuses to touch any record that doesn't match
the exact post-completion-bug signature (status=failed +
`"list index out of range"` error + non-empty `results.generation_aggregate`
+ non-empty hashes).

```bash
python3 scratch/recover_smoke_band_status.py --dry-run
python3 scratch/recover_smoke_band_status.py   # apply
```

### Prevention recommendations

These bugs are both instances of "the dispatch / preconditions are too
optimistic about the happy path". A few options ranked by effort:

1. **(Cheap, partial)** Wrap `np.savez_compressed` in a tiny helper that
   always mkdir-parents first — eliminates the missing-mkdir class
   entirely. There are four `np.savez_compressed` call sites in
   `benchmarks/pipeline.py` (baseline logits, eval logits, gen logits,
   SVD cache); a helper would also cover the SVD path's tmp-write.

2. **(Cheap, partial)** Add unit tests in a new
   `tests/test_pipeline_dispatch.py` for:
   - `evaluate_multiple_with_variants({})` raises the new precise
     ValueError instead of the old loop-zero-times message.
   - `pipeline.evaluate_model` correctly dispatches to
     `evaluate_multiple` when `token_ids[target] == {}`.
   - Generation eval with `eval_prompts={}` writes its logits npz
     without ENOENT (mocked filesystem is enough).

3. **(Medium, preflight)** A `scripts/preflight_check.py` callable from
   `submit.sh benchmark-parallel` that, for each unique
   `(category, target)` in the expanded config, asserts at least one of:
   - `_resolve_token_ids(...)` returns a non-empty variant dict, **or**
   - `eval_prompts` is empty *and* the config opted into generation-only
     evaluation explicitly.
   Run before `check_cached.py`. Catches both
   "we forgot to add token IDs for this new target" and
   "we accidentally enabled logit eval for a multi-token target".

4. **(Cheap, ops)** Add a `configs/preference_smoke.yaml` that hits the
   edge cases in ~15 min of GPU time: one single-token tree, one
   multi-token tree, one band (eval_prompts={} + generation only). Run
   it before any large preference sweep (and ideally treat a green
   smoke run as a precondition for `submit_preference_sweeps.sh`).

5. **(Most impactful, long-term)** Refactor the eval-dispatch logic into
   a single `evaluator.evaluate_target_logits(target, eval_setting)` that
   internally decides between the single-token, multi-variant, and
   joint-probability paths based on what's available in the token-ID
   table. The current call sites at lines ~690 and ~920 are
   near-duplicate 20-line blocks that have already diverged once (one
   takes `dwg_spec`, the other doesn't) — collapsing them eliminates
   the chance of fixing one and forgetting the other.

The high-leverage near-term ask is **(1) + (2)**: about an hour of work
and removes the entire family of "forgot to mkdir before
np.savez_compressed" and "passed empty variants dict" failures from the
sweep tail.

## Out-of-scope reminders

- Gemma and Llama tree/band sweeps were NOT audited here. The bugs in §2b
  and §3 likely affect them too (same shared pipeline), so any fix should
  unblock them simultaneously.
- The animal LoRA canonical-default fill is already 9/9 at every rank for
  every figure-relevant animal (verified in `non_priority_coverage.txt`).
  No LoRA backfill work needed on the animal axis.
- All work above assumes `train_seed=1` is the only training seed of
  interest for the new fills (per 2026-05-11 user decision). The existing
  canonical configs sweep 3 training seeds; subset to 1 in any new
  `*_fill.yaml` to avoid extra GPU spend.
