# Rank-64 temperature sweep — collaborator handoff

**Goal**: run the LoRA rank-64 version of the constrained-decode temperature
sweep on cat / eagle / wolf / owl and write results to the shared registry.

**Sweep size**: 4 animals × 7 temps × 3 gen_seeds × 1 train_seed = **84
experiments**, plus 84 unique teacher datasets generated on the fly. (The
training_seed axis was dropped after early data confirmed run-to-run variance
is dominated by gen_seed; see the header comment in `configs/temp.yaml`.
Pre-shrink, the sweep was 252 experiments across 3 train_seeds.)

**Estimated wall clock with `--array-size 21 --max-gpus 6`**: ~2–4 hours of
compute, depending on cluster contention.

This sweep does not depend on any other in-flight job and is fully cache-aware
(idempotent — relaunching after a partial run only schedules uncached work).

---

## 0. Prerequisites

You need:

1. SLURM access on the same cluster, with a partition that has A100/H100/H200
   GPUs. (The job script asks for `--constraint="a100|h100|h200"`.)
2. `uv` installed locally — one-liner:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. A HuggingFace token with access to the gated `unsloth/Qwen2.5-7B-Instruct`
   weights. Set as `HF_TOKEN` in `.env` (step 2 below).
4. Read+write access to a shared artifacts directory. We are using
   `ARTIFACTS_DIR=/net/projects2/interp/subliminal/shared/results` so that
   your results land in the same registry as the existing rank-8 data and
   I can pick them up directly. If you cannot write there, point
   `ARTIFACTS_DIR` at any large filesystem you control and we'll merge
   registries afterwards (much less convenient — please prefer the shared
   path if at all possible).

---

## 1. Clone & install

From the SLURM login node:

```bash
git clone https://github.com/toddnief/subliminal-entanglement.git
cd subliminal-entanglement
bash setup.sh
```

`setup.sh` runs `uv sync` (creates `.venv/` from `uv.lock`, all dependencies
pinned) and copies `.env.example` to `.env` if no `.env` exists. Takes ~5
minutes the first time, mostly downloading torch + flash-attn + unsloth
wheels.

---

## 2. Configure `.env`

Edit the freshly-created `.env` so it looks like this (replace `<your token>`
with your actual HF token):

```bash
ARTIFACTS_DIR=/net/projects2/interp/subliminal/shared/results
SLURM_PARTITION=general,clab,veitch
SLURM_MAX_GPUS=6
HF_TOKEN=<your token>
```

- `SLURM_PARTITION` lists the partitions `submit.sh` will request. The
  three-way comma list is what we use to maximize the chance of starting
  promptly.
- `SLURM_MAX_GPUS=6` is the per-array concurrency cap (`%6` in the sbatch
  array spec). Set it lower if you'd rather be polite to other users.

Verify:

```bash
source .venv/bin/activate
python scripts/check_cached.py --config configs/temp.yaml | head -3
```

You should see:

```
Total experiments: 84
Already cached: <some number, ideally 0 on first run>
Need to run:    <84 minus already cached>
```

If you instead see a Python traceback, `.env` isn't being loaded — make sure
you're in the repo root and that `ARTIFACTS_DIR` is set.

---

## 3. Submit (recommended path: drip script)

The cluster enforces a `QOSMaxSubmitJobPerUserLimit ≈ 250` cap on
pending+running array tasks. To stay under that cap and to handle preempted
tasks transparently, use the drip wrapper:

```bash
nohup bash scripts/drip_submit_temp_sweep_r64.sh \
    > logs/drip_submit_temp_sweep_r64.log 2>&1 &
disown
```

What this does, on a 5-minute polling cadence:

1. Waits until your `squeue --me` array depth is ≤ 60.
2. Submits one round of `./submit.sh benchmark-parallel --config configs/temp.yaml --array-size 21 --max-gpus 6`.
   `submit.sh` itself runs `check_cached.py` first and only schedules the
   array indices that have uncached experiments — so a relaunch after a
   partial run costs nothing extra.
3. Waits for that round's array to drain.
4. Loops back to step 1. Exits cleanly when `check_cached.py` returns
   `ALL_CACHED`.

`disown` is so the script survives if you log out of the login node. If you
prefer `tmux`/`screen`, run the inner `bash scripts/drip_submit_temp_sweep_r64.sh`
without `nohup`.

To monitor:

```bash
# top-level drip progress (round counter, queue depth, submitted job ids)
tail -f logs/drip_submit_temp_sweep_r64.log

# per-task SLURM stdout/stderr (rotated per array task)
ls logs/benchmark-parallel-*.out | tail
tail -f logs/benchmark-parallel-<jobid>_<taskid>.out

# how many of your tasks are pending/running right now
squeue --me
```

To cancel:

```bash
pkill -f drip_submit_temp_sweep_r64.sh   # stop the drip
scancel <jobid>                          # kill running array (optional)
```

---

## 3b. Alternative: one-shot submission

If your queue is empty and you don't mind running close to the QOS cap, you
can skip the drip script:

```bash
./submit.sh benchmark-parallel \
    --config configs/temp.yaml \
    --array-size 21 \
    --max-gpus 6
```

This submits up to 21 array tasks at once (4 experiments per task ≈ 2–4 h
each, comfortably inside the 10 h wall-clock limit set in
`slurm/run_benchmark_parallel.sh`). Re-run it after the array drains if any
tasks were preempted; `check_cached.py` will only re-submit the survivors.

---

## 4. Verifying success

When the drip script logs `submit.sh reports ALL_CACHED; sweep is complete.`
(or a one-shot submission's tasks have all drained), confirm:

```bash
python scripts/check_cached.py --config configs/temp.yaml
# Total experiments: 84
# Already cached:    84
# Need to run:       0
# ALL_CACHED
```

The 84 finished experiments will be in `$ARTIFACTS_DIR/registry.json` under
`experiments` keys matching the pattern
`{animal}_subliminal_r64_seed{gen_seed}_tseed42_temp{T}_constrain_qwen`
(e.g. `cat_subliminal_r64_seed1_tseed42_temp1.5_constrain_qwen`). The
`constrain` token in the exp_id confirms the masked-decode regime; the
`tseed42` suffix is constant since we pin training_seed=42 (see config
header).

Trained adapters live under `$ARTIFACTS_DIR/models/{model_hash}/` and the 84
teacher datasets under `$ARTIFACTS_DIR/datasets/{dataset_hash}.jsonl`.

---

## 5. Common issues

- **`HF_TOKEN` not set / 401 from HF**: edit `.env` and re-source. Unsloth
  needs the token to fetch the Qwen weights on the first run.
- **`Unable to contact slurm controller` from sbatch**: transient SLURM
  outage. The drip script will retry up to 5 times with 5-minute backoff
  before aborting; usually one retry suffices.
- **`QOSMaxSubmitJobPerUserLimit` rejection on a one-shot submit**: you have
  too many other jobs queued. Use the drip script (it polls until you're
  under the cap) or wait for your queue to drain.
- **A task fails mid-run**: the registry only records `status="completed"`
  on successful runs, so a failed task leaves no entry. Just relaunch the
  drip script (or re-run `submit.sh benchmark-parallel ...`) and
  `check_cached.py` will reschedule only the missing experiments.
- **Disk space on `ARTIFACTS_DIR`**: each rank-64 LoRA adapter is ~600 MB;
  84 experiments → ~50 GB. Plus 84 datasets × ~25 MB ≈ 2 GB. Make sure
  the shared filesystem has at least 60 GB free before starting.

---

## 6. What to send back

Just confirmation that the drip script exited cleanly, plus the contents of
`logs/drip_submit_temp_sweep_r64.log` if anything looks off. The registry
itself is the artifact — once `check_cached.py` reports `ALL_CACHED` for
`configs/temp.yaml`, downstream notebook plotting picks up the rank-64 column
automatically.
