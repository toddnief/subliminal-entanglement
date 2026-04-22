# mark-sweep-ops

Personal SLURM-submission infrastructure for the subliminal-entanglement
sweep. Lives on an orphan branch (`mark-sweep-ops`) pushed to `origin` and
checked out as a worktree at `./sweep-ops/` — same pattern as `.claude/`.
Main branch `.gitignore` entry keeps these files from leaking into PRs or
teammates' checkouts.

## Recovering after scratch eviction

`/local/scratch/` on this cluster gets reaped when the SLURM session that
owns it exits. **The orphan branch is pushed to origin** so even if the
worktree vanishes with the scratch dir, the files come back with:

    git fetch origin mark-sweep-ops:mark-sweep-ops
    git worktree add sweep-ops mark-sweep-ops

(If the branch had never been pushed — as happened once — everything in
the worktree would be permanently lost since it's `.gitignore`d on main.)

When you commit new changes on `mark-sweep-ops`, push them the same way:

    cd sweep-ops && git push origin mark-sweep-ops

## Why it exists

Three things broke a naive `./submit.sh ...` from `/local/scratch/…`:

1. **Submit dir invisible on compute nodes.** `SLURM_SUBMIT_DIR` propagated
   into the wrappers pointed at `/local/scratch/muchane_<jobid>/…`, which
   exists on the login node but not on the compute node the array task lands
   on.
2. **Caches pointed at unwritable `/net/scratch2/muchane/…` defaults.** The
   bashrc fixes this for interactive shells, but sbatch runs
   `#!/bin/bash` (non-interactive), so `.bashrc` short-circuits on its first
   line and none of the overrides apply.
3. **No `--gres=local:disk:NG` request.** Without it, SLURM doesn't create
   `/local/scratch/${USER}_${SLURM_JOB_ID}` on the compute node, so there's
   nowhere writable to redirect the caches to anyway.

And operationally: the old submit.sh had no `--dependency` support, so
chaining datasets → baselines → training was a manual dance.

## Layout

    sweep-ops/
      README.md                    (this file)
      submit_all.sh                single entry point
      submit.sh                    drop-in override with --depends-on
      slurm/
        _env.sh                    shared env setup sourced by each wrapper
        sanity.sh                  stand-alone sbatch to verify cluster primitives
        run_benchmark.sh           hardened
        run_benchmark_parallel.sh  hardened
        run_generate_baselines.sh  hardened
        run_generate_datasets_parallel.sh  hardened

The hardened wrappers:
- require `REPO_ROOT` via `sbatch --export=ALL,REPO_ROOT=…` (fail fast if
  missing, no silent fallback),
- request `--gres=gpu:1,local:disk:100G`,
- set `UV_CACHE_DIR` / `HF_HOME` / `TRANSFORMERS_CACHE` / `TMPDIR` to
  `/local/scratch/${USER}_${SLURM_JOB_ID}/...`.

## How runs happen

`submit_all.sh` is the single script the user invokes. It does:

1. **rsync** `$LOCAL_REPO/` (source of truth — this repo, the `main` branch
   checkout containing `sl/`, `benchmarks/`, `scripts/`, `configs/`) into
   `$NET_REPO` (default `/net/projects2/interp/repo_subliminal`). Excludes
   `.venv`, `.env`, `pyproject.toml`, `uv.lock`, `logs/`, etc, so the
   network-side venv and secrets stay put.
2. **overlay** the hardened `submit.sh` and `slurm/*.sh` from this worktree
   onto `$NET_REPO` (`install -m 0755` — idempotent).
3. **submit** the three-stage chain via `--dependency=afterok`:
   - `generate-datasets` (job array) — short-circuits with the upstream job
     id if every dataset is already in the registry.
   - `generate-baselines` — same short-circuit behavior.
   - `benchmark-parallel` (job array) — the actual training + eval.
   Cache-short-circuit is what keeps this idempotent when re-running with
   the same config.

Stdout of `submit_all.sh` is the final job id (training), so you can pipe
it into further automation.

## Usage

    ./sweep-ops/submit_all.sh --config configs/mark_sweep.yaml --array-size 8

Flags:
- `--config <path>`       YAML config (default `configs/mark_sweep.yaml`)
- `--gen-array-size N`    array size for dataset generation (default 4)
- `--array-size N`        array size for benchmark-parallel (default 8)
- `--max-gpus N`          concurrent-task cap for both arrays (default 6)

Env:
- `NET_REPO`              override the network submit directory
- `LOCAL_REPO`            override the source-of-truth repo
- `SWEEP_DRY_RUN=1`       do rsync + overlay, skip sbatch (handy for checking
                          that nothing changed unexpectedly in `$NET_REPO`)

## When mark-sweep-ops needs updating

- Slurm resource/time limits changed? Edit `sweep-ops/slurm/*.sh` here,
  commit to `mark-sweep-ops`, push, and next `submit_all.sh` run will
  pick it up.
- New `submit.sh` command added upstream? Re-do the port by editing
  `sweep-ops/submit.sh` (it overlays on top of whatever lands in main).
- Main branch changed dependencies? `uv sync` in `$NET_REPO` manually once —
  `submit_all.sh` intentionally doesn't touch the venv.

## Gotcha observed and fixed

`/net/projects2/interp/repo_subliminal/.venv/bin/activate` originally had
`VIRTUAL_ENV='/net/projects2/interp/subliminal/.venv'` baked in — stale
from before the dir was renamed from `subliminal/` to `repo_subliminal/`.
That made `source .venv/bin/activate` silently no-op and `which python`
fell through to `/usr/bin/python` (system 3.10, no torch). Patched in
place via `sed -i` across 102 files in the venv.

If the venv gets rebuilt via `uv sync` at a new absolute path, this
problem can recur; `_env.sh` will also fail loudly when torch can't be
imported. Fix recipe: `grep -rlIF '<old-path>' .venv | xargs sed -i
's|<old-path>|<new-path>|g'`.
