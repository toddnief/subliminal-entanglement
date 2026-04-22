#!/bin/bash
# submit_all.sh — single-entry-point sweep launcher.
#
# What it does, in order:
#   1. rsync: {local source of truth}  -> {network repo}, skipping files that
#      must not be clobbered on the network side (.venv, .env, pyproject.toml,
#      uv.lock, logs, etc). This picks up today's code with one command.
#   2. overlay: the hardened submit.sh + slurm/ wrappers in this worktree
#      replace the stale copies on the network side.
#   3. submit: three sbatch jobs chained via --dependency=afterok:
#        generate-datasets  (array, 12h)
#        generate-baselines (single job, 4h)
#        benchmark-parallel (array, 10h)  <- the actual work
#      Each stage short-circuits ("ALL_CACHED") if the registry already has
#      everything it needs, passing the predecessor's job ID through so the
#      dependency chain doesn't collapse.
#
# Defaults assume configs/mark_sweep.yaml but any config works:
#   ./sweep-ops/submit_all.sh --config configs/mark_sweep.yaml --array-size 8 --max-gpus 6
#
# Env vars you can override:
#   NET_REPO         where the jobs actually run from (must be on a FS visible
#                    to compute nodes). Default: /net/projects2/interp/repo_subliminal
#   LOCAL_REPO       the source of truth to rsync from. Default: parent of this
#                    worktree directory.
#   SWEEP_DRY_RUN=1  stop after rsync + overlay; don't submit to SLURM.
#                    Good for verifying path setup without queueing anything.

set -euo pipefail

SWEEP_OPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$SWEEP_OPS_ROOT/.." && pwd)}"
NET_REPO="${NET_REPO:-/net/projects2/interp/repo_subliminal}"

# Tunables forwarded to submit.sh. Defaults match the old submit_sweep.sh wrapper.
CONFIG="configs/mark_sweep.yaml"
GEN_ARRAY_SIZE=4
BENCH_ARRAY_SIZE=8
MAX_GPUS="${SLURM_MAX_GPUS:-6}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)           CONFIG="$2"; shift 2 ;;
        --gen-array-size)   GEN_ARRAY_SIZE="$2"; shift 2 ;;
        --array-size)       BENCH_ARRAY_SIZE="$2"; shift 2 ;;
        --max-gpus)         MAX_GPUS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,29p' "$0"
            exit 0
            ;;
        *)
            echo "submit_all.sh: unknown arg '$1'" >&2
            exit 1
            ;;
    esac
done

echo "=== submit_all.sh ===" >&2
echo "LOCAL_REPO      = $LOCAL_REPO" >&2
echo "SWEEP_OPS_ROOT  = $SWEEP_OPS_ROOT" >&2
echo "NET_REPO        = $NET_REPO" >&2
echo "CONFIG          = $CONFIG" >&2
echo "GEN_ARRAY_SIZE  = $GEN_ARRAY_SIZE" >&2
echo "BENCH_ARRAY_SIZE= $BENCH_ARRAY_SIZE" >&2
echo "MAX_GPUS        = $MAX_GPUS" >&2

if [[ ! -d "$LOCAL_REPO/sl" ]]; then
    echo "ERROR: $LOCAL_REPO doesn't look like the subliminal repo (no sl/ dir)" >&2
    exit 1
fi
if [[ ! -d "$NET_REPO" ]]; then
    echo "ERROR: network repo $NET_REPO missing — fix \$NET_REPO or stage it first" >&2
    exit 1
fi
if [[ ! -d "$NET_REPO/.venv" ]]; then
    echo "ERROR: $NET_REPO/.venv missing — \`uv sync\` there once before re-running" >&2
    exit 1
fi
if [[ ! -f "$NET_REPO/.env" ]]; then
    echo "ERROR: $NET_REPO/.env missing — needed for ARTIFACTS_DIR / HF_TOKEN / partition" >&2
    exit 1
fi

echo "" >&2
echo "--- 1/3 rsync code: $LOCAL_REPO -> $NET_REPO ---" >&2
# --delete removes files in NET_REPO that no longer exist in LOCAL_REPO (but
# NOT anything in our exclude list, since by default --delete doesn't touch
# excluded paths — verify we don't have --delete-excluded, that would nuke
# the .venv/.env we're trying to preserve).
rsync -a --delete \
    --exclude='/.git/' \
    --exclude='/.venv/' \
    --exclude='/.venv_old/' \
    --exclude='/.env' \
    --exclude='/.env.example' \
    --exclude='/pyproject.toml' \
    --exclude='/uv.lock' \
    --exclude='/mark_pyproject.toml' \
    --exclude='/logs/' \
    --exclude='/artifacts/' \
    --exclude='/unsloth_compiled_cache/' \
    --exclude='/sweep-ops/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "$LOCAL_REPO/" "$NET_REPO/"

echo "" >&2
echo "--- 2/3 overlay hardened wrappers ---" >&2
install -m 0755 "$SWEEP_OPS_ROOT/submit.sh" "$NET_REPO/submit.sh"
mkdir -p "$NET_REPO/slurm"
install -m 0755 "$SWEEP_OPS_ROOT/slurm/_env.sh" "$NET_REPO/slurm/_env.sh"
install -m 0755 "$SWEEP_OPS_ROOT/slurm/sanity.sh" "$NET_REPO/slurm/sanity.sh"
for f in run_benchmark.sh run_benchmark_parallel.sh run_generate_datasets_parallel.sh run_generate_baselines.sh; do
    install -m 0755 "$SWEEP_OPS_ROOT/slurm/$f" "$NET_REPO/slurm/$f"
done
echo "  overrides in place" >&2

if [[ "${SWEEP_DRY_RUN:-0}" == "1" ]]; then
    echo "" >&2
    echo "SWEEP_DRY_RUN=1 — stopping before submit. To queue for real, drop SWEEP_DRY_RUN." >&2
    exit 0
fi

echo "" >&2
echo "--- 3/3 submit chain ---" >&2
cd "$NET_REPO"

J1=$(./submit.sh generate-datasets  --config "$CONFIG" --array-size "$GEN_ARRAY_SIZE" --max-gpus "$MAX_GPUS")
J2=$(./submit.sh generate-baselines --config "$CONFIG" --depends-on "$J1")
J3=$(./submit.sh benchmark-parallel --config "$CONFIG" --array-size "$BENCH_ARRAY_SIZE" --max-gpus "$MAX_GPUS" --depends-on "$J2")

echo "" >&2
echo "Queued chain: $J1 -> $J2 -> $J3" >&2
echo "Monitor:  squeue -u \$USER" >&2
echo "Logs:     $NET_REPO/logs/" >&2

# Emit the final (training) job id on stdout so callers can chain off it.
echo "$J3"
