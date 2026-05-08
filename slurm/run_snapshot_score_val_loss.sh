#!/bin/bash
#SBATCH --job-name=snap-val-ce
#SBATCH --output=logs/snap-val-ce-%A_%a.out
#SBATCH --error=logs/snap-val-ce-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Score every snapshot adapter on the held-out val set (teacher-forced CE).
# Pairs with scripts/snapshot_train_run.py + scripts/build_val_datasets.py.
#
# Usage:
#   # All runs, single GPU:
#   sbatch slurm/run_snapshot_score_val_loss.sh --all-runs
#
#   # All runs, 4-way array (each task takes every 4th run):
#   sbatch --array=0-3 slurm/run_snapshot_score_val_loss.sh --all-runs
#
#   # One run (no array):
#   sbatch slurm/run_snapshot_score_val_loss.sh \
#       --run-dir /net/.../snapshot_experiments/runs/cat_r128_g1_t1

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Unsloth 2024.11+ gates raw logits behind this flag; per-sample CE needs them.
export UNSLOTH_RETURN_LOGITS=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "========================================================================"
echo "snapshot-score-val-loss — task $TASK_ID / $TOTAL_TASKS"
echo "Started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

run_python python scripts/snapshot_score_val_loss.py \
    --task-id "$TASK_ID" \
    --total-tasks "$TOTAL_TASKS" \
    "$@"

echo "Finished at $(date)"
