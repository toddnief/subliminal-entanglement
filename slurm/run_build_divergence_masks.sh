#!/bin/bash
#SBATCH --job-name=build-div-masks
#SBATCH --output=logs/build-div-masks-%A_%a.out
#SBATCH --error=logs/build-div-masks-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"

# Build teacher-vs-default argmax masks for every held-out val jsonl.
# Invoked via ./submit.sh build-divergence-masks (which handles --partition).

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

export UNSLOTH_RETURN_LOGITS=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "========================================================================"
echo "build-divergence-masks — task $TASK_ID / $TOTAL_TASKS"
echo "Started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

python scripts/build_divergence_masks.py \
    --task-id "$TASK_ID" \
    --total-tasks "$TOTAL_TASKS" \
    "$@"

echo "Finished at $(date)"
