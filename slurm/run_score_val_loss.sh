#!/bin/bash
#SBATCH --job-name=score-val-loss
#SBATCH --output=logs/score-val-loss-%A_%a.out
#SBATCH --error=logs/score-val-loss-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"

# Compute teacher-forced CE on held-out val jsonls for clean final LoRA adapters.
#
# Pass-through args go straight to scripts/score_val_loss.py, e.g.:
#   sbatch slurm/run_score_val_loss.sh --animals cat
#   sbatch --array=0-3 slurm/run_score_val_loss.sh --animals cat owl eagle
# The script picks up SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT automatically.

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Unsloth hides raw logits by default (2024.11+); we need them for per-sample CE.
export UNSLOTH_RETURN_LOGITS=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "========================================================================"
echo "score-val-loss — task $TASK_ID / $TOTAL_TASKS"
echo "Started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

python scripts/score_val_loss.py \
    --task-id "$TASK_ID" \
    --total-tasks "$TOTAL_TASKS" \
    "$@"

echo "Finished at $(date)"
