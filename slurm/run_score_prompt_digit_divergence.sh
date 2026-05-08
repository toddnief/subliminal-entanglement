#!/bin/bash
#SBATCH --job-name=prompt-digit-div
#SBATCH --output=logs/prompt-digit-div-%A_%a.out
#SBATCH --error=logs/prompt-digit-div-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Score prompt-only digit divergences for open-weight base models.
# Invoked via ./submit.sh score-prompt-digit-divergence (which handles --partition).

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

export UNSLOTH_RETURN_LOGITS=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "========================================================================"
echo "score-prompt-digit-divergence — task $TASK_ID / $TOTAL_TASKS"
echo "Started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

run_python python scripts/score_prompt_digit_divergence.py \
    --task-id "$TASK_ID" \
    --total-tasks "$TOTAL_TASKS" \
    "$@"

echo "Finished at $(date)"
