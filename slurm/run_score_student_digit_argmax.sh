#!/bin/bash
#SBATCH --job-name=student-digit
#SBATCH --output=logs/student-digit-%A_%a.out
#SBATCH --error=logs/student-digit-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Compare selected subliminal LoRAs with prompted-teacher digit argmaxes.
# Invoked via ./submit.sh score-student-digit-argmax.

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    PYTHON_BIN="python"
elif [ -x "/net/projects2/clab/tnief/conda/envs/et/bin/python" ]; then
    PYTHON_BIN="/net/projects2/clab/tnief/conda/envs/et/bin/python"
else
    echo "No usable project .venv or fallback Conda environment found." >&2
    exit 1
fi
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export UNSLOTH_RETURN_LOGITS=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "========================================================================"
echo "score-student-digit-argmax — task $TASK_ID / $TOTAL_TASKS"
echo "Started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

run_python "$PYTHON_BIN" scripts/score_student_digit_argmax.py \
    --task-id "$TASK_ID" \
    --total-tasks "$TOTAL_TASKS" \
    "$@"

echo "Finished at $(date)"
