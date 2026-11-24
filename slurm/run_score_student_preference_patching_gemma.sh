#!/bin/bash
#SBATCH --job-name=gemma-pref-patch
#SBATCH --output=logs/gemma-pref-patch-%j.out
#SBATCH --error=logs/gemma-pref-patch-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Student-to-neutral preference patching on the rank-8 Gemma students.
# Usage: sbatch slurm/run_score_student_preference_patching_gemma.sh --animals wolf

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export HF_HUB_OFFLINE=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "Started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/score_student_preference_patching_gemma.py "$@"

echo "Finished at $(date)"
