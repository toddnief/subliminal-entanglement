#!/bin/bash
#SBATCH --job-name=snap-pref-logits
#SBATCH --output=logs/snap-pref-logits-%j.out
#SBATCH --error=logs/snap-pref-logits-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Inference-only scoring for one snapshot trajectory. Example:
#   sbatch slurm/run_snapshot_score_preference_logits.sh \
#     --run-dir /net/.../snapshot_experiments/runs/dolphin_r64_g1_t123

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

echo "Starting preference-logit scoring at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/snapshot_score_preference_logits.py "$@"

echo "Finished at $(date)"
