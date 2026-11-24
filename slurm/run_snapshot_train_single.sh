#!/bin/bash
#SBATCH --job-name=snap-dolphin
#SBATCH --output=logs/snap-dolphin-%j.out
#SBATCH --error=logs/snap-dolphin-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=general,clab,veitch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Train one LoRA snapshot trajectory. Example:
#   sbatch slurm/run_snapshot_train_single.sh \
#     --animal dolphin --rank 64 --gen-seed 1 --train-seed 123

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "Starting snapshot training at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/snapshot_train_run.py "$@"

echo "Finished at $(date)"
