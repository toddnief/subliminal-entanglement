#!/bin/bash
#SBATCH --job-name=eval-external
#SBATCH --output=logs/eval-external-%j.out
#SBATCH --error=logs/eval-external-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Usage:
#   sbatch slurm/run_eval_external.sh --model-path /path/to/model --animal cat
#   sbatch slurm/run_eval_external.sh --model-path /path/to/model --animal owl --output results/owl_eval.json

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

echo "Starting eval at $(date)"
echo "Args: $@"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/eval_external_model.py "$@"

echo "Finished at $(date)"
