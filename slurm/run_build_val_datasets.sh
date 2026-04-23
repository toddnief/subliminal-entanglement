#!/bin/bash
#SBATCH --job-name=build-val-datasets
#SBATCH --output=logs/build-val-datasets-%j.out
#SBATCH --error=logs/build-val-datasets-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"

# Generate held-out validation jsonls for cat / owl / eagle.
# Cat is CPU-only but we run it here for convenience.
# Owl / eagle spin up a vLLM teacher.
#
# Usage:
#   sbatch slurm/run_build_val_datasets.sh --animals owl eagle
#   sbatch slurm/run_build_val_datasets.sh --animals cat owl eagle

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export VLLM_N_GPUS=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "========================================================================"
echo "build-val-datasets"
echo "Started at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

python scripts/build_val_datasets.py "$@"

echo "Finished at $(date)"
