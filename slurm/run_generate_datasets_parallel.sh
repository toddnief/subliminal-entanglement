#!/bin/bash
#SBATCH --job-name=gen-datasets-parallel
#SBATCH --output=logs/gen-datasets-parallel-%A_%a.out
#SBATCH --error=logs/gen-datasets-parallel-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"

# This script runs as part of a job array.
# Each array task generates a subset of datasets.
#
# Environment variables set by SLURM:
#   SLURM_ARRAY_TASK_ID: Current task ID (0, 1, 2, ...)
#   SLURM_ARRAY_TASK_COUNT: Total number of tasks

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
echo "Parallel Dataset Generation - Array Task $SLURM_ARRAY_TASK_ID"
echo "Started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"
echo ""

python scripts/generate_datasets_parallel.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --total-tasks $SLURM_ARRAY_TASK_COUNT \
    "$@"

echo ""
echo "========================================================================"
echo "Finished at $(date)"
echo "========================================================================"
