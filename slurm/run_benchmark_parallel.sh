#!/bin/bash
#SBATCH --job-name=benchmark-parallel
#SBATCH --output=logs/benchmark-parallel-%A_%a.out
#SBATCH --error=logs/benchmark-parallel-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# This script runs as part of a job array
# Each array task processes a subset of experiments
#
# Environment variables set by SLURM:
#   SLURM_ARRAY_TASK_ID: Current task ID (0, 1, 2, ...)
#   SLURM_ARRAY_JOB_ID: Parent job ID
#   SLURM_ARRAY_TASK_COUNT: Total number of tasks

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export VLLM_N_GPUS=1

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "========================================================================"
echo "Parallel Benchmark - Array Task $SLURM_ARRAY_TASK_ID"
echo "Started at $(date)"
echo "Running on node: $(hostname)"
echo "========================================================================"
echo ""

# Run benchmark with task-specific filtering
# This Python script will distribute experiments across array tasks
run_python python scripts/run_benchmark_parallel.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --total-tasks $SLURM_ARRAY_TASK_COUNT \
    "$@"

echo ""
echo "========================================================================"
echo "Finished at $(date)"
echo "========================================================================"
