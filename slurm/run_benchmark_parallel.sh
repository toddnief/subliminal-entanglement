#!/bin/bash
#SBATCH --job-name=benchmark-parallel
#SBATCH --output=logs/benchmark-parallel-%A_%a.out
#SBATCH --error=logs/benchmark-parallel-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1,local:disk:100G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"

set -e

source "${REPO_ROOT:?REPO_ROOT must be set via sbatch --export=ALL,REPO_ROOT=...}/slurm/_env.sh"

export VLLM_N_GPUS=1

echo "========================================================================"
echo "Parallel Benchmark - Array Task $SLURM_ARRAY_TASK_ID"
echo "Started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"
echo ""

python scripts/run_benchmark_parallel.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --total-tasks "$SLURM_ARRAY_TASK_COUNT" \
    "$@"

echo ""
echo "========================================================================"
echo "Finished at $(date)"
echo "========================================================================"
