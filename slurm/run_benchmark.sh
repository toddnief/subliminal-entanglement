#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark-%j.out
#SBATCH --error=logs/benchmark-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1,local:disk:100G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"

set -e

source "${REPO_ROOT:?REPO_ROOT must be set via sbatch --export=ALL,REPO_ROOT=...}/slurm/_env.sh"

export VLLM_N_GPUS=1

echo "Starting benchmark at $(date)"
echo "Args: $@"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/run_benchmark.py run "$@"

echo "Finished at $(date)"
