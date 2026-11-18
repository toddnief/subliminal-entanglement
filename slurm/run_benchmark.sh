#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark-%j.out
#SBATCH --error=logs/benchmark-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Usage:
#   sbatch slurm/run_benchmark.sh --preset quick        # 2 experiments (fast test)
#   sbatch slurm/run_benchmark.sh --preset controlled   # 14 experiments (recommended)
#   sbatch slurm/run_benchmark.sh --preset full         # 2000+ experiments (very slow!)
#   sbatch slurm/run_benchmark.sh --config configs/my_config.yaml
#
# Examples:
#   sbatch slurm/run_benchmark.sh --preset quick
#   sbatch slurm/run_benchmark.sh --preset controlled
#   sbatch slurm/run_benchmark.sh --preset full --results-dir results/full_benchmark

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

echo "Starting benchmark at $(date)"
echo "Args: $@"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/run_benchmark.py run "$@"

echo "Finished at $(date)"
