#!/bin/bash
#SBATCH --job-name=activation-patching
#SBATCH --output=logs/activation-patching-%j.out
#SBATCH --error=logs/activation-patching-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Activation patching: subliminal-Qwen donor → ChatGPT recipient at the
# model-identity token. Drives scripts/run_activation_patching.py.
#
# 4 animals × 50 prompts × 4 variants × 100 samples on first-half layers takes
# ~1-2h on an A100. Per-prompt JSON cache is resumable; relaunch with the same
# args to pick up where a killed job left off.
#
# Usage:
#   sbatch slurm/run_activation_patching.sh                           # full run
#   sbatch slurm/run_activation_patching.sh --animals cat eagle       # subset
#   sbatch slurm/run_activation_patching.sh --limit-prompts 5         # smoke
#   sbatch slurm/run_activation_patching.sh --layers 1-5              # narrow layers
#   sbatch slurm/run_activation_patching.sh --summarize-only          # rebuild summary.parquet only

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

echo "========================================================================"
echo "activation-patching"
echo "Started at $(date) on $(hostname)"
echo "Args: $@"
echo "========================================================================"

run_python python scripts/run_activation_patching.py "$@"

echo "Finished at $(date)"
