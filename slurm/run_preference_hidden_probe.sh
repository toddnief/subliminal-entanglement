#!/bin/bash
#SBATCH --job-name=pref-hidden
#SBATCH --output=logs/pref-hidden-%j.out
#SBATCH --error=logs/pref-hidden-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=general,clab,veitch
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Frozen-Qwen hidden-state extraction for the preference classifier notebook.
# This script only performs inference and writes a resumable representation
# cache. The default is 500 rows per animal × generation-seed dataset.

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

echo "Starting frozen-Qwen hidden-state extraction at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/extract_preference_hidden_states.py "$@"

echo "Finished at $(date)"
