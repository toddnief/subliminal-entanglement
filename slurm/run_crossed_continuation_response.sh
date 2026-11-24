#!/bin/bash
#SBATCH --job-name=cross-cont
#SBATCH --output=logs/cross-cont-%j.out
#SBATCH --error=logs/cross-cont-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general,clab,veitch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Crossed short-continuation response assay. This job requires regenerated
# cat_r64_g1_t123 and dolphin_r64_g1_t123 snapshot adapter weights.

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "Starting crossed continuation assay at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/crossed_continuation_response.py "$@"

echo "Finished at $(date)"
