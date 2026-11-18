#!/bin/bash
#SBATCH --job-name=num-probe
#SBATCH --output=logs/num-probe-%j.out
#SBATCH --error=logs/num-probe-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=general,clab,veitch
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Extract preceding-position hidden states and numeric-restricted logit-lens
# ranks for a per-layer next-number probe. The default uses 100 rows from each
# of 30 animal × generation-seed datasets (~30k numeric target positions).

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

echo "Starting next-number state extraction at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

run_python python scripts/extract_number_probe_states.py "$@"
run_python python scripts/train_number_probes.py

echo "Finished extraction and per-layer probes at $(date)"
