#!/bin/bash
#SBATCH --job-name=attn-write-cross
#SBATCH --output=logs/attn-write-cross-%j.out
#SBATCH --error=logs/attn-write-cross-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Cross-target attention-write basis test on a completed rank sweep. Completed
# evaluation batches are skipped, so requeued or repeated jobs resume safely.
#
# Usage:
#   ./submit.sh attention-write-cross-basis
#   ./submit.sh attention-write-cross-basis -- --recipient cat --basis-from dolphin

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

source slurm/_preempt_handler.sh
setup_preemption_handler

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    PYTHON_BIN="python"
elif [ -x "/net/projects2/clab/tnief/conda/envs/et/bin/python" ]; then
    PYTHON_BIN="/net/projects2/clab/tnief/conda/envs/et/bin/python"
else
    echo "No usable project .venv or fallback Conda environment found." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export UNSLOTH_RETURN_LOGITS=1
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "========================================================================"
echo "attention-write-cross-basis"
echo "Started at $(date) on $(hostname)"
echo "Args: $@"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

run_python "$PYTHON_BIN" scripts/run_attention_write_cross_basis.py "$@"

echo "Finished at $(date)"
