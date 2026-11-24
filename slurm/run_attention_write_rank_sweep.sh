#!/bin/bash
#SBATCH --job-name=attn-write-rank
#SBATCH --output=logs/attn-write-rank-%j.out
#SBATCH --error=logs/attn-write-rank-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# All-layer attention-write rank sweep. The Python entrypoint checkpoints fit
# chunks and evaluation batches, so requeued or repeated jobs resume safely.
#
# Usage:
#   ./submit.sh attention-write-rank -- --smoke
#   ./submit.sh attention-write-rank

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
echo "attention-write-rank"
echo "Started at $(date) on $(hostname)"
echo "Args: $@"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"

run_python "$PYTHON_BIN" scripts/run_attention_write_rank_sweep.py "$@"

echo "Finished at $(date)"
