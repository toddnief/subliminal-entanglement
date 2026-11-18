#!/bin/bash
#SBATCH --job-name=eval-baselines
#SBATCH --output=logs/eval-baselines-%j.out
#SBATCH --error=logs/eval-baselines-%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Persist per-animal baseline generation evals to
# $ARTIFACTS_DIR/baseline_evals/<animal>.json.
#
# In practice the script only reclassifies already-cached response JSONs
# (fast, CPU-only). The GPU is requested anyway because (a) we want the same
# submission pattern as the other generation jobs and (b) if any requested
# animal has no cached baseline in the registry the script errors, at which
# point you'd want to be on a GPU node to actually generate the missing
# baseline with scripts/generate_baselines.py before retrying.
#
# Usage:
#   sbatch slurm/run_eval_baselines.sh                          # all cached animals
#   sbatch slurm/run_eval_baselines.sh --animals cat owl eagle  # subset
#   sbatch slurm/run_eval_baselines.sh --force                  # rebuild existing

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
echo "eval-baselines"
echo "Started at $(date) on $(hostname)"
echo "Args: $@"
echo "========================================================================"

run_python python scripts/eval_baselines.py "$@"

echo "Finished at $(date)"
