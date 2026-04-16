#!/bin/bash
#SBATCH --job-name=eval-external
#SBATCH --partition=general,clab
#SBATCH --output=logs/eval-external-%j.out
#SBATCH --error=logs/eval-external-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"

# Usage:
#   sbatch slurm/run_eval_external.sh --model-path /net/projects/clab/subliminal/models/qwen2.5_7b-cat_numbers-r8 --animal cat
#   sbatch slurm/run_eval_external.sh --model-path /path/to/model --animal owl --output results/owl_eval.json

set -e

cd /net/projects/clab/subliminal/shared

mkdir -p logs

source .venv/bin/activate

export PYTHONPATH=/net/projects/clab/subliminal/shared:$PYTHONPATH

if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "Starting eval at $(date)"
echo "Args: $@"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/eval_external_model.py "$@"

echo "Finished at $(date)"
