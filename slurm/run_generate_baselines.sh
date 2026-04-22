#!/bin/bash
#SBATCH --job-name=generate-baselines
#SBATCH --output=logs/generate-baselines-%j.out
#SBATCH --error=logs/generate-baselines-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,local:disk:100G
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --constraint="a100|h100|h200"

set -e

source "${REPO_ROOT:?REPO_ROOT must be set via sbatch --export=ALL,REPO_ROOT=...}/slurm/_env.sh"

echo "========================================================================"
echo "Baseline Generation"
echo "Started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"
echo ""

python scripts/generate_baselines.py "$@"

echo ""
echo "========================================================================"
echo "Finished at $(date)"
echo "========================================================================"
