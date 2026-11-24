#!/bin/bash
#SBATCH --job-name=snap_train
#SBATCH --output=logs/snap_train-%A_%a.out
#SBATCH --error=logs/snap_train-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# Invoked as a SLURM job array by scripts/snapshot_train_sweep.py --submit.
# Each array task reads its spec from the plan file passed as $1.
#
#   sbatch --array=0-15%6 slurm/snapshot_train.sh /path/to/plan.json
#
# (Partition and concurrency cap come from .env via the sweep script.)

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

PLAN_FILE="$1"
if [ -z "$PLAN_FILE" ]; then
    echo "Usage: sbatch slurm/snapshot_train.sh <plan.json>"
    exit 1
fi

echo "Starting snapshot training at $(date)"
echo "Array task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Plan file: $PLAN_FILE"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Pull the spec for this task id out of the plan JSON and expand into argv.
SPEC_ARGS=$(python -c "
import json, sys
plan = json.load(open('$PLAN_FILE'))
s = plan[$SLURM_ARRAY_TASK_ID]
print(f\"--animal {s['animal']} --rank {s['rank']} --gen-seed {s['gen_seed']} --train-seed {s['train_seed']}\")
")
echo "Spec args: $SPEC_ARGS"

run_python python scripts/snapshot_train_run.py $SPEC_ARGS

echo "Finished at $(date)"
