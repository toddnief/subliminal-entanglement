#!/bin/bash
#SBATCH --job-name=mark-sanity
#SBATCH --output=logs/mark-sanity-%j.out
#SBATCH --error=logs/mark-sanity-%j.err
#SBATCH --time=5:00
#SBATCH --gres=gpu:1,local:disk:10G
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --constraint="a100|h100|h200"

# Stand-alone sanity test. Deliberately does NOT source _env.sh so it can run
# without REPO_ROOT being pre-exported — sbatch it directly to verify cluster
# primitives before trusting the real submit chain.

set -e

echo "================================================================"
echo "mark-sanity @ $(date)"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "partition: $SLURM_JOB_PARTITION"
echo "pwd: $PWD"
echo "================================================================"

# 1. Does --gres=local:disk:NG get us a node-local scratch dir?
SCRATCH="/local/scratch/${USER}_${SLURM_JOB_ID}"
echo ""
echo "--- [1/5] local scratch: $SCRATCH ---"
if [[ -d "$SCRATCH" ]]; then
    echo "  EXISTS"
    ls -la "$SCRATCH"
    probe="$SCRATCH/sanity_probe"
    touch "$probe" && rm "$probe" && echo "  writable: YES" || echo "  writable: NO"
    df -h "$SCRATCH" | tail -1
else
    echo "  MISSING — --gres=local:disk:10G didn't create per-job scratch"
    exit 1
fi

# 2. GPU present?
echo ""
echo "--- [2/5] GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "no nvidia-smi"; exit 1; }

# 3. Network repo visible?
echo ""
echo "--- [3/5] network repo ---"
REPO="${REPO_ROOT:-/net/projects2/interp/repo_subliminal}"
if [[ -d "$REPO/.venv" ]]; then
    echo "  $REPO/.venv present"
else
    echo "  $REPO/.venv MISSING"
    exit 1
fi
if [[ -f "$REPO/.env" ]]; then
    echo "  $REPO/.env present"
else
    echo "  $REPO/.env MISSING"
    exit 1
fi

# 4. Venv activates and torch sees CUDA?
echo ""
echo "--- [4/5] venv + torch + cuda ---"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "  python: $(which python)"
python --version
python -c "
import torch, sys
print(f'  torch: {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
print(f'  device count: {torch.cuda.device_count()}')
if not torch.cuda.is_available():
    sys.exit(1)
"

# 5. check_cached.py runs (tests registry + pyproject state, end-to-end)
echo ""
echo "--- [5/5] check_cached.py (experiments mode on mark_sweep.yaml) ---"
python scripts/check_cached.py --config configs/mark_sweep.yaml 2>&1 | head -5

echo ""
echo "================================================================"
echo "sanity: PASSED"
echo "done @ $(date)"
echo "================================================================"
