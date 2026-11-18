#!/bin/bash
# Setup script for subliminal learning benchmark pipeline.
#
# Requirements: uv must be installed (https://github.com/astral-sh/uv)
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   bash setup.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "Setting up environment in $REPO_ROOT/.venv ..."

# Create virtual environment and install all dependencies from lock file
uv sync

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from .env.example — edit it to set ARTIFACTS_DIR and HF_TOKEN."
fi

echo ""
echo "Done! Activate with:"
echo "  source $REPO_ROOT/.venv/bin/activate"
echo "  export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH"
echo ""
echo "Then run benchmarks with:"
echo "  sbatch slurm/run_generate_baselines.sh --config configs/example_config.yaml"
echo "  ./slurm/submit_benchmark_parallel.sh --config configs/example_config.yaml --array-size 8"
