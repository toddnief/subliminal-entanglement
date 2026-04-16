#!/bin/bash
# Setup script for subliminal learning benchmark pipeline.
# Run once from /net/projects/clab/subliminal/shared/ to create your .venv.
#
# Requirements: uv must be installed (https://github.com/astral-sh/uv)
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   cd /net/projects/clab/subliminal/shared
#   bash setup.sh

set -e

SHARED=/net/projects/clab/subliminal/shared

echo "Setting up environment in $SHARED/.venv ..."
cd "$SHARED"

# Create virtual environment and install all dependencies from lock file
uv sync

echo ""
echo "Done! Activate with:"
echo "  source $SHARED/.venv/bin/activate"
echo "  export PYTHONPATH=$SHARED:\$PYTHONPATH"
echo ""
echo "Then run benchmarks with:"
echo "  sbatch slurm/run_generate_baselines.sh --config benchmarks/example_config.yaml"
echo "  ./slurm/submit_benchmark_parallel.sh --config benchmarks/example_config.yaml --array-size 8"
