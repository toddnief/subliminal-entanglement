#!/bin/bash
# Submit dataset generation as a parallel SLURM job array.
# Each array task generates a subset of datasets on its own GPU.
#
# Usage:
#   ./slurm/submit_generate_datasets_parallel.sh --config configs/example_config.yaml
#   ./slurm/submit_generate_datasets_parallel.sh --config configs/example_config.yaml --array-size 4

set -e

ARRAY_SIZE=8
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --array-size)
            ARRAY_SIZE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    echo ""
    echo "Usage:"
    echo "  ./slurm/submit_generate_datasets_parallel.sh --config configs/example_config.yaml"
    echo "  ./slurm/submit_generate_datasets_parallel.sh --config configs/example_config.yaml --array-size 4"
    exit 1
fi

mkdir -p logs

JOB_NAME="gen-datasets-parallel"

echo "Submitting parallel dataset generation"
echo "Config: $CONFIG"
echo "Array size: 0-$((ARRAY_SIZE-1)) (${ARRAY_SIZE} parallel jobs)"
echo "Logs: logs/${JOB_NAME}-%A_%a.out/err"
echo ""

sbatch \
    --array=0-$((ARRAY_SIZE-1)) \
    slurm/run_generate_datasets_parallel.sh \
    --config "$CONFIG"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
