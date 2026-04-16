#!/bin/bash
# Submit benchmark as parallel SLURM job array
# Each array task runs a subset of experiments on its own GPU
#
# Usage:
#   ./slurm/submit_benchmark_parallel.sh --config benchmarks/example_config.yaml --array-size 8
#   ./slurm/submit_benchmark_parallel.sh --preset controlled --array-size 4

set -e

# Default values
ARRAY_SIZE=8
CONFIG_ARG=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --array-size)
            ARRAY_SIZE="$2"
            shift 2
            ;;
        --config)
            CONFIG_ARG="--config $2"
            shift 2
            ;;
        --preset)
            CONFIG_ARG="--preset $2"
            shift 2
            ;;
        *)
            # Collect unknown args and forward them to the Python script
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [ -z "$CONFIG_ARG" ]; then
    echo "Error: Must specify --config or --preset"
    echo ""
    echo "Usage:"
    echo "  ./slurm/submit_benchmark_parallel.sh --config benchmarks/example_config.yaml --array-size 8"
    echo "  ./slurm/submit_benchmark_parallel.sh --preset controlled --array-size 4"
    exit 1
fi

# Create logs directory
mkdir -p logs

JOB_NAME="benchmark-parallel"

echo "Submitting parallel benchmark job array"
echo "Array size: 0-$((ARRAY_SIZE-1)) (${ARRAY_SIZE} parallel jobs)"
echo "Config: $CONFIG_ARG"
echo "Logs: logs/${JOB_NAME}-%A_%a.out/err"
echo ""

sbatch \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}-%A_%a.out" \
    --error="logs/${JOB_NAME}-%A_%a.err" \
    --array=0-$((ARRAY_SIZE-1)) \
    slurm/run_benchmark_parallel.sh $CONFIG_ARG --array-size $ARRAY_SIZE $EXTRA_ARGS

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  ./monitor_benchmark.sh <JOB_ID>"
