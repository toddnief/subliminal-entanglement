#!/bin/bash
# Unified SLURM job submission script.
# Sources .env for SLURM_PARTITION and other settings.
#
# Usage:
#   ./submit.sh benchmark --config configs/baseline.yaml
#   ./submit.sh benchmark --preset quick
#   ./submit.sh benchmark-parallel --config configs/example_config.yaml --array-size 8
#   ./submit.sh generate-datasets --config configs/example_config.yaml
#   ./submit.sh generate-datasets --config configs/example_config.yaml --array-size 4
#   ./submit.sh generate-baselines --config configs/example_config.yaml
#   ./submit.sh eval-external --model-path /path/to/model --animal cat

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
else
    echo "Warning: .env not found. Copy .env.example to .env and configure it."
fi

PARTITION="${SLURM_PARTITION:-general}"

if [ $# -lt 1 ]; then
    echo "Usage: ./submit.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  benchmark             Single-GPU benchmark (sequential)"
    echo "  benchmark-parallel    Multi-GPU benchmark (job array)"
    echo "  generate-datasets     Generate datasets (job array)"
    echo "  generate-baselines    Generate baseline evaluations"
    echo "  eval-external         Evaluate an external model"
    echo ""
    echo "Examples:"
    echo "  ./submit.sh benchmark --config configs/baseline.yaml"
    echo "  ./submit.sh benchmark-parallel --config configs/example_config.yaml --array-size 8"
    echo "  ./submit.sh generate-datasets --config configs/example_config.yaml"
    echo "  ./submit.sh generate-baselines --config configs/example_config.yaml"
    exit 1
fi

COMMAND="$1"
shift

# Parse --array-size from args (needed for parallel commands)
ARRAY_SIZE=8
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --array-size)
            ARRAY_SIZE="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$REPO_ROOT/logs"

# Helper: check which experiments are already cached
check_cached() {
    local config_arg=""
    for arg in "${PASSTHROUGH_ARGS[@]}"; do
        if [ "$prev" = "--config" ]; then
            config_arg="$arg"
        fi
        prev="$arg"
    done
    if [ -z "$config_arg" ]; then
        return 1
    fi
    python3 "$REPO_ROOT/scripts/check_cached.py" --config "$config_arg" --total-tasks "$ARRAY_SIZE" 2>&1
}

case "$COMMAND" in
    benchmark)
        # Check for cached experiments
        CONFIG_ARG=""
        prev=""
        for arg in "${PASSTHROUGH_ARGS[@]}"; do
            if [ "$prev" = "--config" ]; then
                CONFIG_ARG="$arg"
            fi
            prev="$arg"
        done
        if [ -n "$CONFIG_ARG" ]; then
            CACHE_OUTPUT=$(python3 "$REPO_ROOT/scripts/check_cached.py" --config "$CONFIG_ARG" 2>&1)
            CACHE_STDERR=$(echo "$CACHE_OUTPUT" | head -3)
            CACHE_RESULT=$(echo "$CACHE_OUTPUT" | tail -1)
            echo "$CACHE_STDERR"
            if [ "$CACHE_RESULT" = "ALL_CACHED" ]; then
                echo "All experiments already completed. Nothing to submit."
                exit 0
            fi
        fi
        echo "Submitting benchmark job (partition: $PARTITION)"
        sbatch --partition="$PARTITION" \
            slurm/run_benchmark.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    benchmark-parallel)
        JOB_NAME="benchmark-parallel"
        # Check which task IDs actually have uncached work
        CONFIG_ARG=""
        prev=""
        for arg in "${PASSTHROUGH_ARGS[@]}"; do
            if [ "$prev" = "--config" ]; then
                CONFIG_ARG="$arg"
            fi
            prev="$arg"
        done
        if [ -n "$CONFIG_ARG" ]; then
            CACHE_OUTPUT=$(python3 "$REPO_ROOT/scripts/check_cached.py" --config "$CONFIG_ARG" --total-tasks "$ARRAY_SIZE" 2>&1)
            CACHE_STDERR=$(echo "$CACHE_OUTPUT" | head -3)
            CACHE_RESULT=$(echo "$CACHE_OUTPUT" | tail -1)
            echo "$CACHE_STDERR"
            if [ "$CACHE_RESULT" = "ALL_CACHED" ]; then
                echo "All experiments already completed. Nothing to submit."
                exit 0
            fi
            ARRAY_SPEC="$CACHE_RESULT"
        else
            ARRAY_SPEC="0-$((ARRAY_SIZE-1))"
        fi
        echo "Submitting parallel benchmark (partition: $PARTITION, array: $ARRAY_SPEC)"
        sbatch --partition="$PARTITION" \
            --job-name="$JOB_NAME" \
            --output="logs/${JOB_NAME}-%A_%a.out" \
            --error="logs/${JOB_NAME}-%A_%a.err" \
            --array="$ARRAY_SPEC" \
            slurm/run_benchmark_parallel.sh "${PASSTHROUGH_ARGS[@]}" --array-size "$ARRAY_SIZE"
        ;;

    generate-datasets)
        JOB_NAME="gen-datasets-parallel"
        echo "Submitting dataset generation (partition: $PARTITION, array: 0-$((ARRAY_SIZE-1)))"
        sbatch --partition="$PARTITION" \
            --array=0-$((ARRAY_SIZE-1)) \
            slurm/run_generate_datasets_parallel.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    generate-baselines)
        echo "Submitting baseline generation (partition: $PARTITION)"
        sbatch --partition="$PARTITION" \
            slurm/run_generate_baselines.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    eval-external)
        echo "Submitting external model evaluation (partition: $PARTITION)"
        sbatch --partition="$PARTITION" \
            slurm/run_eval_external.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    *)
        echo "Unknown command: $COMMAND"
        echo "Run ./submit.sh without arguments to see available commands."
        exit 1
        ;;
esac

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
