#!/bin/bash
# Unified SLURM job submission script.
# Sources .env for SLURM_PARTITION and other settings.
#
# Usage:
#   ./submit.sh benchmark --config configs/baseline.yaml
#   ./submit.sh benchmark --preset quick
#   ./submit.sh benchmark-parallel --config configs/example_config.yaml --array-size 8
#   ./submit.sh benchmark-parallel --config configs/cat_filtered.yaml --array-size 9 --max-gpus 4
#   ./submit.sh generate-datasets --config configs/example_config.yaml
#   ./submit.sh generate-datasets --config configs/example_config.yaml --array-size 4
#   ./submit.sh generate-baselines --config configs/example_config.yaml
#   ./submit.sh eval-external --model-path /path/to/model --animal cat
#   ./submit.sh build-val-datasets --animals cat owl eagle
#   ./submit.sh score-val-loss --animals cat owl eagle                       # serial
#   ./submit.sh score-val-loss --array 0-3 -- --animals cat owl eagle        # array (4-way)
#   ./submit.sh snapshot-score-val-loss --all-runs                           # serial
#   ./submit.sh snapshot-score-val-loss --array 0-3 -- --all-runs            # array (4-way)
#   ./submit.sh eval-baselines
#   ./submit.sh build-divergence-masks -- --animals cat owl eagle
#   ./submit.sh score-val-divergence-accuracy --array 0-3 -- --animals cat owl eagle
#
# --max-gpus N   Limit concurrent GPU jobs (default: $SLURM_MAX_GPUS from .env, or 6)

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
MAX_GPUS="${SLURM_MAX_GPUS:-6}"

if [ $# -lt 1 ]; then
    echo "Usage: ./submit.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  benchmark                  Single-GPU benchmark (sequential)"
    echo "  benchmark-parallel         Multi-GPU benchmark (job array)"
    echo "  generate-datasets          Generate datasets (job array)"
    echo "  generate-baselines         Generate baseline evaluations"
    echo "  eval-external              Evaluate an external model"
    echo "  build-val-datasets         Build 2k held-out val jsonls per animal"
    echo "  score-val-loss             Score final LoRA adapters on val (--array optional)"
    echo "  snapshot-score-val-loss    Score snapshot adapters on val (--array optional)"
    echo "  eval-baselines             Extract/reclassify base-model generation baselines"
    echo "  build-divergence-masks     Build teacher-vs-default argmax masks per val file"
    echo "  score-val-divergence-accuracy"
    echo "                             Score adapters on accuracy-on-divergent-positions (--array optional)"
    echo ""
    echo "Examples:"
    echo "  ./submit.sh benchmark --config configs/baseline.yaml"
    echo "  ./submit.sh benchmark-parallel --config configs/example_config.yaml --array-size 8"
    echo "  ./submit.sh generate-datasets --config configs/example_config.yaml"
    echo "  ./submit.sh generate-baselines --config configs/example_config.yaml"
    echo "  ./submit.sh score-val-loss --array 0-3 -- --animals cat owl eagle"
    echo "  ./submit.sh snapshot-score-val-loss --array 0-3 -- --all-runs"
    exit 1
fi

COMMAND="$1"
shift

# Parse --array-size, --max-gpus, --array from args.
# --array takes a raw sbatch array spec (e.g. "0-3" or "0-7%4") for the new
# ad-hoc scripts (score-val-loss, snapshot-score-val-loss). Distinct from
# --array-size (which is just an integer count used by benchmark-parallel).
ARRAY_SIZE=8
ARRAY_SPEC_OVERRIDE=""
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --array-size)
            ARRAY_SIZE="$2"
            shift 2
            ;;
        --max-gpus)
            MAX_GPUS="$2"
            shift 2
            ;;
        --array)
            ARRAY_SPEC_OVERRIDE="$2"
            shift 2
            ;;
        --)
            shift
            PASSTHROUGH_ARGS+=("$@")
            break
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
        ARRAY_SPEC="${ARRAY_SPEC}%${MAX_GPUS}"
        echo "Submitting parallel benchmark (partition: $PARTITION, array: $ARRAY_SPEC, max concurrent: $MAX_GPUS)"
        sbatch --partition="$PARTITION" \
            --job-name="$JOB_NAME" \
            --output="logs/${JOB_NAME}-%A_%a.out" \
            --error="logs/${JOB_NAME}-%A_%a.err" \
            --array="$ARRAY_SPEC" \
            slurm/run_benchmark_parallel.sh "${PASSTHROUGH_ARGS[@]}" --array-size "$ARRAY_SIZE"
        ;;

    generate-datasets)
        JOB_NAME="gen-datasets-parallel"
        echo "Submitting dataset generation (partition: $PARTITION, array: 0-$((ARRAY_SIZE-1))%${MAX_GPUS}, max concurrent: $MAX_GPUS)"
        sbatch --partition="$PARTITION" \
            --array=0-$((ARRAY_SIZE-1))%${MAX_GPUS} \
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

    build-val-datasets)
        echo "Submitting val-dataset build (partition: $PARTITION)"
        sbatch --partition="$PARTITION" \
            slurm/run_build_val_datasets.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    score-val-loss)
        # Teacher-forced CE on final adapters. Array sharding is handled inside
        # the python script via --task-id/--total-tasks, which run_score_val_loss.sh
        # reads from SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT. Pass --array to
        # this submitter for an array submission.
        ARRAY_ARG=()
        if [ -n "$ARRAY_SPEC_OVERRIDE" ]; then
            ARRAY_ARG=(--array="$ARRAY_SPEC_OVERRIDE")
            echo "Submitting score-val-loss (partition: $PARTITION, array: $ARRAY_SPEC_OVERRIDE)"
        else
            echo "Submitting score-val-loss (partition: $PARTITION, serial)"
        fi
        sbatch --partition="$PARTITION" "${ARRAY_ARG[@]}" \
            slurm/run_score_val_loss.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    snapshot-score-val-loss)
        # Per-step val CE on snapshot adapters. Same sharding pattern as
        # score-val-loss.
        ARRAY_ARG=()
        if [ -n "$ARRAY_SPEC_OVERRIDE" ]; then
            ARRAY_ARG=(--array="$ARRAY_SPEC_OVERRIDE")
            echo "Submitting snapshot-score-val-loss (partition: $PARTITION, array: $ARRAY_SPEC_OVERRIDE)"
        else
            echo "Submitting snapshot-score-val-loss (partition: $PARTITION, serial)"
        fi
        sbatch --partition="$PARTITION" "${ARRAY_ARG[@]}" \
            slurm/run_snapshot_score_val_loss.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    eval-baselines)
        echo "Submitting baseline extraction (partition: $PARTITION)"
        sbatch --partition="$PARTITION" \
            slurm/run_eval_baselines.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    build-divergence-masks)
        # Teacher-vs-default argmax masks per val file. Sharding by val file
        # via --task-id/--total-tasks inside the python script.
        ARRAY_ARG=()
        if [ -n "$ARRAY_SPEC_OVERRIDE" ]; then
            ARRAY_ARG=(--array="$ARRAY_SPEC_OVERRIDE")
            echo "Submitting build-divergence-masks (partition: $PARTITION, array: $ARRAY_SPEC_OVERRIDE)"
        else
            echo "Submitting build-divergence-masks (partition: $PARTITION, serial)"
        fi
        sbatch --partition="$PARTITION" "${ARRAY_ARG[@]}" \
            slurm/run_build_divergence_masks.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    score-val-divergence-accuracy)
        # Per-adapter accuracy on divergent positions. Same sharding pattern
        # as score-val-loss.
        ARRAY_ARG=()
        if [ -n "$ARRAY_SPEC_OVERRIDE" ]; then
            ARRAY_ARG=(--array="$ARRAY_SPEC_OVERRIDE")
            echo "Submitting score-val-divergence-accuracy (partition: $PARTITION, array: $ARRAY_SPEC_OVERRIDE)"
        else
            echo "Submitting score-val-divergence-accuracy (partition: $PARTITION, serial)"
        fi
        sbatch --partition="$PARTITION" "${ARRAY_ARG[@]}" \
            slurm/run_score_val_divergence_accuracy.sh "${PASSTHROUGH_ARGS[@]}"
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
