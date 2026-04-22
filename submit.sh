#!/bin/bash
# Unified SLURM job submission script.
# Sources .env for SLURM_PARTITION and other settings.
#
# Usage:
#   ./submit.sh benchmark --config configs/baseline.yaml
#   ./submit.sh benchmark-parallel --config configs/example_config.yaml --array-size 8
#   ./submit.sh generate-datasets --config configs/example_config.yaml --array-size 4
#   ./submit.sh generate-baselines --config configs/example_config.yaml
#   ./submit.sh eval-external --model-path /path/to/model --animal cat
#
# Common options:
#   --max-gpus N       Limit concurrent GPU jobs (default: $SLURM_MAX_GPUS from .env, or 6)
#   --depends-on JID   Wait for SLURM job JID to finish (afterok) before starting.
#                      Pass "" to skip. Stdout is the submitted job's ID
#                      (or the passthrough dep ID when ALL_CACHED), so chains
#                      survive cache hits without breaking. All chatty output
#                      goes to stderr. Example — fire all three at once:
#
#                        J1=$(./submit.sh generate-datasets  --config configs/mark_sweep.yaml --array-size 4)
#                        J2=$(./submit.sh generate-baselines --config configs/mark_sweep.yaml --depends-on "$J1")
#                        J3=$(./submit.sh benchmark-parallel --config configs/mark_sweep.yaml --array-size 8 --depends-on "$J2")

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
else
    echo "Warning: .env not found. Copy .env.example to .env and configure it." >&2
fi

PARTITION="${SLURM_PARTITION:-general}"
MAX_GPUS="${SLURM_MAX_GPUS:-6}"

if [ $# -lt 1 ]; then
    {
        echo "Usage: ./submit.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  benchmark             Single-GPU benchmark (sequential)"
        echo "  benchmark-parallel    Multi-GPU benchmark (job array)"
        echo "  generate-datasets     Generate datasets (job array)"
        echo "  generate-baselines    Generate baseline evaluations"
        echo "  eval-external         Evaluate an external model"
        echo ""
        echo "Common options:"
        echo "  --array-size N        Job array size (default: 8 where applicable)"
        echo "  --max-gpus N          Max concurrent GPU jobs in array (default: 6)"
        echo "  --depends-on JOBID    Wait for SLURM JOBID (afterok) before starting"
    } >&2
    exit 1
fi

COMMAND="$1"
shift

# Parse common flags. Everything else is forwarded to the wrapper script.
ARRAY_SIZE=8
DEPENDS_ON=""
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
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$REPO_ROOT/logs"

SBATCH_DEPS=()
if [ -n "$DEPENDS_ON" ]; then
    SBATCH_DEPS=("--dependency=afterok:$DEPENDS_ON")
fi

# Every sbatch needs REPO_ROOT in its environment so the wrappers know where
# to cd. Passing it via --export (along with ALL) is more robust than relying
# on SLURM_SUBMIT_DIR, which defaults to whatever CWD the user happened to be
# in — that path may not exist on the compute node (looking at /local/scratch).
SBATCH_EXPORT="ALL,REPO_ROOT=$REPO_ROOT"

# Submit and emit the job ID on stdout. --parsable returns "JOBID" or
# "JOBID;cluster"; strip the cluster suffix so $(./submit.sh ...) is clean.
# A friendly "Submitted batch job N" goes to stderr to match plain `sbatch`.
submit() {
    local jobid rc
    jobid=$(sbatch --parsable --export="$SBATCH_EXPORT" "$@")
    rc=$?
    if [ $rc -ne 0 ] || [ -z "$jobid" ]; then
        echo "sbatch failed (rc=$rc, output='$jobid')" >&2
        exit ${rc:-1}
    fi
    jobid="${jobid%%;*}"
    echo "Submitted batch job $jobid" >&2
    echo "$jobid"
}

# When everything is cached we don't submit a new job. Pass through any
# incoming dependency so the caller's chain (J1 -> J2 -> J3) doesn't break.
emit_cached_passthrough() {
    if [ -n "$DEPENDS_ON" ]; then
        echo "$DEPENDS_ON"
    fi
}

# Run check_cached.py via uv so deps like dotenv are guaranteed available.
run_check_cached() {
    local out
    if ! out=$(uv run --project "$REPO_ROOT" "$REPO_ROOT/scripts/check_cached.py" "$@" 2>&1); then
        echo "check_cached.py failed:" >&2
        echo "$out" >&2
        exit 1
    fi
    printf '%s\n' "$out"
}

case "$COMMAND" in
    benchmark)
        CONFIG_ARG=""
        prev=""
        for arg in "${PASSTHROUGH_ARGS[@]}"; do
            if [ "$prev" = "--config" ]; then CONFIG_ARG="$arg"; fi
            prev="$arg"
        done
        if [ -n "$CONFIG_ARG" ]; then
            CACHE_OUTPUT=$(run_check_cached --config "$CONFIG_ARG")
            CACHE_STDERR=$(echo "$CACHE_OUTPUT" | head -3)
            CACHE_RESULT=$(echo "$CACHE_OUTPUT" | tail -1)
            echo "$CACHE_STDERR" >&2
            if [ "$CACHE_RESULT" = "ALL_CACHED" ]; then
                echo "All experiments already completed. Nothing to submit." >&2
                emit_cached_passthrough
                exit 0
            fi
        fi
        echo "Submitting benchmark job (partition: $PARTITION)" >&2
        submit --partition="$PARTITION" \
            "${SBATCH_DEPS[@]}" \
            slurm/run_benchmark.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    benchmark-parallel)
        JOB_NAME="benchmark-parallel"
        CONFIG_ARG=""
        prev=""
        for arg in "${PASSTHROUGH_ARGS[@]}"; do
            if [ "$prev" = "--config" ]; then CONFIG_ARG="$arg"; fi
            prev="$arg"
        done
        if [ -n "$CONFIG_ARG" ]; then
            CACHE_OUTPUT=$(run_check_cached --config "$CONFIG_ARG" --total-tasks "$ARRAY_SIZE")
            CACHE_STDERR=$(echo "$CACHE_OUTPUT" | head -3)
            CACHE_RESULT=$(echo "$CACHE_OUTPUT" | tail -1)
            echo "$CACHE_STDERR" >&2
            if [ "$CACHE_RESULT" = "ALL_CACHED" ]; then
                echo "All experiments already completed. Nothing to submit." >&2
                emit_cached_passthrough
                exit 0
            fi
            ARRAY_SPEC="$CACHE_RESULT"
        else
            ARRAY_SPEC="0-$((ARRAY_SIZE-1))"
        fi
        ARRAY_SPEC="${ARRAY_SPEC}%${MAX_GPUS}"
        echo "Submitting parallel benchmark (partition: $PARTITION, array: $ARRAY_SPEC, max concurrent: $MAX_GPUS)" >&2
        submit --partition="$PARTITION" \
            "${SBATCH_DEPS[@]}" \
            --job-name="$JOB_NAME" \
            --output="logs/${JOB_NAME}-%A_%a.out" \
            --error="logs/${JOB_NAME}-%A_%a.err" \
            --array="$ARRAY_SPEC" \
            slurm/run_benchmark_parallel.sh "${PASSTHROUGH_ARGS[@]}" --array-size "$ARRAY_SIZE"
        ;;

    generate-datasets)
        JOB_NAME="gen-datasets-parallel"
        CONFIG_ARG=""
        prev=""
        for arg in "${PASSTHROUGH_ARGS[@]}"; do
            if [ "$prev" = "--config" ]; then CONFIG_ARG="$arg"; fi
            prev="$arg"
        done
        if [ -n "$CONFIG_ARG" ]; then
            CACHE_OUTPUT=$(run_check_cached --config "$CONFIG_ARG" --stage datasets)
            CACHE_STDERR=$(echo "$CACHE_OUTPUT" | head -3)
            CACHE_RESULT=$(echo "$CACHE_OUTPUT" | tail -1)
            echo "$CACHE_STDERR" >&2
            if [ "$CACHE_RESULT" = "ALL_CACHED" ]; then
                echo "All datasets already cached. Nothing to submit." >&2
                emit_cached_passthrough
                exit 0
            fi
        fi
        echo "Submitting dataset generation (partition: $PARTITION, array: 0-$((ARRAY_SIZE-1))%${MAX_GPUS}, max concurrent: $MAX_GPUS)" >&2
        submit --partition="$PARTITION" \
            "${SBATCH_DEPS[@]}" \
            --array="0-$((ARRAY_SIZE-1))%${MAX_GPUS}" \
            slurm/run_generate_datasets_parallel.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    generate-baselines)
        CONFIG_ARG=""
        prev=""
        for arg in "${PASSTHROUGH_ARGS[@]}"; do
            if [ "$prev" = "--config" ]; then CONFIG_ARG="$arg"; fi
            prev="$arg"
        done
        if [ -n "$CONFIG_ARG" ]; then
            CACHE_OUTPUT=$(run_check_cached --config "$CONFIG_ARG" --stage baselines)
            CACHE_STDERR=$(echo "$CACHE_OUTPUT" | head -3)
            CACHE_RESULT=$(echo "$CACHE_OUTPUT" | tail -1)
            echo "$CACHE_STDERR" >&2
            if [ "$CACHE_RESULT" = "ALL_CACHED" ]; then
                echo "All baselines already cached. Nothing to submit." >&2
                emit_cached_passthrough
                exit 0
            fi
        fi
        echo "Submitting baseline generation (partition: $PARTITION)" >&2
        submit --partition="$PARTITION" \
            "${SBATCH_DEPS[@]}" \
            slurm/run_generate_baselines.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    eval-external)
        echo "Submitting external model evaluation (partition: $PARTITION)" >&2
        submit --partition="$PARTITION" \
            "${SBATCH_DEPS[@]}" \
            slurm/run_eval_external.sh "${PASSTHROUGH_ARGS[@]}"
        ;;

    *)
        echo "Unknown command: $COMMAND" >&2
        exit 1
        ;;
esac

{
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
} >&2
