#!/usr/bin/env bash
# Generate cat-on-wolf-basis datasets, then train/evaluate their students.
#
# Run from a login node:
#   nohup bash scripts/chain_cat_wolfbasis.sh \
#       > logs/chain_cat_wolfbasis.log 2>&1 &
#
# Monitor:
#   tail -f logs/chain_cat_wolfbasis.log
#
# The script stops before training if dataset generation fails or any expected
# dataset is missing. Benchmark submission is cache-aware and retries any
# incomplete experiments after the preceding array drains.

set -u

cd "$(dirname "$0")/.."
mkdir -p logs

CONFIG="configs/attention_subspace_replace_greedy_cat_wolfbasis_r8_bs4.yaml"
POLL_SECONDS=300
MAX_SUBMIT_ATTEMPTS=5
MAX_BENCHMARK_ROUNDS=3

DATASET_DIR="/net/projects2/interp/subliminal/shared/results/attention_subspace_datasets/replace_cat_on_wolf_g123_k128-256_greedy"
EXPECTED_DATASETS=(
    "$DATASET_DIR/repgrank128xwolf_cat_on_wolf_g123.jsonl"
    "$DATASET_DIR/repgrank256xwolf_cat_on_wolf_g123.jsonl"
    "$DATASET_DIR/repgneutral_cat_on_wolf_g123.jsonl"
    "$DATASET_DIR/repgteacher_cat_on_wolf_g123.jsonl"
)

log() {
    echo "[$(date -Iseconds)] $*"
}

submit_with_retries() {
    local label="$1"
    shift
    local attempt=0
    local output

    while (( attempt < MAX_SUBMIT_ATTEMPTS )); do
        output=$("$@" 2>&1)
        echo "$output"
        if grep -q "Submitted batch job\|All experiments already completed" <<<"$output"; then
            LAST_SUBMIT_OUTPUT="$output"
            return 0
        fi
        attempt=$((attempt + 1))
        log "$label submission failed (attempt $attempt/$MAX_SUBMIT_ATTEMPTS); retrying in ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
    done
    return 1
}

wait_for_job() {
    local job_id="$1"
    local queue_output
    local state

    log "Waiting for job $job_id to leave the queue"
    while :; do
        if queue_output=$(squeue -h -j "$job_id" -o "%i" 2>&1); then
            if [[ -z "$queue_output" ]]; then
                break
            fi
            log "Job $job_id is still active; checking again in ${POLL_SECONDS}s"
        else
            log "squeue failed; checking again in ${POLL_SECONDS}s: $queue_output"
        fi
        sleep "$POLL_SECONDS"
    done

    while :; do
        state=$(sacct -X -n -j "$job_id" -o State 2>/dev/null | awk 'NF {print $1; exit}')
        if [[ -n "$state" ]]; then
            break
        fi
        log "Waiting for accounting state for job $job_id"
        sleep 30
    done
    log "Job $job_id finished with state $state"
    [[ "$state" == COMPLETED ]]
}

log "Submitting cat-on-wolf-basis dataset generation"
LAST_SUBMIT_OUTPUT=""
if ! submit_with_retries "Dataset" ./submit.sh attention-subspace-dataset -- \
    --recipient cat \
    --basis-from wolf \
    --source-run-dir /net/projects2/interp/subliminal/shared/results/attention_write_rank_sweep/qwen-owl-wolf-eagle-g123 \
    --replace-ranks 128,256 \
    --temperature 0; then
    log "Dataset submission failed after $MAX_SUBMIT_ATTEMPTS attempts; aborting"
    exit 1
fi

dataset_job_id=$(grep "Submitted batch job" <<<"$LAST_SUBMIT_OUTPUT" | awk '{print $NF}')
if [[ -z "$dataset_job_id" ]]; then
    log "Could not parse the dataset job ID; aborting"
    exit 1
fi
log "Dataset generation submitted as job $dataset_job_id"

if ! wait_for_job "$dataset_job_id"; then
    log "Dataset job $dataset_job_id did not complete successfully; not submitting training"
    exit 1
fi

missing=0
for dataset in "${EXPECTED_DATASETS[@]}"; do
    if [[ ! -s "$dataset" ]]; then
        log "Missing or empty expected dataset: $dataset"
        missing=1
    fi
done
if (( missing )); then
    log "Dataset verification failed; not submitting training"
    exit 1
fi
log "All four expected datasets are present"

for (( round = 1; round <= MAX_BENCHMARK_ROUNDS; round++ )); do
    log "Benchmark round $round/$MAX_BENCHMARK_ROUNDS"
    LAST_SUBMIT_OUTPUT=""
    if ! submit_with_retries "Benchmark" ./submit.sh benchmark-parallel \
        --config "$CONFIG" \
        --array-size 4 \
        --max-gpus 4; then
        log "Benchmark submission failed after $MAX_SUBMIT_ATTEMPTS attempts; aborting"
        exit 1
    fi

    if grep -q "All experiments already completed" <<<"$LAST_SUBMIT_OUTPUT"; then
        log "All reciprocal cat-on-wolf-basis experiments are cached"
        exit 0
    fi

    benchmark_job_id=$(grep "Submitted batch job" <<<"$LAST_SUBMIT_OUTPUT" | awk '{print $NF}')
    if [[ -z "$benchmark_job_id" ]]; then
        log "Could not parse the benchmark job ID; aborting"
        exit 1
    fi
    log "Benchmark round $round submitted as job $benchmark_job_id"
    if ! wait_for_job "$benchmark_job_id"; then
        log "Benchmark job $benchmark_job_id was not fully successful; cache-aware retry will cover incomplete experiments"
    fi
done

log "Reached $MAX_BENCHMARK_ROUNDS benchmark rounds without reaching ALL_CACHED"
exit 2
