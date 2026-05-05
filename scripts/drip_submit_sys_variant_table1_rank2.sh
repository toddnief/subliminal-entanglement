#!/usr/bin/env bash
# Drip-submit configs/sys_variant_table1_rank2.yaml under the per-user
# QOSMaxSubmitJobPerUserLimit (~250 pending+running jobs).
#
# Fills rank=2 for all 9 Table-1 cells (subliminal + 6 cew + 2 new corners)
# across cat / owl / eagle / wolf:
#   324 total experiments (~274 uncached at submission time, ~50 cached
#   from cat_filtered.yaml etc.).
#
# Rank 2 trains fast (~5-10 min/run), so wall time is roughly:
#   274 / 6 * 7.5 min ≈ 6 hours. Comfortably overnight.
#
# Generation template matches sys_variant_cew + sys_variant_coew_new_corners,
# so the required datasets are all already in the registry -- no race with
# the collaborator's submit_new_corners.sh dataset generation.
#
# Run on fe02 in the background:
#   nohup bash scripts/drip_submit_sys_variant_table1_rank2.sh \
#     > logs/drip_submit_sys_variant_table1_rank2.log 2>&1 &
#
# Tail progress with:
#   tail -f logs/drip_submit_sys_variant_table1_rank2.log

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIG="configs/sys_variant_table1_rank2.yaml"
ARRAY_SIZE=150          # 274 uncached / 150 ≈ 2 experiments/task at rank 2
MAX_GPUS=6
THRESHOLD=60            # leave headroom under the ~250 QOS cap; collaborator's
                        # new-corners run is on a separate user but be polite
POLL_SECONDS=180        # 3 min polling cadence (rank 2 is faster than the
                        # full sweep, so don't over-sleep)
MAX_ROUNDS=10

log() {
    echo "[$(date -Iseconds)] $*"
}

count_my_jobs() {
    squeue --me -h -t all --array 2>/dev/null | wc -l
}

wait_for_queue_capacity() {
    while :; do
        local cur
        cur=$(count_my_jobs)
        if [[ -z "$cur" ]]; then
            log "squeue returned empty; retrying in ${POLL_SECONDS}s"
            sleep "$POLL_SECONDS"
            continue
        fi
        if (( cur <= THRESHOLD )); then
            log "Queue depth $cur <= $THRESHOLD; ok to submit"
            return 0
        fi
        log "Queue depth $cur > $THRESHOLD; sleeping ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
    done
}

wait_for_job_drain() {
    local job_id="$1"
    log "Waiting for job $job_id to drain..."
    while :; do
        local remaining
        remaining=$(squeue -j "$job_id" -h -t all --array 2>/dev/null | wc -l)
        if [[ -z "$remaining" || "$remaining" -eq 0 ]]; then
            log "Job $job_id drained"
            return 0
        fi
        log "Job $job_id still has $remaining tasks in flight; sleeping ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
    done
}

submit_round() {
    local attempt=0
    while (( attempt < 5 )); do
        LAST_SUBMIT_OUT=$(./submit.sh benchmark-parallel \
            --config "$CONFIG" \
            --array-size "$ARRAY_SIZE" \
            --max-gpus "$MAX_GPUS" 2>&1)
        echo "$LAST_SUBMIT_OUT"
        if grep -q "ALL_CACHED\|All experiments already completed" <<<"$LAST_SUBMIT_OUT"; then
            return 0
        fi
        if grep -q "Submitted batch job" <<<"$LAST_SUBMIT_OUT"; then
            return 0
        fi
        attempt=$((attempt + 1))
        log "Submission failed (attempt $attempt/5); sleeping ${POLL_SECONDS}s and retrying"
        sleep "$POLL_SECONDS"
    done
    return 1
}

log "Starting drip submission for $CONFIG"
log "  ARRAY_SIZE=$ARRAY_SIZE, MAX_GPUS=$MAX_GPUS, THRESHOLD=$THRESHOLD, MAX_ROUNDS=$MAX_ROUNDS"

round=0
while (( round < MAX_ROUNDS )); do
    round=$((round + 1))
    log "===== Round $round / $MAX_ROUNDS ====="

    wait_for_queue_capacity

    LAST_SUBMIT_OUT=""
    if ! submit_round; then
        log "FAILED to submit round $round after 5 attempts; aborting"
        exit 1
    fi

    if grep -q "All experiments already completed" <<<"$LAST_SUBMIT_OUT"; then
        log "submit.sh reports ALL_CACHED; sweep is complete."
        exit 0
    fi

    job_id=$(grep "Submitted batch job" <<<"$LAST_SUBMIT_OUT" | awk '{print $NF}')
    if [[ -z "$job_id" ]]; then
        log "Could not parse job id from submit.sh output; aborting"
        exit 1
    fi
    log "Round $round submitted as job $job_id"

    wait_for_job_drain "$job_id"
done

log "Hit MAX_ROUNDS=$MAX_ROUNDS without ALL_CACHED. Inspect logs and registry."
exit 2
