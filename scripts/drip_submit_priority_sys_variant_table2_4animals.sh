#!/usr/bin/env bash
# Drip-submit configs/priority_sys_variant_table2_4animals.yaml under the
# per-user QOSMaxSubmitJobPerUserLimit (~250 pending+running jobs).
#
# Fills Table 2 gaps (SYS_VARIANT_SCENARIOS in sl/tables.py) for all four
# animals (cat, eagle, owl, wolf) at the priority 1x6 seed grid
# (gen_seeds=[1,42,123,7,11,13], train_seed=42). Raw experiment count is
# 5 variants x 4 animals x 10 ranks x 6 gen_seeds x 1 train_seed = 1200;
# per the 2026-05-16 registry audit, 832 are already cached, leaving 368
# fresh experiments (363 cat/eagle, 5 owl/wolf rank-1 holes).
#
# Drip-resubmits cache-aware array jobs:
#   1. wait for the user queue to drop below THRESHOLD
#   2. submit ./submit.sh benchmark-parallel ... --array-size $ARRAY_SIZE
#   3. wait for that array to fully drain
#   4. re-submit; if "ALL_CACHED" we're done, otherwise loop
#
# Run on a login node so it survives ssh disconnects:
#   nohup bash scripts/drip_submit_priority_sys_variant_table2_4animals.sh \
#     > logs/drip_submit_priority_sys_variant_table2_4animals.log 2>&1 &
#
# Tail progress:
#   tail -f logs/drip_submit_priority_sys_variant_table2_4animals.log

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIG="configs/priority_sys_variant_table2_4animals.yaml"
# 368 fresh / 75 tasks ~= 5 experiments/task at peak; ARRAY_SIZE=75 fits
# comfortably under the 250 QOS cap with THRESHOLD=60 headroom for any
# other in-flight jobs.
ARRAY_SIZE=75
MAX_GPUS=6
THRESHOLD=60
POLL_SECONDS=300        # 5 min polling cadence
MAX_ROUNDS=12           # hard ceiling: 12 rounds * 75 tasks ~= 900 experiments,
                        # well above the 368-fresh count of this config

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
