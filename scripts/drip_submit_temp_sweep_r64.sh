#!/usr/bin/env bash
# Drip-submit configs/temp.yaml (rank-64 constrained-decode temperature sweep)
# under the per-user QOSMaxSubmitJobPerUserLimit (~250 pending+running jobs).
#
# Sweep: 4 animals (cat, eagle, wolf, owl) × 7 temps (0.0, 0.5, 0.7, 1.0, 1.3,
# 1.5, 2.0) × 3 gen_seeds × 1 train_seed (42) = 84 experiments at LoRA rank 64.
# Unique datasets: 84 (one per animal × temp × gen_seed). The training_seed
# axis was dropped after a few cells confirmed run-to-run variance is
# dominated by gen_seed; see the header comment in configs/temp.yaml.
#
# Each round:
#   1. wait for the user queue to drop below THRESHOLD
#   2. submit ./submit.sh benchmark-parallel ... --array-size $ARRAY_SIZE
#      (submit.sh internally uses scripts/check_cached.py to schedule only
#       the array indices that still have uncached experiments)
#   3. wait for that array to fully drain
#   4. loop until check_cached.py reports ALL_CACHED
#
# Idempotent: safe to relaunch after a partial run; only uncached experiments
# are scheduled. Handles preemption / failed tasks transparently.
#
# Run on a login node in the background:
#   nohup bash scripts/drip_submit_temp_sweep_r64.sh \
#     > logs/drip_submit_temp_sweep_r64.log 2>&1 &
#   disown
#
# To tail progress:
#   tail -f logs/drip_submit_temp_sweep_r64.log
#
# To cancel:
#   pkill -f drip_submit_temp_sweep_r64.sh
#   scancel -u $USER --jobname=benchmark-parallel   # optional, kills running tasks too

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIG="configs/temp.yaml"
ARRAY_SIZE=21           # 84 experiments / 21 tasks = 4 experiments/task; ~2-4 h/task
MAX_GPUS=6
THRESHOLD=60            # leave headroom under the 250 QOS cap
POLL_SECONDS=300        # 5 min polling cadence
MAX_ROUNDS=12           # hard ceiling so a buggy loop can't run forever

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

log "Starting drip submission for $CONFIG (rank-64 constrained-decode temperature sweep)"
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
