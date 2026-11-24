#!/usr/bin/env bash
# Drip-submit configs/epochs_sweep_cat_owl_r256.yaml under the per-user
# QOSMaxSubmitJobPerUserLimit (~250 pending+running jobs).
#
# Epoch sweep: cat + owl, LoRA rank 256, n_epochs in {3,5,10,20,40},
# gen_seeds {1,42,123}, train_seed 42.
#   2 animals x 5 epoch values x 3 gen_seeds = 30 experiments
#   (the 6 n_epochs=3 points are already cached; submit.sh skips them, so
#    ~24 are genuinely fresh)
#
# ARRAY_SIZE is set to one experiment per task on purpose: the 40-epoch runs
# are long (~13x the 3-epoch train time), and array indices are assigned by
# (index mod ARRAY_SIZE), so packing two long runs into one task could blow
# the 10h SLURM wall. One-per-task keeps every task safely under the limit.
#
# Run on a login node in the background:
#   nohup bash scripts/drip_submit_epochs_cat_owl_r256.sh \
#     > logs/drip_submit_epochs_cat_owl_r256.log 2>&1 &
#
# To tail progress:
#   tail -f logs/drip_submit_epochs_cat_owl_r256.log
#
# Prerequisites (run once before launching):
#   ./submit.sh generate-datasets  --config configs/epochs_sweep_cat_owl_r256.yaml
#   ./submit.sh generate-baselines --config configs/epochs_sweep_cat_owl_r256.yaml

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIG="configs/epochs_sweep_cat_owl_r256.yaml"
ARRAY_SIZE=30           # one experiment per task (30 cells, ~24 fresh)
MAX_GPUS=6              # per-array GPU concurrency
THRESHOLD=60            # leave headroom under the 250 QOS cap
POLL_SECONDS=300        # 5 min polling cadence
MAX_ROUNDS=20           # hard ceiling so a buggy loop can't run forever

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
