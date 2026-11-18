#!/usr/bin/env bash
# Drip-submit configs/optimizer_sweep_sgd_lr3e-4_4animals.yaml under the
# per-user QOSMaxSubmitJobPerUserLimit (~250 pending+running jobs).
#
# Follow-up to scripts/drip_submit_optimizer_sweep_sgd.sh (which ran lr=1e-2
# and diverged everywhere). The cat smoke test settled on lr=3e-4 as the
# stable choice across the rank ladder; this drip runs that lr across the
# canonical 4-animal x 10-rank x 3-gen_seed grid so SGD has a fair head-to-
# head against AdamW / Muon in the Appendix-K figure.
#
# Submits 4 animals x 10 ranks x 3 gen_seeds x 1 train_seed = 120 experiments
# (all fresh; the `_sgd_lr0.0003` infix puts these in a different model_hash
# bucket than the lr=1e-2 SGD cache and the AdamW/Muon caches) as a sequence
# of cache-aware array jobs:
#
#   1. wait for the user queue to drop below THRESHOLD
#   2. submit ./submit.sh benchmark-parallel ... --array-size $ARRAY_SIZE
#   3. wait for that array to fully drain
#   4. re-run check_cached.py via submit.sh; if "ALL_CACHED" we're done,
#      otherwise loop and submit the remaining work
#
# Run on fe02 in the background:
#   nohup bash scripts/drip_submit_optimizer_sweep_sgd_lr3e-4.sh \
#     > logs/drip_submit_optimizer_sweep_sgd_lr3e-4.log 2>&1 &
#
# To tail progress:
#   tail -f logs/drip_submit_optimizer_sweep_sgd_lr3e-4.log
#
# Prerequisites (run once before launching; both should be no-ops since
# datasets/baselines are shared with priority_lora_qwen_default_5animals.yaml,
# optimizer_sweep_muon_4animals.yaml, and optimizer_sweep_sgd_4animals.yaml):
#   ./submit.sh generate-datasets  --config configs/optimizer_sweep_sgd_lr3e-4_4animals.yaml
#   ./submit.sh generate-baselines --config configs/optimizer_sweep_sgd_lr3e-4_4animals.yaml
#
# Expected duration: 120 runs / 60 tasks ~= 2 experiments/task; at ~30 min/run
# on a 7B LoRA, ~1 hour per task; one or two drip rounds clears the grid in
# overnight wall-clock at MAX_GPUS=8 (the original lr=1e-2 sweep took ~12h end
# to end on a busy cluster; expect similar).
#
# NOTE: requires the SGD-aware sl/finetuning code (sl/finetuning/data_models.py
# accepts optimizer="sgd" via the widened Literal, and services.py builds a
# torch.optim.SGD with momentum=0, weight_decay=0 for that branch).

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIG="configs/optimizer_sweep_sgd_lr3e-4_4animals.yaml"
ARRAY_SIZE=60           # ~120 experiments / 60 tasks ~= 2 experiments/task (well under SLURM 10h cap)
MAX_GPUS=8              # max out the user's per-job GPU concurrency
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
