#!/usr/bin/env bash
# === COLLABORATOR ENTRY POINT: tree + band rank sweeps ===
#
# Submits the four production preference-category rank sweeps in sequence:
#
#   1. configs/qwen_tree_rank_sweep.yaml    (315 experiments)
#   2. configs/gemma_tree_rank_sweep.yaml   (315 experiments)
#   3. configs/qwen_band_rank_sweep.yaml    (315 experiments)
#   4. configs/gemma_band_rank_sweep.yaml   (315 experiments)
#
#   Total: 1260 experiments across 4 chained sweeps.
#
# Why this script exists:
#   - Each sweep has 315 array tasks, but the cluster's
#     QOSMaxSubmitJobPerUserLimit is ~250 jobs in queue per user.
#   - So we drip-feed: submit one sweep with --array-size 63 (5 experiments
#     per task), wait for it to drain, then submit the next.
#   - This script is cache-aware via scripts/check_cached.py, so re-running
#     it after a partial run will only submit the experiments that haven't
#     completed yet.
#
# Wall-clock estimate at --max-gpus 6 with ~45-90 min/experiment:
#   ~2-4 days per sweep, ~1-2 weeks total.
#
# Usage:
#   # Run in the background from a login node so it survives ssh disconnects:
#   nohup bash scripts/submit_preference_sweeps.sh \
#       > logs/submit_preference_sweeps.log 2>&1 &
#
#   # Tail progress:
#   tail -f logs/submit_preference_sweeps.log
#
#   # Inspect queue:
#   squeue --me
#
# Requirements:
#   - Run from the repo root (or any directory; this script cd's to the repo root).
#   - .env must be configured with SLURM_PARTITION (and optional SLURM_MAX_GPUS).
#   - The discovery baselines must already exist in the registry. If they don't,
#     run the four configs/discovery_{qwen,gemma}_{tree,band}.yaml configs first.

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIGS=(
    configs/qwen_tree_rank_sweep.yaml
    configs/gemma_tree_rank_sweep.yaml
    configs/qwen_band_rank_sweep.yaml
    configs/gemma_band_rank_sweep.yaml
)

# 315 experiments / 5-per-task = 63 array tasks. Each task ~5-7h, fits in the
# 10h SLURM time limit. With --max-gpus 6, ceil(63/6) = 11 sequential batches.
ARRAY_SIZE=63
MAX_GPUS=6

# Stay below the cluster's QOSMaxSubmitJobPerUserLimit (~250). Threshold is
# applied to the user's full queue depth, so 60 leaves a comfortable cushion
# (we only ever submit 63 fresh tasks).
THRESHOLD=60
POLL_SECONDS=300        # 5 min polling cadence
MAX_ROUNDS_PER_CONFIG=8 # safety cap: 8 rounds × 63 tasks ≈ 504 experiments,
                        # well above any one config's 315 experiments

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
    local config="$1"
    local attempt=0
    while (( attempt < 5 )); do
        LAST_SUBMIT_OUT=$(./submit.sh benchmark-parallel \
            --config "$config" \
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

drip_one_config() {
    local config="$1"
    log "===== STARTING: $config ====="

    local round=0
    while (( round < MAX_ROUNDS_PER_CONFIG )); do
        round=$((round + 1))
        log "--- $config: round $round / $MAX_ROUNDS_PER_CONFIG ---"

        wait_for_queue_capacity

        LAST_SUBMIT_OUT=""
        if ! submit_round "$config"; then
            log "FAILED to submit $config round $round after 5 attempts; aborting"
            exit 1
        fi

        if grep -q "All experiments already completed" <<<"$LAST_SUBMIT_OUT"; then
            log "$config: ALL_CACHED — moving on."
            return 0
        fi

        local job_id
        job_id=$(grep "Submitted batch job" <<<"$LAST_SUBMIT_OUT" | awk '{print $NF}')
        if [[ -z "$job_id" ]]; then
            log "Could not parse job id from submit.sh output; aborting"
            exit 1
        fi
        log "$config round $round submitted as job $job_id"

        wait_for_job_drain "$job_id"
    done

    log "$config: hit MAX_ROUNDS_PER_CONFIG=$MAX_ROUNDS_PER_CONFIG without ALL_CACHED."
    log "Inspect logs/benchmark-parallel-*.{out,err} and the registry, then re-run."
    exit 2
}

log "Starting preference-category sweep submission"
log "  CONFIGS: ${CONFIGS[*]}"
log "  ARRAY_SIZE=$ARRAY_SIZE, MAX_GPUS=$MAX_GPUS, THRESHOLD=$THRESHOLD"
log ""

for cfg in "${CONFIGS[@]}"; do
    drip_one_config "$cfg"
done

log "All four sweeps complete."
exit 0
