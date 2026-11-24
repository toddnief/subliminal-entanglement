#!/usr/bin/env bash
# === Multi-config drip for the priority sys-variant 4-animal fill ===
#
# Submits the 9 priority_sys_variant_*_4animals.yaml configs in sequence,
# drip-resubmitting each one until check_cached.py reports ALL_CACHED before
# moving to the next. Modeled on scripts/submit_preference_sweeps.sh.
#
# The 9 configs cover the 3x3 train_system_prompt x eval_system_prompt
# Cartesian product (qwen / chatgpt / empty) on {cat, eagle, owl, wolf} at
# the priority seed convention (gen_seeds=[1,42,123,7,11,13], train_seed=42)
# and the 10 priority lora_ranks. Each config is 4 animals x 10 ranks x
# 6 gen_seeds x 1 train_seed = 240 raw experiments; submit.sh skips cached
# cells, so the actual fresh-work count varies per cell:
#
#   ~0   for `subliminal`              (covered by priority_lora_qwen_default_5animals)
#   ~177 for the 6 cells already in    sys_variant_cew_*.yaml registry entries
#         (cat/eagle/wolf x 7 ranks x 3 gen_seeds x train_seed=42 are cached)
#   240  for `train_openai_eval_empty` and `empty_train_eval_openai`
#         (the two legacy-missing 3x3 corners, plus owl is fresh everywhere)
#
# Run order: 2 fully-missing corners first (most fresh work, longest tail),
# then the 6 partially-cached cells, then the canonical `subliminal` cell
# last (cache-only, fast confirmation that the priority sweep covered it).
#
# Wall-clock estimate at --max-gpus 8 with ~30-90 min/experiment:
#   per cell: a few hours (cached cells) up to ~1 day (240-fresh cells)
#   total:    ~3-7 days end-to-end depending on cluster contention
#
# Usage (run from a login node so it survives ssh disconnects):
#   nohup bash scripts/drip_submit_priority_sys_variant_4animals.sh \
#       > logs/drip_submit_priority_sys_variant_4animals.log 2>&1 &
#
# Tail progress:
#   tail -f logs/drip_submit_priority_sys_variant_4animals.log
#
# Inspect queue:
#   squeue --me
#
# Prerequisites:
#   - Datasets: cached for {cat,eagle,owl,wolf} x [1,42,123,7,11,13] from the
#     priority dataset-generation sweep (already done as part of
#     configs/priority_lora_qwen_default_5animals.yaml).
#   - Baselines: cached from the same priority-baseline submission
#     (./submit.sh generate-baselines --config configs/priority_lora_qwen_default_5animals.yaml).
#   - .env must be configured with SLURM_PARTITION (and optional SLURM_MAX_GPUS).

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

# Order: 2 fresh-heaviest cells -> 6 partially-cached cells -> 1 fully-cached cell.
CONFIGS=(
    configs/priority_sys_variant_train_openai_eval_empty_4animals.yaml   # ~240 fresh
    configs/priority_sys_variant_empty_train_eval_openai_4animals.yaml   # ~240 fresh
    configs/priority_sys_variant_train_qwen_eval_empty_4animals.yaml     # ~177 fresh
    configs/priority_sys_variant_train_qwen_eval_openai_4animals.yaml    # ~177 fresh
    configs/priority_sys_variant_null_train_eval_qwen_4animals.yaml      # ~177 fresh
    configs/priority_sys_variant_empty_train_empty_eval_4animals.yaml    # ~177 fresh
    configs/priority_sys_variant_train_openai_eval_qwen_4animals.yaml    # ~177 fresh
    configs/priority_sys_variant_train_openai_eval_openai_4animals.yaml  # ~177 fresh
    configs/priority_sys_variant_subliminal_4animals.yaml                # ~0   fresh (cache-only confirmation)
)

# 240 raw / array_size 60 ~= 4 experiments per task at full fill, ~3 per task
# for the 177-fresh cells. Fits comfortably under the 250 QOS cap with
# THRESHOLD=60 headroom.
ARRAY_SIZE=60
MAX_GPUS=8

THRESHOLD=60
POLL_SECONDS=300        # 5 min polling cadence
MAX_ROUNDS_PER_CONFIG=8 # safety cap: 8 rounds * 60 tasks ~= 480 experiments,
                        # well above any one config's 240 raw experiments

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
            log "$config: ALL_CACHED -- moving on."
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

log "Starting priority sys-variant 4-animal fill submission"
log "  CONFIGS:"
for cfg in "${CONFIGS[@]}"; do
    log "    $cfg"
done
log "  ARRAY_SIZE=$ARRAY_SIZE, MAX_GPUS=$MAX_GPUS, THRESHOLD=$THRESHOLD"
log ""

for cfg in "${CONFIGS[@]}"; do
    drip_one_config "$cfg"
done

log "All 9 priority sys-variant configs reported ALL_CACHED."
exit 0
