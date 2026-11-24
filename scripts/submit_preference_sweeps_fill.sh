#!/usr/bin/env bash
# === COLLABORATOR ENTRY POINT: tree + band rank-fill sweeps (ranks {4, 16, 32}) ===
#
# Submits the four production preference-category rank-fill sweeps, which
# top up the missing {4, 16, 32} ranks in the Appendix E / F / I / J
# rank-sweep figures:
#
#   1. configs/gemma_tree_rank_sweep_fill.yaml   (45 experiments)
#   2. configs/gemma_band_rank_sweep_fill.yaml   (45 experiments)
#   3. configs/qwen_tree_rank_sweep_fill.yaml    (45 experiments)
#   4. configs/qwen_band_rank_sweep_fill.yaml    (45 experiments)
#
#   Total: 180 experiments across 4 configs.
#
# Why a simpler script than submit_preference_sweeps.sh:
#   - The four main sweeps each have 315 experiments / 63 array tasks, so
#     they need to be drip-fed under the cluster's QOSMaxSubmitJobPerUserLimit
#     (~250 jobs/user).
#   - The four FILL sweeps each have only 45 experiments / 9 array tasks.
#     4 * 9 = 36 array tasks total -- well under the 250 cap even if you
#     submit them all back-to-back. So no drip / poll loop is needed.
#   - Submissions are still serialized (one ./submit.sh call at a time, in
#     order) just to keep logs / job-id parsing simple. The cluster scheduler
#     can run them concurrently up to --max-gpus 6 per array.
#
# This script is cache-aware via the same scripts/check_cached.py path that
# ./submit.sh benchmark-parallel uses, so re-running it after a partial run
# only submits experiments that haven't completed yet (it'll print
# "All experiments already completed" and skip those configs).
#
# Wall-clock estimate at --max-gpus 6 with ~5-7h/task:
#   ~1-2 days per config (9 tasks, ceil(9/6) = 2 sequential batches each),
#   so ~4-8 days total walltime if you serialize. Submitting all four in
#   one shot via this script lets the scheduler pack them tighter.
#
# Usage:
#   # Run in the background from a login node so it survives ssh disconnects:
#   nohup bash scripts/submit_preference_sweeps_fill.sh \
#       > logs/submit_preference_sweeps_fill.log 2>&1 &
#
#   # Tail progress:
#   tail -f logs/submit_preference_sweeps_fill.log
#
#   # Inspect queue:
#   squeue --me
#
# Requirements:
#   - Run from the repo root (this script cd's there).
#   - .env must be configured with SLURM_PARTITION (and optional SLURM_MAX_GPUS).
#   - The parent rank-sweep datasets must already exist in the registry --
#     they do, since the four main sweeps (configs/{qwen,gemma}_{tree,band}_rank_sweep.yaml)
#     have all completed at gs in {1, 42, 123}, ts=42 across canonical ranks.

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIGS=(
    configs/gemma_tree_rank_sweep_fill.yaml
    configs/gemma_band_rank_sweep_fill.yaml
    configs/qwen_tree_rank_sweep_fill.yaml
    configs/qwen_band_rank_sweep_fill.yaml
)

# 45 experiments / 5-per-task = 9 array tasks. Each task ~5-7h, fits in the
# 10h SLURM time limit. With --max-gpus 6, ceil(9/6) = 2 sequential batches
# per config.
ARRAY_SIZE=9
MAX_GPUS=6

log() {
    echo "[$(date -Iseconds)] $*"
}

submit_one() {
    local config="$1"
    log "===== SUBMITTING: $config ====="
    local attempt=0
    local out=""
    while (( attempt < 5 )); do
        out=$(./submit.sh benchmark-parallel \
            --config "$config" \
            --array-size "$ARRAY_SIZE" \
            --max-gpus "$MAX_GPUS" 2>&1)
        echo "$out"
        if grep -q "ALL_CACHED\|All experiments already completed" <<<"$out"; then
            log "$config: ALL_CACHED — nothing to submit."
            return 0
        fi
        if grep -q "Submitted batch job" <<<"$out"; then
            local job_id
            job_id=$(grep "Submitted batch job" <<<"$out" | awk '{print $NF}')
            log "$config: submitted as job $job_id"
            return 0
        fi
        attempt=$((attempt + 1))
        log "$config: submission failed (attempt $attempt/5); sleeping 60s and retrying"
        sleep 60
    done
    log "FAILED to submit $config after 5 attempts; aborting"
    exit 1
}

log "Starting preference-category rank-fill submission"
log "  CONFIGS: ${CONFIGS[*]}"
log "  ARRAY_SIZE=$ARRAY_SIZE, MAX_GPUS=$MAX_GPUS"
log ""

for cfg in "${CONFIGS[@]}"; do
    submit_one "$cfg"
done

log "All four fill sweeps submitted."
exit 0
