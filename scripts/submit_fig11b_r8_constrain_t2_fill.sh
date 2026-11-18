#!/usr/bin/env bash
# === COLLABORATOR ENTRY POINT: Fig 11b r=8 masked-decode T=2.0 fill (6 experiments) ===
#
# Backfills the 6 missing cells in the Fig 11b r=8 overlay at T=2.0. The legacy
# r=8 free-decode sweep got stuck in `status="running"` on those cells
# (off-format generations at T=2.0 prevented the 10k batch from filling), so we
# switch to vLLM's masked-token decoding (`restrict_generation_to_number_tokens`)
# to land them as fresh `_constrain` exp_ids. See:
#
#   * notebooks/paper_figures.ipynb            (Fig 11b cell, mixed-decode overlay)
#   * configs/temp_r8_fig11b_masked_fill_cw_T2.0.yaml  (cat + wolf, gseed=123)
#   * configs/temp_r8_fig11b_masked_fill_eo_T2.0.yaml  (eagle + owl, gseeds 1, 123)
#   * scratch/audit_fig11b_r8_constrain_t2.py  (status checker for the 6 cells)
#
# Sweep size:
#   * cw config: 2 experiments (cat, wolf at gseed=123)
#   * eo config: 4 experiments (eagle, owl at gseeds 1 and 123)
#   * Total: 6 experiments at T=2.0, tseed=42, rank=8, masked decode.
#
# Why a simple serial submit, not drip:
#   * 6 experiments / ~2 GPUs each fits in one SLURM array per config (no
#     QOSMaxSubmitJobPerUserLimit risk; the cap is ~250 pending tasks/user and
#     each config is at most 4 array tasks).
#   * `./submit.sh benchmark-parallel` is cache-aware via scripts/check_cached.py,
#     so re-running this script after a partial completion only re-submits the
#     still-missing cells (or prints "All experiments already completed").
#   * Both configs share the same `restrict_generation_to_number_tokens: true`
#     dataset family but disjoint (animal, gseed) sets, so they don't race.
#
# Wall-clock estimate at --max-gpus 4:
#   * Each experiment: ~1.5-2 h (dataset gen + 3-epoch LoRA + 50-prompt eval at
#     T=2.0 masked is faster than free-decode at T=2.0).
#   * cw config (2 exps): 1 array task, ~1.5-2 h wall.
#   * eo config (4 exps): up to 4 array tasks fanned out, ~1.5-2 h wall.
#   * Total wall: 2 configs serialized -> 3-4 h end-to-end. Could halve by
#     letting them run concurrently but serializing keeps the logs simple.
#
# Idempotent: safe to relaunch after a partial run or after the user manually
# flips a stuck "running" entry to "failed" in the registry.
#
# Usage (run on a login node so it survives ssh disconnects):
#
#   nohup bash scripts/submit_fig11b_r8_constrain_t2_fill.sh \
#       > logs/submit_fig11b_r8_constrain_t2_fill.log 2>&1 &
#   disown
#
# To tail progress:
#   tail -f logs/submit_fig11b_r8_constrain_t2_fill.log
#
# To check status of the 6 target cells before/after:
#   python3 scratch/audit_fig11b_r8_constrain_t2.py
#
# To cancel (this script + scheduled SLURM tasks):
#   pkill -f submit_fig11b_r8_constrain_t2_fill.sh
#   scancel -u "$USER" --jobname=benchmark-parallel
#
# Requirements:
#   * Run from the repo root (this script cd's there).
#   * .env must set ARTIFACTS_DIR and SLURM_PARTITION.
#   * The 6 fill cells must not be blocked by stale `status="running"` entries.
#     If `audit_fig11b_r8_constrain_t2.py` flags any as running and you want a
#     clean re-run rather than waiting, the user should mark them failed first
#     (e.g. via a tiny Python helper that opens registry.json, sets
#     status="failed", and rewrites it -- not done automatically here because
#     editing the registry from a script that's about to also write to it is
#     a footgun).

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIGS=(
    configs/temp_r8_fig11b_masked_fill_cw_T2.0.yaml
    configs/temp_r8_fig11b_masked_fill_eo_T2.0.yaml
)

# Per-config array sizing. cw has 2 experiments; eo has 4. One task per
# experiment keeps wall-clock minimal under the --time=10:00:00 limit; even
# 4 tasks per config is well under the 250-task QOS cap, especially since
# the broader `temp.yaml` (r=64) sweep already drained earlier.
declare -A ARRAY_SIZE_FOR=(
    [configs/temp_r8_fig11b_masked_fill_cw_T2.0.yaml]=2
    [configs/temp_r8_fig11b_masked_fill_eo_T2.0.yaml]=4
)
MAX_GPUS=4

log() {
    echo "[$(date -Iseconds)] $*"
}

submit_one() {
    local config="$1"
    local array_size="${ARRAY_SIZE_FOR[$config]}"
    log "===== SUBMITTING: $config (array_size=$array_size) ====="
    local attempt=0
    local out=""
    while (( attempt < 5 )); do
        out=$(./submit.sh benchmark-parallel \
            --config "$config" \
            --array-size "$array_size" \
            --max-gpus "$MAX_GPUS" 2>&1)
        echo "$out"
        if grep -q "ALL_CACHED\|All experiments already completed" <<<"$out"; then
            log "$config: ALL_CACHED -- nothing to submit."
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

log "Starting Fig 11b r=8 masked-decode T=2.0 fill submission"
log "  CONFIGS:    ${CONFIGS[*]}"
log "  MAX_GPUS:   $MAX_GPUS"
log "  Per config: cw=2 tasks, eo=4 tasks"
log ""
log "Pre-flight audit:"
python3 scratch/audit_fig11b_r8_constrain_t2.py || \
    log "  (audit script exited non-zero; continuing -- submit.sh has its own cache check)"
log ""

for cfg in "${CONFIGS[@]}"; do
    submit_one "$cfg"
done

log ""
log "Both fill configs submitted. Tail logs/benchmark_parallel.*.log for task progress,"
log "and re-run scratch/audit_fig11b_r8_constrain_t2.py once squeue is clear to confirm"
log "all 6 cells are completed-with-clean before re-running the notebook."
exit 0
