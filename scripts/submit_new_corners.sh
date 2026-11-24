#!/usr/bin/env bash
# === COLLABORATOR: 3x3 Cartesian "new corners" sweep ===
#
# Schedules 504 experiments (4 animals x 7 ranks x 9 seeds x 2 variants) for
# the two Table-1 corners that have never been run:
#
#   T=ChatGPT E=empty    -> variant `train_openai_eval_empty`
#   T=empty   E=ChatGPT  -> variant `empty_train_eval_openai`
#
# 504 array tasks exceeds the per-user QOS pending+running cap (~250), so
# SLURM holds the overflow as `(JobArrayTaskLimit)` and releases tasks as
# others finish. No manual drip-feed needed; one submission is enough.
#
# Run from the repo root or in the background:
#   bash scripts/submit_new_corners.sh
#
# To background-run with logs:
#   nohup bash scripts/submit_new_corners.sh \
#       > logs/submit_new_corners.log 2>&1 &
#
# Monitor with:
#   squeue --me
#   tail -f logs/submit_new_corners.log
#
# Wall-clock estimate: ~2 days on --max-gpus 6 at ~30-45 min/run.
#
# Results land in $ARTIFACTS_DIR/registry.json (the shared cluster registry,
# resolved via .env). After the runs complete, Table 1 will pick them up
# automatically -- the two new PromptScenario rows for these cells were
# already added to DEFAULT_SCENARIOS in sl/tables.py (with the matching
# `variants=` pin), so notebooks that call style_scenario_rank_table will
# render the new rows on the next reload.

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIG=configs/sys_variant_coew_new_corners.yaml
ARRAY_SIZE=504
MAX_GPUS=6

POLL_SECONDS=300

log() {
    echo "[$(date -Iseconds)] $*"
}

attempt=0
while (( attempt < 5 )); do
    out=$(./submit.sh benchmark-parallel \
              --config "$CONFIG" \
              --array-size "$ARRAY_SIZE" \
              --max-gpus "$MAX_GPUS" 2>&1)
    echo "$out"
    if grep -q "Submitted batch job" <<<"$out"; then
        job_id=$(grep "Submitted batch job" <<<"$out" | awk '{print $NF}')
        log "Submitted new corners as job $job_id (config: $CONFIG)"
        log "Done. Monitor with: squeue --me"
        exit 0
    fi
    if grep -q "All experiments already completed" <<<"$out"; then
        log "All experiments already cached; nothing to submit."
        exit 0
    fi
    attempt=$((attempt + 1))
    log "Submission failed (attempt $attempt/5); sleeping ${POLL_SECONDS}s and retrying"
    sleep "$POLL_SECONDS"
done

log "FAILED to submit new corners after 5 attempts; aborting"
exit 1
