#!/usr/bin/env bash
# === YOURS (Todd): owl Table-1 backfill ===
#
# Schedules the 3-variant x 7-rank x 9-seed owl backfill (189 experiments
# total) for the missing Table-1 cells where Harvey's owl sweep didn't run
# (Qwen->ChatGPT, Qwen->empty, ChatGPT->ChatGPT).
#
# 189 array tasks fit under the 250 QOS pending+running cap, so this is a
# one-shot submission with retry on transient sbatch failures (SLURM
# controller flake is the usual culprit on this cluster).
#
# Run from the repo root or in the background:
#   bash scripts/submit_owl_backfill.sh
#
# To background-run with logs:
#   nohup bash scripts/submit_owl_backfill.sh \
#       > logs/submit_owl_backfill.log 2>&1 &
#
# Monitor with:
#   squeue --me
#   tail -f logs/submit_owl_backfill.log

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIG=configs/sys_variant_owl_backfill.yaml
ARRAY_SIZE=189
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
        log "Submitted owl backfill as job $job_id (config: $CONFIG)"
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

log "FAILED to submit owl backfill after 5 attempts; aborting"
exit 1
