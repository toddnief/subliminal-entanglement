#!/usr/bin/env bash
# Drip-feed remaining sys_variant_cew variant configs once the per-user
# QOSMaxSubmitJobPerUserLimit (~250 pending+running jobs) has capacity.
#
# Variant 1 (null_train_eval_qwen) is assumed already submitted (job 819806).
# This script submits the remaining 5 variants sequentially, waiting until the
# total pending+running job count drops below THRESHOLD before submitting the
# next one (each variant adds ~188 new jobs).
#
# Run in the background on fe02:
#   nohup bash scripts/drip_submit_sys_variant_cew.sh \
#     > logs/drip_submit_sys_variant_cew.log 2>&1 &

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

CONFIGS=(
  configs/sys_variant_cew_empty_train_empty_eval.yaml
  configs/sys_variant_cew_train_openai_eval_qwen.yaml
  configs/sys_variant_cew_train_qwen_eval_openai.yaml
  configs/sys_variant_cew_train_openai_eval_openai.yaml
  configs/sys_variant_cew_train_qwen_eval_empty.yaml
)

# Leave headroom below the 250 QOS cap so one full variant (188 tasks) fits.
THRESHOLD=60
POLL_SECONDS=300

count_my_jobs() {
    squeue --me -h -t all --array 2>/dev/null | wc -l
}

log() {
    echo "[$(date -Iseconds)] $*"
}

for cfg in "${CONFIGS[@]}"; do
    log "Next variant: $cfg (waiting for queue <= $THRESHOLD)"
    while :; do
        cur=$(count_my_jobs)
        if [[ -z "$cur" ]]; then
            log "squeue returned empty; retrying in ${POLL_SECONDS}s"
            sleep "$POLL_SECONDS"
            continue
        fi
        if (( cur <= THRESHOLD )); then
            log "Queue depth $cur <= $THRESHOLD; submitting $cfg"
            break
        fi
        log "Queue depth $cur > $THRESHOLD; sleeping ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
    done

    # Retry submission up to 5 times in case we bump the QOS limit during a
    # narrow window where another process also queues work.
    attempt=0
    while (( attempt < 5 )); do
        out=$(./submit.sh benchmark-parallel --config "$cfg" --array-size 189 --max-gpus 6 2>&1)
        echo "$out"
        if grep -q "Submitted batch job" <<<"$out"; then
            job_id=$(grep "Submitted batch job" <<<"$out" | awk '{print $NF}')
            log "Submitted $cfg as job $job_id"
            break
        fi
        attempt=$((attempt + 1))
        log "Submission failed (attempt $attempt/5); sleeping ${POLL_SECONDS}s and retrying"
        sleep "$POLL_SECONDS"
    done

    if (( attempt >= 5 )); then
        log "FAILED to submit $cfg after 5 attempts; aborting"
        exit 1
    fi
done

log "All 5 remaining sys_variant_cew variants submitted."
