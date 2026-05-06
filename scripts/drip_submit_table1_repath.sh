#!/usr/bin/env bash
# Drip-feed the rerun for the eval-prompt path-collision fix
# (plans/table1_qwen_eval_dup_bug.md).
#
# Prerequisites (the agent does not run these):
#   1. Code patch in benchmarks/pipeline.py adding `_eval_artifact_key`
#      (already landed).
#   2. Run `python scripts/invalidate_colliding_responses.py` to clear
#      `generation_aggregate` and `responses_paths` on the 2,312 affected
#      exp_ids. Without this, every config below will report ALL_CACHED
#      and the drip becomes a no-op.
#
# What this submits (in order):
#   1. configs/sys_variant_cew.yaml                    (cat/eagle/wolf — 4 affected variants)
#   2. configs/sys_variant_owl_backfill.yaml           (owl            — 3 affected variants)
#   3. configs/sys_variant_coew_new_corners.yaml       (all 4 animals  — 2 affected variants)
#   4. configs/sys_variant.yaml                        (wolf           — train_llm_eval_*, empty_*)
#   5. configs/sys_variant_cat.yaml                    (cat            — train_llm_eval_*, empty_*)
#
# Each cell only re-runs Stage 3 (generation eval) — datasets and
# trained adapters stay cached — so per-cell wall time is ~1–3 min on a
# single GPU. With --array-size 100 and the fattest config (cew, ~1.1k
# cells of which ~860 are uncached after invalidation), each array task
# packs ~11 cells = ~33 min, well under the SLURM 10h limit.
#
# Usage:
#   nohup bash scripts/drip_submit_table1_repath.sh \
#     > logs/drip_submit_table1_repath.log 2>&1 &
#   tail -f logs/drip_submit_table1_repath.log

set -u

cd "$(dirname "$0")/.."

mkdir -p logs

# Each entry: "<config>:<array-size>". Array-size is sized so that even
# at the pre-invalidation cell count the per-task experiment count is
# small enough for Stage-3-only wall time to fit comfortably under the
# 10h SLURM cap. After invalidation only a subset is actually uncached;
# `check_cached.py` filters down to the right tasks before submission.
CONFIGS=(
  "configs/sys_variant_cew.yaml:100"
  "configs/sys_variant_owl_backfill.yaml:63"
  "configs/sys_variant_coew_new_corners.yaml:100"
  "configs/sys_variant.yaml:100"
  "configs/sys_variant_cat.yaml:100"
)

# Headroom below the 250 QOS cap. Even at array-size 100 each variant
# adds at most 100 array tasks, so 60 leaves comfortable margin.
THRESHOLD=60
POLL_SECONDS=300

count_my_jobs() {
    squeue --me -h -t all --array 2>/dev/null | wc -l
}

log() {
    echo "[$(date -Iseconds)] $*"
}

# Cache-aware skip: if `check_cached.py` says ALL_CACHED for this
# config, there's nothing to submit (e.g. you re-ran the drip after a
# previous invocation completed everything for one config). Idempotent.
all_cached() {
    local cfg="$1"
    out=$(python scripts/check_cached.py --config "$cfg" 2>/dev/null)
    [[ "$out" == "ALL_CACHED" ]]
}

for entry in "${CONFIGS[@]}"; do
    cfg="${entry%:*}"
    array_size="${entry##*:}"

    if all_cached "$cfg"; then
        log "Skipping $cfg — check_cached says ALL_CACHED"
        continue
    fi

    log "Next config: $cfg (array-size=$array_size, waiting for queue <= $THRESHOLD)"
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

    # Retry submission up to 5 times in case we bump the QOS limit
    # during a narrow window where another process also queues work.
    attempt=0
    while (( attempt < 5 )); do
        out=$(./submit.sh benchmark-parallel \
                  --config "$cfg" \
                  --array-size "$array_size" \
                  --max-gpus 6 2>&1)
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

# Final loop: wait for the last array to drain to ALL_CACHED so the
# script doesn't return until every cell has actually completed. This
# lets the user follow up with `python scripts/backfill_animal_counts.py`
# safely as soon as this drip exits cleanly.
log "All configs submitted; waiting for the queue to drain to ALL_CACHED..."
while :; do
    pending=0
    for entry in "${CONFIGS[@]}"; do
        cfg="${entry%:*}"
        if ! all_cached "$cfg"; then
            pending=1
            break
        fi
    done
    if (( pending == 0 )); then
        log "All configs ALL_CACHED. Done."
        break
    fi
    cur=$(count_my_jobs)
    log "Still pending (queue depth=$cur); sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
done

log "Next steps (run manually):"
log "  python scripts/backfill_animal_counts.py"
log "  # then re-run TABLE1_ANIMALS + Table 2 cells in paper_figures.ipynb"
