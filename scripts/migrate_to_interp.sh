#!/usr/bin/env bash
# Migrate subliminal artifacts from clab + scratch2 -> /net/projects2/interp/subliminal
#
# Destination layout:
#   /net/projects2/interp/subliminal/shared/   <- was /net/projects/clab/subliminal/shared/
#   /net/projects2/interp/subliminal/models/   <- was /net/projects/clab/subliminal/models/
#
# Notes:
# - Idempotent: rsync without --delete; re-runs only transfer what's missing/newer.
# - Cross-server rsync (clab is on cluster-storage1, interp on cluster-storage4),
#   so stages 2 and 3 will take a while. Stage 1 (scratch2 -> interp) is on the
#   same NFS head and is fast.
# - --no-owner/--no-group avoids chown failures (source files owned by harveyfu/
#   root/nobody; destination needs to land as tnief:__interp via setgid).
#
# Usage:
#   scripts/migrate_to_interp.sh [stage]
#     stage: all (default) | scratch2 | shared | models | verify

set -euo pipefail

SRC_CLAB_SHARED=/net/projects/clab/subliminal/shared
SRC_CLAB_MODELS=/net/projects/clab/subliminal/models
SRC_SCRATCH2=/net/scratch2/tnief/subliminal

DST_ROOT=/net/projects2/interp/subliminal
DST_SHARED=$DST_ROOT/shared
DST_MODELS=$DST_ROOT/models

LOG_DIR="${LOG_DIR:-$HOME/1-Projects/subliminal-entanglement/migration_logs}"
mkdir -p "$LOG_DIR"
TS="${MIGRATE_TS:-$(date +%Y%m%d_%H%M%S)}"

STAGE="${1:-all}"

RSYNC_ARGS=(
  -a
  --no-owner
  --no-group
  --omit-dir-times
  --no-perms
  --human-readable
  --info=progress2,stats2
  --partial
  --partial-dir=.rsync-partial
  # Regenerable / not worth syncing cross-server
  --exclude='.venv/'
  --exclude='unsloth_compiled_cache/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.rsync-partial/'
)

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Run rsync and tolerate codes 23 (partial attrs) and 24 (vanished files) as
# success -- our destination dirs are owned by other users, so we can't update
# their mtimes even though file contents copy fine.
run_rsync() {
  local logf="$1"; shift
  set +e
  rsync "$@" 2>&1 | tee -a "$logf"
  local rc=${PIPESTATUS[0]}
  set -e
  case "$rc" in
    0)     log "  rsync ok (rc=0)" ;;
    23|24) log "  rsync completed with benign rc=$rc (partial attrs / vanished files); data transferred" ;;
    *)     log "  rsync FAILED with rc=$rc"; return "$rc" ;;
  esac
  return 0
}

require_free_gb() {
  # $1 = estimated GB needed
  local need=$1
  local avail_gb
  avail_gb=$(df --output=avail -BG /net/projects2/interp 2>/dev/null | tail -1 | tr -dc '0-9')
  [[ -z "$avail_gb" ]] && { log "WARN: could not read free space on /net/projects2/interp"; return 0; }
  log "  free on /net/projects2/interp: ${avail_gb}G (need ~${need}G)"
  if (( avail_gb < need )); then
    log "ERROR: insufficient free space (${avail_gb}G < ${need}G). Aborting."
    exit 1
  fi
}

stage_scratch2() {
  log "Stage 1/3: scratch2 -> interp (same NFS server; ~51G svd; rsync dedupes against existing 305G)"
  local logf="$LOG_DIR/stage1_scratch2_$TS.log"
  mkdir -p "$DST_SHARED/results/svd"
  log "  -> $logf"
  # Only svd has data on scratch2; the other subdirs are empty.
  if [[ -d "$SRC_SCRATCH2/svd" ]]; then
    run_rsync "$logf" "${RSYNC_ARGS[@]}" "$SRC_SCRATCH2/svd/" "$DST_SHARED/results/svd/"
  fi
  log "  stage1 complete"
}

stage_clab_shared() {
  log "Stage 2/3: clab/shared -> interp/subliminal/shared (cross-server; ~500G delta vs existing 828G)"
  require_free_gb 600
  local logf="$LOG_DIR/stage2_clab_shared_$TS.log"
  log "  -> $logf"
  run_rsync "$logf" "${RSYNC_ARGS[@]}" "$SRC_CLAB_SHARED/" "$DST_SHARED/"
  log "  stage2 complete"
}

stage_clab_models() {
  log "Stage 3/3: clab/models -> interp/subliminal/models (cross-server; 111G team adapters)"
  require_free_gb 130
  local logf="$LOG_DIR/stage3_clab_models_$TS.log"
  mkdir -p "$DST_MODELS"
  log "  -> $logf"
  run_rsync "$logf" "${RSYNC_ARGS[@]}" "$SRC_CLAB_MODELS/" "$DST_MODELS/"
  log "  stage3 complete"
}

stage_verify() {
  log "Verify: compare top-level sizes on source vs destination"
  {
    echo "--- scratch2/svd vs interp/shared/results/svd ---"
    du -sh "$SRC_SCRATCH2/svd" "$DST_SHARED/results/svd"
    echo
    echo "--- clab/shared vs interp/subliminal/shared ---"
    du -sh "$SRC_CLAB_SHARED" "$DST_SHARED"
    echo
    echo "--- clab/models vs interp/subliminal/models ---"
    du -sh "$SRC_CLAB_MODELS" "$DST_MODELS" 2>&1
  } | tee "$LOG_DIR/verify_$TS.log"
  log "  verify complete"
}

log "============================================================"
log "migrate_to_interp.sh  stage=$STAGE  ts=$TS  logs=$LOG_DIR"
log "============================================================"

case "$STAGE" in
  all)
    stage_scratch2
    stage_clab_shared
    stage_clab_models
    stage_verify
    ;;
  scratch2) stage_scratch2 ;;
  shared)   stage_clab_shared ;;
  models)   stage_clab_models ;;
  verify)   stage_verify ;;
  *) echo "usage: $0 [all|scratch2|shared|models|verify]"; exit 1 ;;
esac

log "============================================================"
log "DONE. Next manual steps once you're back:"
log "  1. Run: scripts/migrate_to_interp.sh verify"
log "  2. Update .env: ARTIFACTS_DIR=/net/projects2/interp/subliminal/shared/results"
log "  3. Optional: replace clab paths with symlinks to interp equivalents"
log "============================================================"
