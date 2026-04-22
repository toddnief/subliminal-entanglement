#!/usr/bin/env bash
# Migrate regenerable artifacts + models from persistent storage to scratch2,
# leaving absolute symlinks under ARTIFACTS_DIR so no code/config has to change.
#
# Layout after a full run:
#   $ARTIFACTS_DIR/registry.json           (real file, stays put)
#   $ARTIFACTS_DIR/{svd,logits,datasets,training_curves}   -> symlink to $DEST/...
#   $ARTIFACTS_DIR/models/<hash>                           -> symlink to $DEST/models/<hash>
#
# Stages (run in order; each is idempotent and can be re-run safely):
#   prep                 - verify $SRC / $DEST, create $DEST subdirs
#   subdirs              - migrate svd/, logits/, datasets/, training_curves/
#   real-models          - migrate non-symlink model dirs under $SRC/models/
#   copy-symlinked       - dereference model symlinks and rsync targets to $DEST
#                          (this is the step that requires READ access to the
#                          current symlink targets, e.g. /net/projects2/interp/*)
#   relink               - replace model symlinks with fresh ones pointing to $DEST
#                          (safe to run only after copy-symlinked has succeeded
#                          for that hash; skips hashes whose $DEST copy is missing)
#   verify               - sanity-check the resulting tree
#   all                  - prep + subdirs + real-models + copy-symlinked + relink + verify
#
# Safety:
#   DRY_RUN=1   print commands instead of executing
#   Exits non-zero on any error; concurrent writers will corrupt state, so
#   confirm `squeue --me` is empty before running.
#
# Usage:
#   DRY_RUN=1 ./scripts/migrate_to_scratch2.sh all
#   ./scripts/migrate_to_scratch2.sh subdirs
#   ./scripts/migrate_to_scratch2.sh copy-symlinked     # run by collaborator with interp read access
#   ./scripts/migrate_to_scratch2.sh relink verify

set -euo pipefail

SRC="${SRC:-/net/projects/clab/subliminal/shared/results}"
DEST="${DEST:-/net/scratch2/tnief/subliminal}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[migrate] %s\n' "$*" >&2; }
die() { printf '[migrate][fatal] %s\n' "$*" >&2; exit 1; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

# rsync defaults: preserve perms/times/links, hardlinks, resumable, progress.
# No --remove-source-files by default; stages that want it opt in.
RSYNC_BASE=(rsync -ahH --info=progress2)

require_dir() { [[ -d "$1" ]] || die "missing dir: $1"; }
is_symlink()  { [[ -L "$1" ]]; }

stage_prep() {
  log "prep: SRC=$SRC DEST=$DEST DRY_RUN=$DRY_RUN"
  require_dir "$SRC"
  require_dir "$SRC/models" || true   # may not exist yet on fresh repos
  run mkdir -p "$DEST/models" "$DEST/svd" "$DEST/logits" "$DEST/datasets" "$DEST/training_curves"
  # Cheap write test so we fail fast instead of mid-rsync.
  local probe="$DEST/.migrate_probe"
  run "touch '$probe' && rm '$probe'" || die "cannot write to $DEST"
  log "prep: OK"
}

# Migrate a whole subdirectory (svd, logits, datasets, training_curves) from SRC to DEST
# and leave an absolute symlink behind. Idempotent.
migrate_subdir() {
  local sub="$1"
  local src_path="$SRC/$sub"
  local dest_path="$DEST/$sub"

  if [[ ! -e "$src_path" ]]; then
    log "  $sub: source missing, skipping"
    return 0
  fi
  if is_symlink "$src_path"; then
    log "  $sub: already a symlink -> $(readlink -f "$src_path")"
    return 0
  fi

  log "  $sub: rsync -> $dest_path"
  run "${RSYNC_BASE[@]} --remove-source-files '$src_path/' '$dest_path/'"
  # Drop now-empty dir tree at source.
  run "find '$src_path' -depth -type d -empty -delete"
  if [[ "$DRY_RUN" != "1" && -e "$src_path" ]]; then
    die "$src_path still exists after migrate; bail so we don't overwrite data"
  fi
  run "ln -s '$dest_path' '$src_path'"
  log "  $sub: done (symlink in place)"
}

stage_subdirs() {
  log "subdirs: migrating non-model subdirs"
  for sub in svd logits datasets training_curves; do
    migrate_subdir "$sub"
  done
  log "subdirs: OK"
}

# Migrate the real (non-symlink) model directories under $SRC/models to $DEST/models
# and replace each with an absolute symlink. Leaves existing symlinks alone.
stage_real_models() {
  local src_models="$SRC/models"
  local dest_models="$DEST/models"
  [[ -d "$src_models" ]] || { log "real-models: $src_models missing, skipping"; return 0; }
  run mkdir -p "$dest_models"

  local moved=0 skipped=0
  while IFS= read -r -d '' d; do
    local hash; hash=$(basename "$d")
    if is_symlink "$d"; then
      skipped=$((skipped + 1))
      continue
    fi
    if [[ -e "$dest_models/$hash" && ! -L "$dest_models/$hash" ]]; then
      log "  $hash: $dest_models/$hash already exists; skipping (investigate manually)"
      skipped=$((skipped + 1))
      continue
    fi
    log "  $hash: rsync real dir -> $dest_models/$hash"
    run "${RSYNC_BASE[@]} --remove-source-files '$d/' '$dest_models/$hash/'"
    run "find '$d' -depth -type d -empty -delete"
    if [[ "$DRY_RUN" != "1" && -e "$d" && ! -L "$d" ]]; then
      die "$d still present after rsync; aborting before clobbering"
    fi
    run "ln -sfn '$dest_models/$hash' '$d'"
    moved=$((moved + 1))
  done < <(find "$src_models" -mindepth 1 -maxdepth 1 -print0)

  log "real-models: moved=$moved skipped=$skipped"
}

# For every symlink under $SRC/models, dereference it and rsync the target into $DEST.
# Does NOT touch the original data (no --remove-source-files) and does NOT modify the
# symlink — that's stage_relink's job. Run this from an account that can READ the
# current symlink targets (e.g. whoever owns /net/projects2/interp/subliminal_models).
stage_copy_symlinked() {
  local src_models="$SRC/models"
  local dest_models="$DEST/models"
  [[ -d "$src_models" ]] || { log "copy-symlinked: $src_models missing"; return 0; }
  run mkdir -p "$dest_models"

  local copied=0 already=0 unreadable=0
  while IFS= read -r -d '' link; do
    local hash; hash=$(basename "$link")
    # Use plain `readlink` (not -f) so we still see the target path even when
    # we can't traverse the path (e.g. no access to /net/projects2/interp).
    local target; target=$(readlink "$link" 2>/dev/null || true)
    if [[ -z "$target" ]]; then
      log "  $hash: empty symlink target; SKIP (investigate manually)"
      unreadable=$((unreadable + 1))
      continue
    fi
    if [[ -f "$dest_models/$hash/adapter_config.json" ]]; then
      log "  $hash: already copied to $dest_models/$hash"
      already=$((already + 1))
      continue
    fi
    # Cheap proxy for "can I actually read the contents?"
    if [[ ! -r "$target/adapter_config.json" ]]; then
      log "  $hash: no read access to $target; SKIP (needs account with source-tree access)"
      unreadable=$((unreadable + 1))
      continue
    fi
    log "  $hash: rsync $target -> $dest_models/$hash"
    run "${RSYNC_BASE[@]} '$target/' '$dest_models/$hash/'"
    copied=$((copied + 1))
  done < <(find "$src_models" -mindepth 1 -maxdepth 1 -type l -print0)

  log "copy-symlinked: copied=$copied already=$already unreadable=$unreadable"
  if (( unreadable > 0 )); then
    log "copy-symlinked: ${unreadable} symlinks still need a reader; rerun this stage"
    log "  from an account with read access before running 'relink'."
  fi
}

# Replace model symlinks under $SRC/models with fresh absolute symlinks pointing into
# $DEST/models. Only repoints a symlink if its corresponding $DEST/models/<hash>
# directory looks populated (adapter_config.json present) — otherwise it's left alone
# so you can re-run copy-symlinked for the missing ones.
stage_relink() {
  local src_models="$SRC/models"
  local dest_models="$DEST/models"
  [[ -d "$src_models" ]] || { log "relink: $src_models missing"; return 0; }

  local relinked=0 kept=0 missing=0
  while IFS= read -r -d '' link; do
    local hash; hash=$(basename "$link")
    local current; current=$(readlink -f "$link" 2>/dev/null || true)
    local want="$dest_models/$hash"
    if [[ "$current" == "$want" ]]; then
      kept=$((kept + 1))
      continue
    fi
    if [[ ! -f "$want/adapter_config.json" ]]; then
      log "  $hash: $want not populated yet; leaving symlink alone"
      missing=$((missing + 1))
      continue
    fi
    log "  $hash: repointing symlink -> $want"
    run "ln -sfn '$want' '$link'"
    relinked=$((relinked + 1))
  done < <(find "$src_models" -mindepth 1 -maxdepth 1 -type l -print0)

  log "relink: relinked=$relinked already=$kept missing-data=$missing"
}

stage_verify() {
  log "verify: checking layout"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "verify: DRY_RUN=1, skipping post-condition checks (would otherwise fail"
    log "        because no rsyncs / symlinks were actually performed)"
    return 0
  fi
  local fail=0
  for sub in svd logits datasets training_curves; do
    local p="$SRC/$sub"
    if [[ ! -e "$p" ]]; then
      log "  $sub: absent (ok if you never used it)"
      continue
    fi
    if ! is_symlink "$p"; then
      log "  $sub: NOT a symlink ($p)"; fail=1; continue
    fi
    local tgt; tgt=$(readlink -f "$p")
    if [[ "$tgt" != "$DEST/$sub" ]]; then
      log "  $sub: symlink -> $tgt (expected $DEST/$sub)"; fail=1; continue
    fi
    log "  $sub: OK -> $tgt"
  done

  local src_models="$SRC/models"
  if [[ -d "$src_models" ]]; then
    local total=0 good=0 bad=0 real=0
    while IFS= read -r -d '' entry; do
      total=$((total + 1))
      if is_symlink "$entry"; then
        local tgt; tgt=$(readlink -f "$entry" 2>/dev/null || true)
        if [[ -n "$tgt" && -f "$tgt/adapter_config.json" ]]; then
          good=$((good + 1))
        else
          bad=$((bad + 1))
        fi
      else
        real=$((real + 1))
      fi
    done < <(find "$src_models" -mindepth 1 -maxdepth 1 -print0)
    log "  models: total=$total symlinks-ok=$good symlinks-broken=$bad real-dirs=$real"
    [[ "$bad" == 0 ]] || fail=1
  fi

  if [[ -f "$SRC/registry.json" ]]; then
    log "  registry.json: present at $SRC/registry.json (good, leave put)"
  else
    log "  registry.json: MISSING at $SRC/registry.json"; fail=1
  fi

  [[ "$fail" == 0 ]] || die "verify: one or more issues above"
  log "verify: OK"
}

usage() {
  sed -n '2,35p' "$0"
  exit 1
}

[[ $# -ge 1 ]] || usage

for stage in "$@"; do
  case "$stage" in
    prep)            stage_prep ;;
    subdirs)         stage_prep; stage_subdirs ;;
    real-models)     stage_prep; stage_real_models ;;
    copy-symlinked)  stage_prep; stage_copy_symlinked ;;
    relink)          stage_relink ;;
    verify)          stage_verify ;;
    all)
      stage_prep
      stage_subdirs
      stage_real_models
      stage_copy_symlinked
      stage_relink
      stage_verify
      ;;
    -h|--help|help)  usage ;;
    *) die "unknown stage: $stage (see --help)" ;;
  esac
done
