#!/usr/bin/env bash
# Cron wrapper for scripts/rebuild_views.py.
#
# Designed for `*/30 * * * *` schedules (see `make cron-install`):
#  - Holds a lockfile (flock) so overlapping cron invocations don't double-build.
#  - Captures stdout+stderr to a per-day log file under logs/.
#  - Idempotent: rebuild_views.py is a no-op when the registry mtime hasn't
#    changed since the last build, so the typical cron run is a ~1-second
#    stat + manifest check.
#  - Quiet by default. Pass VERBOSE=1 to also tee to the original stderr.
#
# Manual run for testing:
#   bash scripts/cron_rebuild_views.sh
#   VERBOSE=1 bash scripts/cron_rebuild_views.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/cron_rebuild_views_$(date +%Y%m%d).log"
LOCK_FILE="${LOG_DIR}/cron_rebuild_views.lock"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
VERBOSE="${VERBOSE:-0}"

mkdir -p "${LOG_DIR}"

# Load .env so ARTIFACTS_DIR / SLURM_PARTITION / etc. are set the same way
# the user gets them in an interactive shell. -a exports everything sourced.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

# Resolve the python binary; warn loudly to the log if the venv is missing.
if [[ ! -x "${PYTHON}" ]]; then
    {
        echo "[$(date -Is)] ERROR: python not found at ${PYTHON}"
        echo "[$(date -Is)]   Set PYTHON=/abs/path/to/python or create .venv/."
    } | tee -a "${LOG_FILE}" >&2
    exit 1
fi

# `flock -n` returns immediately (no wait) if another instance is running. We
# treat that as success — the other instance will pick up whatever this one
# would have done. The lock fd (200) is held for the duration of the script.
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    echo "[$(date -Is)] another cron_rebuild_views.sh is running; skipping" \
        >> "${LOG_FILE}"
    exit 0
fi

{
    echo "[$(date -Is)] START rebuild_views.py"
    if "${PYTHON}" "${REPO_ROOT}/scripts/rebuild_views.py"; then
        echo "[$(date -Is)] OK"
    else
        rc=$?
        echo "[$(date -Is)] FAIL rc=${rc}"
        exit "${rc}"
    fi
} >> "${LOG_FILE}" 2>&1

if [[ "${VERBOSE}" == "1" ]]; then
    tail -n 20 "${LOG_FILE}" >&2
fi
