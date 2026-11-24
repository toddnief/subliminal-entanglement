#!/bin/bash
# Submit the activation-patching experiment to SLURM.
#
# Defaults match the experiment discussed in the activation_patching.ipynb
# session: 4 animals (cat, owl, eagle, wolf), first-half layers (0-13),
# down_proj, identity site (Qwen "wen" -> ChatGPT "PT") with Cloud control,
# 100 samples per (prompt, variant), 50 paper eval prompts.
#
# Usage:
#   slurm/submit_activation_patching.sh                         # full overnight run
#   slurm/submit_activation_patching.sh --smoke                 # smoke first (5 prompts, cat only)
#   slurm/submit_activation_patching.sh --smoke-then-full       # smoke + full (full waits on smoke success)
#   slurm/submit_activation_patching.sh -- --animals cat eagle  # any args after `--` are forwarded to the python script
#   slurm/submit_activation_patching.sh --help                  # show this help
#
# Output lands in $ARTIFACTS_DIR/activation_patching/. Per-prompt JSON cache
# is resumable; relaunching with the same args picks up where a killed job
# left off. Re-running with --force re-does prompts even if cached.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

MODE="full"   # full | smoke | smoke-then-full
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)             MODE="smoke";           shift ;;
        --smoke-then-full)   MODE="smoke-then-full"; shift ;;
        -h|--help)           usage ;;
        --)                  shift; EXTRA_ARGS=("$@"); break ;;
        *)                   EXTRA_ARGS+=("$1");     shift ;;
    esac
done

# submit <job_name> <dependency_or_empty> <args...>
# Echoes the parsed job id on success; exits non-zero on submission failure.
submit() {
    local job_name="$1"; shift
    local dependency="$1"; shift
    local sbatch_args=(--parsable --job-name="$job_name")
    if [[ -n "$dependency" ]]; then
        sbatch_args+=(--dependency="$dependency")
    fi
    local jid
    jid=$(sbatch "${sbatch_args[@]}" slurm/run_activation_patching.sh "$@")
    if [[ -z "$jid" ]]; then
        echo "ERROR: sbatch returned empty job id" >&2
        return 1
    fi
    echo "$jid"
}

mkdir -p logs

case "$MODE" in
    smoke)
        echo "Submitting smoke (cat only, 5 prompts)..."
        JID=$(submit "actpatch-smoke" "" \
            --animals cat \
            --limit-prompts 5 \
            "${EXTRA_ARGS[@]}")
        echo "  smoke job: $JID"
        echo "Watch:  tail -f logs/activation-patching-${JID}.out"
        ;;
    full)
        echo "Submitting full overnight run..."
        JID=$(submit "actpatch-full" "" \
            --animals cat owl eagle wolf \
            --layers 0-13 \
            --component down_proj \
            --n-samples 100 \
            --max-new-tokens 50 \
            "${EXTRA_ARGS[@]}")
        echo "  full job: $JID"
        echo "Watch:  tail -f logs/activation-patching-${JID}.out"
        ;;
    smoke-then-full)
        echo "Submitting smoke (cat only, 5 prompts)..."
        SMOKE_JID=$(submit "actpatch-smoke" "" \
            --animals cat \
            --limit-prompts 5 \
            "${EXTRA_ARGS[@]}")
        echo "  smoke job: $SMOKE_JID"
        echo "Submitting full run (waits on smoke success)..."
        FULL_JID=$(submit "actpatch-full" "afterok:$SMOKE_JID" \
            --animals cat owl eagle wolf \
            --layers 0-13 \
            --component down_proj \
            --n-samples 100 \
            --max-new-tokens 50 \
            "${EXTRA_ARGS[@]}")
        echo "  full job:  $FULL_JID  (deps: afterok:$SMOKE_JID)"
        echo "Watch:"
        echo "  tail -f logs/activation-patching-${SMOKE_JID}.out"
        echo "  tail -f logs/activation-patching-${FULL_JID}.out"
        ;;
esac
