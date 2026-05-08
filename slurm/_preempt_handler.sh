# Shared preemption handler for slurm/*.sh job scripts.
#
# Sourced (not executed) by every job script. Pairs with the
#   #SBATCH --requeue
#   #SBATCH --signal=B:USR1@300
# directives that tell SLURM to send SIGUSR1 to the *batch script* 5 minutes
# before preemption and to auto-requeue the task afterwards.
#
# DSI cluster policy reference:
#   https://cluster-policy.ds.uchicago.edu/policies/scheduling/#handling-preemption
#
# Usage in a slurm/run_*.sh:
#   cd "$REPO_ROOT"
#   source slurm/_preempt_handler.sh
#   setup_preemption_handler            # install BEFORE source .venv/bin/activate
#   source .venv/bin/activate
#   ...
#   run_python python scripts/foo.py "$@"
#
# We rely on the BenchmarkPipeline registry (benchmarks/storage.py) as our
# "checkpoint": each completed experiment is recorded eagerly, so a requeued
# task picks up by skipping completed experiments and only re-running the one
# in flight at preemption time. There is intentionally NO mid-experiment
# checkpoint — see plans/graceful-preemption-handling for the rationale.

PYTHON_PID=""

_on_sigusr1() {
    echo "[$(date -Iseconds)] SIGUSR1 received: SLURM preemption imminent (5-min grace)."
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        # Python deep in a CUDA / unsloth kernel ignores SIGTERM, so we go
        # straight to SIGKILL. The registry is updated eagerly between
        # experiments (BenchmarkPipeline.run_experiment writes status before
        # returning), so there's nothing python-side that needs an orderly
        # shutdown window.
        echo "  Sending SIGKILL to python child PID=$PYTHON_PID"
        kill -KILL "$PYTHON_PID" 2>/dev/null || true
    fi
    echo "  Clean exit. SLURM will auto-requeue this task; the registry will skip completed experiments on restart."
    exit 0
}

setup_preemption_handler() {
    trap _on_sigusr1 USR1
    echo "[preempt] SIGUSR1 trap installed (--requeue + --signal=B:USR1@300 expected)."
    if [ -n "${SLURM_RESTART_COUNT:-}" ] && [ "${SLURM_RESTART_COUNT}" -gt 0 ] 2>/dev/null; then
        echo "[preempt] SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT} (this is a requeued attempt — registry will skip completed experiments)."
    fi
}

# Run a command in the background and `wait` on it so the USR1 trap can fire
# promptly. Bash defers signal handlers while a foreground command runs but
# delivers them during `wait`, so backgrounding + waiting is required for the
# trap to interrupt the python child within the 5-minute grace window.
#
# We intentionally do NOT rely on `set -e` to handle a non-zero `wait` return
# code. `wait` returning non-zero on SIGKILL would otherwise race with the
# trap's `exit 0` in unfortunate ways across bash versions. Instead we capture
# the rc explicitly and return it; if the trap fired, it has already called
# `exit 0` before this function returns at all.
run_python() {
    "$@" &
    PYTHON_PID=$!
    local rc=0
    wait "$PYTHON_PID" || rc=$?
    PYTHON_PID=""
    return $rc
}
