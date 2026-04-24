#!/bin/bash
# Shared env setup sourced by every wrapper. Not an #SBATCH script.
#
# Contract:
# - $REPO_ROOT must be exported to the job (submit.sh does this via
#   sbatch --export=ALL,REPO_ROOT=...). If it isn't, we fail loudly — silent
#   fallbacks would cd into a SLURM-shipped tmp script dir and then fail
#   mysteriously a few lines later trying to activate .venv.
# - $SLURM_JOB_ID is present (SLURM sets it for every job/step).
# - local-scratch GRES has been requested; SLURM creates
#   /local/scratch/${USER}_${SLURM_JOB_ID} on the compute node. We redirect
#   UV/HF/Transformers caches and TMPDIR to it so nothing writes to the
#   non-writable /net/scratch2/muchane defaults baked into PATH / user env.

: "${REPO_ROOT:?REPO_ROOT must be set — submit via ./submit.sh (or pass --export=ALL,REPO_ROOT=<abs path>)}"
cd "$REPO_ROOT"

mkdir -p logs

SCRATCH_DIR="/local/scratch/${USER}_${SLURM_JOB_ID:-nojob}"
if [[ -d "$SCRATCH_DIR" ]]; then
    mkdir -p \
        "$SCRATCH_DIR/uv_cache" \
        "$SCRATCH_DIR/transformers_cache" \
        "$SCRATCH_DIR/hf_home" \
        "$SCRATCH_DIR/tmp" \
        "$SCRATCH_DIR/torchinductor" \
        "$SCRATCH_DIR/triton"
    export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
    export TRANSFORMERS_CACHE="$SCRATCH_DIR/transformers_cache"
    export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_home"
    export HF_HOME="$SCRATCH_DIR/hf_home"
    export TMPDIR="$SCRATCH_DIR/tmp"

    # Pin torch + triton caches to this job's scratch, don't read or write any
    # global inductor/triton cache. Llama-array tasks hit
    # `PermissionError: Permission denied: '/local/scratch/muchane_820137'`
    # during vLLM EngineCore startup — a stale absolute path from some earlier
    # job's scratch was being re-emitted at line 841 of the generated
    # torch._inductor kernel file (`k6/ck6l4y...py`). Pinning
    # TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR + the three piecemeal disables
    # below did NOT stop it (820137 kept reappearing). The master switch
    # TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 is documented in torch/_inductor/
    # config.py (see also TORCH_COMPILE_FORCE_DISABLE_CACHES) and disables
    # every inductor cache path; that's the sledgehammer. Costs a few minutes
    # of kernel recompile per vLLM startup but removes the cross-job bug.
    export TORCHINDUCTOR_CACHE_DIR="$SCRATCH_DIR/torchinductor"
    export TRITON_CACHE_DIR="$SCRATCH_DIR/triton"
    export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
    export TORCH_COMPILE_FORCE_DISABLE_CACHES=1
    export TORCHINDUCTOR_FX_GRAPH_CACHE=0
    export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=0
    export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
else
    echo "WARN: $SCRATCH_DIR not present on $(hostname). Did you forget --gres=local:disk:NG?" >&2
fi

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
