#!/usr/bin/env bash
# Submit the 6 Fig 11 rank-8 free-decode fill experiments
# =========================================================
# Fills the 6 cells the registry audit (scratch/audit_temp_sweep.py) identified
# as truly missing from the Fig 11 3x3 seed grid (i.e. neither cached nor
# in-flight at status="running"):
#
#   wolf T=1.3, tseed=1,   gseed=1
#   wolf T=2.0, tseed=123, gseed=123
#   owl  T=0.5, tseed=42,  gseed=1
#   owl  T=2.0, tseed=1,   gseed=123
#   owl  T=2.0, tseed=42,  gseed=123
#   owl  T=2.0, tseed=123, gseed=123
#
# What it does:
#   1. Pre-generates the dataset for the owl_T=2.0 config (which has 3 array
#      tasks sharing one dataset — pre-generating avoids a 3-way race).
#   2. Submits all 4 benchmark-parallel jobs immediately; the owl_T2.0 one
#      depends on its dataset job via sbatch afterany. The other 3 configs
#      are single-task and auto-generate their dataset inside the task.
#
# Total: 1 dataset SLURM task + 6 benchmark SLURM tasks = 7 tasks queued at
# once, all under the QOS cap. Fire-and-forget; tail the SLURM logs in logs/
# or `squeue --me` to monitor.
#
# Usage (from repo root):
#   bash scripts/submit_fig11_fill.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[$(date -Iseconds)] Submitting Fig 11 fill jobs..."

# Pre-generate the only multi-task dataset to avoid an intra-config race.
echo
echo "--- Step 1/2: generate dataset for owl T=2.0 (shared across 3 array tasks) ---"
GEN_OUT=$(./submit.sh generate-datasets \
    --config configs/temp_r8_fig11_fill_owl_T2.0.yaml \
    --array-size 1 --max-gpus 1)
echo "$GEN_OUT"
GEN_JOB=$(echo "$GEN_OUT" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}')
if [[ -z "${GEN_JOB:-}" ]]; then
    echo "ERROR: could not parse dataset job id from submit.sh output" >&2
    exit 1
fi
echo "owl_T2.0 dataset job id: $GEN_JOB"

# Fire the 4 benchmark jobs. The 3 single-task configs auto-generate their
# (race-free) dataset inside the task; the owl_T2.0 job waits for $GEN_JOB.
echo
echo "--- Step 2/2: submit 4 benchmark-parallel jobs (6 total experiments) ---"

./submit.sh benchmark-parallel \
    --config configs/temp_r8_fig11_fill_wolf_T1.3.yaml \
    --array-size 1 --max-gpus 1

./submit.sh benchmark-parallel \
    --config configs/temp_r8_fig11_fill_wolf_T2.0.yaml \
    --array-size 1 --max-gpus 1

./submit.sh benchmark-parallel \
    --config configs/temp_r8_fig11_fill_owl_T0.5.yaml \
    --array-size 1 --max-gpus 1

./submit.sh benchmark-parallel \
    --config configs/temp_r8_fig11_fill_owl_T2.0.yaml \
    --array-size 3 --max-gpus 3 \
    --depends-on "$GEN_JOB"

echo
echo "[$(date -Iseconds)] All Fig 11 fill jobs submitted."
echo "  Monitor:  squeue --me"
echo "  Re-audit: python3 scratch/audit_temp_sweep.py"
