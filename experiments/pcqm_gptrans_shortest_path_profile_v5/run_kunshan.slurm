#!/bin/bash
#SBATCH --job-name=molgap-gptrans-sp-profile
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/gptrans-shortest-path-profile-v5/logs/job-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/gptrans-shortest-path-profile-v5/logs/job-%j.err

set -euo pipefail

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export MOLGAP_PROFILE_LOADER_WORKERS=2

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
JOB_ROOT="${ROOT}/gptrans-shortest-path-profile-v5"
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-500k/input/pcqm4mv2-ogb-fixed-500k-scnet-v1"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT must name the immutable source tree}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE_SHA256 is required}"
ATTEMPT_ROOT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}"

mkdir -p "${JOB_ROOT}/logs" "${ATTEMPT_ROOT}/output"
test -f "${SOURCE_ROOT}/src/molgap/pcqm_gptrans_shortest_path_profile.py"
test -f "${CACHE}/manifest.json"

JOB_RECORD="$(scontrol show job -o "${SLURM_JOB_ID}")"
SUBMIT_TIME="$(printf '%s\n' "${JOB_RECORD}" | sed -n 's/.*SubmitTime=\([^ ]*\).*/\1/p')"
START_TIME="$(printf '%s\n' "${JOB_RECORD}" | sed -n 's/.*StartTime=\([^ ]*\).*/\1/p')"
ALLOCATION_SECONDS="$(${PYTHON} - "${SUBMIT_TIME}" "${START_TIME}" <<'PY'
from datetime import datetime
import sys
submit = datetime.fromisoformat(sys.argv[1])
start = datetime.fromisoformat(sys.argv[2])
print(max(0.0, (start - submit).total_seconds()))
PY
)"

"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_gptrans_shortest_path_profile_v5/run_profile.py" \
  --source-root "${SOURCE_ROOT}" \
  --cache-root "${CACHE}" \
  --output-root "${ATTEMPT_ROOT}/output" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  --allocation-seconds "${ALLOCATION_SECONDS}" \
  --slurm-job-id "${SLURM_JOB_ID}"
