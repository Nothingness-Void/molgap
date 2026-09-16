#!/bin/bash
#SBATCH --job-name=molgap-k1-pair-token-500k
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token-500k/logs/job-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token-500k/logs/job-%j.err

set -euo pipefail

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export MOLGAP_V4_LOADER_WORKERS=2

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
JOB_ROOT="${ROOT}/k1-pair-token-500k"
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-500k/input/pcqm4mv2-ogb-fixed-500k-scnet-v1"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT must name the immutable source tree}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE_SHA256 is required}"
ATTEMPT_ROOT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}"
EXTRA_ARGS=()
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--preflight-only)
fi

mkdir -p "${JOB_ROOT}/logs" "${ATTEMPT_ROOT}/output"
test -f "${SOURCE_ROOT}/src/molgap/pcqm_k1_pair_token_500k.py"
test -f "${CACHE}/manifest.json"

"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_pair_token_500k/run_candidate.py" \
  --source-root "${SOURCE_ROOT}" \
  --cache-root "${CACHE}" \
  --output-root "${ATTEMPT_ROOT}/output/neural_atom_k1_pair_token" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  "${EXTRA_ARGS[@]}"
