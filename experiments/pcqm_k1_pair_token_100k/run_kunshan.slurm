#!/bin/bash
#SBATCH --job-name=molgap-k1-pair-token
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token/logs/job-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token/logs/job-%j.err

set -euo pipefail

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
JOB_ROOT="${ROOT}/k1-pair-token"
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-100k/input/pcqm4mv2-ogb-fixed-100k-v1"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT must name the immutable source tree}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE_SHA256 is required}"
ATTEMPT_ROOT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}"

mkdir -p "${JOB_ROOT}/logs" "${ATTEMPT_ROOT}/output"
test -f "${SOURCE_ROOT}/src/molgap/k1_pair_token.py"
test -f "${CACHE}/manifest.json"

"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_pair_token_100k/run_candidate.py" \
  --source-root "${SOURCE_ROOT}" \
  --cache-root "${CACHE}" \
  --output-root "${ATTEMPT_ROOT}/output/${MODE:-neural_atom_k1_pair_token}" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}"
