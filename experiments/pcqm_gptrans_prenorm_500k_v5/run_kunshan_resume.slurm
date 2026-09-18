#!/bin/bash
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00

set -euo pipefail

: "${MODE:?MODE=reference|pair_prenorm is required}"
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
: "${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE_SHA256 is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RESUME_ROOT:?RESUME_ROOT must name an accepted atomic checkpoint directory}"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export MOLGAP_V5_LOADER_WORKERS=2

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-500k/input/pcqm4mv2-ogb-fixed-500k-scnet-v1"
OUTPUT="${CAMPAIGN_ROOT}/attempts/${SLURM_JOB_ID}/${MODE}/train"

mkdir -p "${CAMPAIGN_ROOT}/logs" "${OUTPUT}"
test -f "${SOURCE_ROOT}/src/molgap/pcqm_gptrans_prenorm_500k.py"
test -f "${CACHE}/manifest.json"
test -f "${RESUME_ROOT}/last_checkpoint.pt"
test -f "${RESUME_ROOT}/best_model.pt"
test -f "${RESUME_ROOT}/best_predictions.pt"

"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_gptrans_prenorm_500k_v5/run_arm.py" \
  --source-root "${SOURCE_ROOT}" \
  --cache-root "${CACHE}" \
  --output-root "${OUTPUT}" \
  --mode "${MODE}" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  --resume "${RESUME_ROOT}"
