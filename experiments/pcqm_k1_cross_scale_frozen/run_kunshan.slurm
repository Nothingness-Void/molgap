#!/bin/bash
#SBATCH --job-name=molgap-k1-cross-scale
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/k1-cross-scale-frozen/logs/job-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/k1-cross-scale-frozen/logs/job-%j.err

set -euo pipefail
module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
JOB_ROOT="${ROOT}/k1-cross-scale-frozen"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE_SHA256 is required}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
INPUT="${JOB_ROOT}/inputs"
OUTPUT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}/output"
mkdir -p "${OUTPUT}"
export PYTHONPATH="${SOURCE_ROOT}/src"

RUN_ARGS=(--source-commit "${SOURCE_COMMIT}" --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}")
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then RUN_ARGS+=(--preflight-only); fi
"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_cross_scale_frozen/run.py" \
  --cache-100k "${ROOT}/molgap-v4-100k/input/pcqm4mv2-ogb-fixed-100k-v1" \
  --cache-500k "${ROOT}/molgap-v4-500k/input/pcqm4mv2-ogb-fixed-500k-scnet-v1" \
  --k1-model "${INPUT}/k1_best_model.pt" \
  --k1-payload "${INPUT}/k1_best_development_payload.pt" \
  --pair-model "${INPUT}/pair_best_model.pt" \
  --pair-payload "${INPUT}/pair_best_development_payload.pt" \
  --target-transform "${INPUT}/target_transform.json" \
  --output-root "${OUTPUT}" \
  "${RUN_ARGS[@]}"
