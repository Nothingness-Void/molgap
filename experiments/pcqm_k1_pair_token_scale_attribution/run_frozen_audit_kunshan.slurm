#!/bin/bash
#SBATCH --job-name=molgap-pair-scale-audit
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token-scale-audit/logs/job-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token-scale-audit/logs/job-%j.err

set -euo pipefail
module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-500k/input/pcqm4mv2-ogb-fixed-500k-scnet-v1"
MODEL_ROOT="${ROOT}/k1-pair-token-500k/attempts/122312462/output/neural_atom_k1_pair_token"
JOB_ROOT="${ROOT}/k1-pair-token-scale-audit"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE_SHA256 is required}"
ATTEMPT_ROOT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}"

mkdir -p "${JOB_ROOT}/logs" "${ATTEMPT_ROOT}/output"
test -f "${SOURCE_ROOT}/src/molgap/pcqm_k1_pair_token_scale_audit.py"
test -f "${CACHE}/manifest.json"
test -f "${MODEL_ROOT}/best_model.pt"
test -f "${MODEL_ROOT}/best_predictions.pt"
export PYTHONPATH="${SOURCE_ROOT}/src"
export MOLGAP_FIXED_500K_CACHE_ROOT="${CACHE}"

"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_pair_token_scale_attribution/run_frozen_audit.py" \
  --cache-root "${CACHE}" \
  --checkpoint "${MODEL_ROOT}/best_model.pt" \
  --reference-payload "${MODEL_ROOT}/best_predictions.pt" \
  --output-root "${ATTEMPT_ROOT}/output" \
  --source-commit "${SOURCE_COMMIT}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}"
