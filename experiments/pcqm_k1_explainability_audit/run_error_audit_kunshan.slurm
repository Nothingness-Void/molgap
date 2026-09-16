#!/bin/bash
#SBATCH --job-name=molgap-k1-xai-stage1
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/k1-xai-stage1/logs/job-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/k1-xai-stage1/logs/job-%j.err

set -euo pipefail

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
JOB_ROOT="${ROOT}/k1-xai-stage1"
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-100k/input/pcqm4mv2-ogb-fixed-100k-v1"
SOURCE_ROOT="${SOURCE_ROOT:-${JOB_ROOT}/source}"
ATTEMPT_ROOT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}"

mkdir -p "${JOB_ROOT}/logs" "${ATTEMPT_ROOT}/output"
test -f "${SOURCE_ROOT}/src/molgap/pcqm_k1_explainability.py"
test -f "${SOURCE_ROOT}/src/molgap/pcqm_wedge.py"
test -f "${SOURCE_ROOT}/experiments/pcqm_k1_explainability_audit/preflight_error_audit.py"
export PYTHONPATH="${SOURCE_ROOT}/src"
"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_explainability_audit/preflight_error_audit.py" \
  --cache-root "${CACHE}" \
  --payload-root "${JOB_ROOT}/payloads" \
  --output "${ATTEMPT_ROOT}/preflight.json"
"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_explainability_audit/run_error_audit.py" \
  --cache-root "${CACHE}" \
  --payload-root "${JOB_ROOT}/payloads" \
  --output-root "${ATTEMPT_ROOT}/output"
