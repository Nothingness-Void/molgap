#!/bin/bash
#SBATCH --job-name=molgap-k1-xai-stage1
#SBATCH --partition=kshctest02
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
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

mkdir -p "${JOB_ROOT}/logs" "${JOB_ROOT}/output"
export PYTHONPATH="${JOB_ROOT}/source/src"
"${PYTHON}" "${JOB_ROOT}/source/experiments/pcqm_k1_explainability_audit/run_error_audit.py" \
  --cache-root "${CACHE}" \
  --payload-root "${JOB_ROOT}/payloads" \
  --output-root "${JOB_ROOT}/output"
