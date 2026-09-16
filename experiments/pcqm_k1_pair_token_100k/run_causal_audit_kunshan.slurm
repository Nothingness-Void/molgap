#!/bin/bash
#SBATCH --job-name=molgap-k1-pair-audit
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token-audit/logs/job-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/k1-pair-token-audit/logs/job-%j.err

set -euo pipefail
module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-100k/input/pcqm4mv2-ogb-fixed-100k-v1"
MODEL_ROOT="${ROOT}/k1-pair-token/attempts/122275244/output/neural_atom_k1_pair_token"
JOB_ROOT="${ROOT}/k1-pair-token-audit"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
ATTEMPT_ROOT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}"

mkdir -p "${JOB_ROOT}/logs" "${ATTEMPT_ROOT}/output"
export PYTHONPATH="${SOURCE_ROOT}/src"
"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_pair_token_100k/run_causal_audit.py" \
  --cache-root "${CACHE}" \
  --checkpoint "${MODEL_ROOT}/best_model.pt" \
  --reference-payload "${MODEL_ROOT}/best_development_payload.pt" \
  --output-root "${ATTEMPT_ROOT}/output" \
  --batch-size 128

