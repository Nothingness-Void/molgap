#!/bin/bash
#SBATCH --job-name=k1-representation
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail
module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${SOURCE_ROOT:?}/src"
ROOT="${MOLGAP_REMOTE_ROOT:?}"
exec "${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python" -u -m molgap.k1_representation_audit \
  --cache100 "${ROOT}/molgap-v4-100k/input/pcqm4mv2-ogb-fixed-100k-v1" \
  --cache500 "${ROOT}/molgap-v4-500k/input/pcqm4mv2-ogb-fixed-500k-scnet-v1" \
  --inputs100 "${ROOT}/k1-cross-scale-frozen/inputs" \
  --inputs500 "${ROOT}/k1-matched60-reference/attempts/122743291/output/neural_atom_k1" \
  --output "${ROOT}/k1-representation-audit/attempts/${SLURM_JOB_ID}/output" \
  --source-commit "${SOURCE_COMMIT:?}" --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256:?}"
