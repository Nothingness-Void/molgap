#!/bin/bash
#SBATCH --job-name=k1-spair-prof
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/k1-sparse-pair/logs/profile-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/k1-sparse-pair/logs/profile-%j.err

set -euo pipefail

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1

ROOT=/public/home/changfeng2006/molgap-v4-bootstrap
JOB_ROOT="${ROOT}/k1-sparse-pair"
PYTHON="${ROOT}/runtime-v4/miniconda3-py310-25.7.0-2/bin/python"
CACHE="${ROOT}/molgap-v4-100k/input/pcqm4mv2-ogb-fixed-100k-v1"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE_SHA256 is required}"
SOURCE_ARCHIVE="${SOURCE_ROOT}/source.tar.gz"
OUTPUT="${JOB_ROOT}/attempts/${SLURM_JOB_ID}/profile"

mkdir -p "${JOB_ROOT}/logs" "${OUTPUT}"
test -f "${CACHE}/manifest.json"
test -f "${SOURCE_ROOT}/src/molgap/k1_sparse_pair.py"
test -f "${SOURCE_ARCHIVE}"
export PYTHONPATH="${SOURCE_ROOT}/src"

"${PYTHON}" "${SOURCE_ROOT}/experiments/pcqm_k1_sparse_pair_100k/run.py" \
  profile \
  --dataset-root "${CACHE}" \
  --manifest "${CACHE}/manifest.json" \
  --source-archive "${SOURCE_ARCHIVE}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  --source-commit "${SOURCE_COMMIT}" \
  --output "${OUTPUT}" \
  --platform-id scnet-kunshan-dcu
