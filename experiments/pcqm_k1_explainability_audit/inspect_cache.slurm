#!/bin/bash
#SBATCH --job-name=molgap-inspect-cache
#SBATCH --partition=kshctest02
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/public/home/changfeng2006/molgap-v4-bootstrap/inspect/inspect-%j.out
#SBATCH --error=/public/home/changfeng2006/molgap-v4-bootstrap/inspect/inspect-%j.err

set -euo pipefail

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1

PYTHON=/public/home/changfeng2006/molgap-v4-bootstrap/runtime-v4/miniconda3-py310-25.7.0-2/bin/python
ROOT=/public/home/changfeng2006/molgap-v4-bootstrap/molgap-v4-100k

"${PYTHON}" /public/home/changfeng2006/molgap-v4-bootstrap/inspect_cache.py \
  "${ROOT}/input/pcqm4mv2-ogb-fixed-100k-v1/train/train_shard_0002.pt"
