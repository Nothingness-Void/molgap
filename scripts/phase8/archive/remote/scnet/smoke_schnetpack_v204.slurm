#!/usr/bin/env bash
# Isolated SchNetPack 2.0.4 DCU gate; does not use torch-cluster or PyG radius_graph.
# Submit with: sbatch scripts/phase8/archive/remote/scnet/smoke_schnetpack_v204.slurm
#SBATCH --job-name=molgap-schnetpack-smoke
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/.scnet/schnetpack-v204:$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

python scripts/phase8/archive/remote/scnet/smoke_schnetpack_v204.py \
  --max-samples 2000 \
  --batch-size 32 \
  --output results/scnet/schnetpack_v204_smoke.json
