#!/usr/bin/env bash
# One-DCU GPS-only smoke gate for BW-1 while the platform lacks torch-cluster.
# Submit with: sbatch scripts/phase8/remote/scnet/smoke_phase8_gps_dcu.slurm
#SBATCH --job-name=molgap-dcu-gps-smoke
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=04:00:00
# BW-1 account policy exposes one DCU with 8 CPU cores and about 28 GiB RAM.
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/remote/scnet/check_dcu_environment.py \
  --skip-schnet \
  --output results/scnet/dcu_environment_gps.json

python scripts/phase8/training/train_encoder.py \
  --kind gps \
  --graphs results/phase8/pyg_2d_graphs_bond_expansion_500k.pt \
  --max-samples 2000 \
  --epochs 2 \
  --patience 2 \
  --batch-size 64 \
  --no-embeddings \
  --model-out results/scnet/phase8_dcu_smoke_gps.pt \
  --metrics-out results/scnet/phase8_dcu_smoke_gps_metrics.json
