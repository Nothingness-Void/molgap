#!/usr/bin/env bash
# Submit with: sbatch scripts/phase8/remote/scnet/smoke_phase8_dcu.slurm
# Do not request eight DCUs until this single-DCU portability gate passes.
#SBATCH --job-name=molgap-dcu-smoke
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10

# `.scnet/site` contains only the PyG Python package. The DTK-adapted PyTorch
# remains owned by the platform environment, never by a CUDA/PyPI torch wheel.
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/remote/scnet/check_dcu_environment.py \
  --output results/scnet/dcu_environment.json

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

python scripts/phase8/training/train_encoder.py \
  --kind schnet \
  --graphs results/phase8/pyg_3d_graphs_etkdg_expansion_500k.pt \
  --max-samples 2000 \
  --epochs 2 \
  --patience 2 \
  --batch-size 64 \
  --no-embeddings \
  --model-out results/scnet/phase8_dcu_smoke_schnet.pt \
  --metrics-out results/scnet/phase8_dcu_smoke_schnet_metrics.json
