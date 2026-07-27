#!/usr/bin/env bash
# Full 1M 2D GPS7 continuation from the selected 500K GPS7 checkpoint.
# Submit with: sbatch scripts/phase8/archive/remote/scnet/train_phase8_expansion_1m_gps7.slurm
#SBATCH --job-name=molgap-gps7-1m
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs models/scnet

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/training/train_encoder.py \
  --kind gps \
  --graphs results/phase8/pyg_2d_graphs_bond_expansion_1m.pt \
  --init-from models/phase8_gps_expansion_500k.pt \
  --epochs 40 \
  --patience 12 \
  --lr 0.0001 \
  --batch-size 256 \
  --eval-batch-size 256 \
  --embedding-batch-size 256 \
  --model-out models/scnet/phase8_gps_expansion_1m.pt \
  --metrics-out results/phase8/gps_expansion_1m_metrics.json \
  --embeddings-out results/phase8/gps_expansion_1m_embeddings.pt
