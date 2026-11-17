#!/usr/bin/env bash
# Formal 500K SchNetPack 3D encoder run after the bounded 30K gate succeeds.
# Submit with: sbatch --dependency=afterok:<30K_JOB_ID> scripts/phase8/archive/remote/scnet/train_schnetpack_500k.slurm
#SBATCH --job-name=molgap-spk-500k
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=59G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs models/scnet

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/.scnet/schnetpack-v204:$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

python scripts/phase8/archive/remote/scnet/train_schnetpack.py \
  --max-samples 500000 \
  --batch-size 32 \
  --epochs 60 \
  --patience 12 \
  --model-out models/scnet/schnetpack_expansion_500k.pt \
  --metrics-out results/scnet/schnetpack_expansion_500k_metrics.json \
  --split-out results/scnet/schnetpack_expansion_500k_split.pt
