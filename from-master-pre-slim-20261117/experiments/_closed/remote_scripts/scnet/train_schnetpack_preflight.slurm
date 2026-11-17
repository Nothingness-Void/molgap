#!/usr/bin/env bash
# One-epoch end-to-end preflight before reserving the 30K trainability gate.
# Submit with: sbatch scripts/phase8/archive/remote/scnet/train_schnetpack_preflight.slurm
#SBATCH --job-name=molgap-spk-preflight
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
mkdir -p results/scnet/logs models/scnet

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/.scnet/schnetpack-v204:$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

python scripts/phase8/archive/remote/scnet/train_schnetpack.py \
  --max-samples 128 \
  --batch-size 32 \
  --epochs 1 \
  --patience 1 \
  --model-out models/scnet/schnetpack_preflight.pt \
  --metrics-out results/scnet/schnetpack_preflight_metrics.json \
  --split-out results/scnet/schnetpack_preflight_split.pt
