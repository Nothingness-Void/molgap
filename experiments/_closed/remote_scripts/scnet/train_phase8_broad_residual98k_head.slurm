#!/usr/bin/env bash
#SBATCH --job-name=molgap-br98k-head
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/broad_residual98k_scnet"
RUN_DIR="$RUN_ROOT/head_uniform"
mkdir -p "$RUN_DIR" results/scnet/logs models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --gps7-emb "$RUN_ROOT/gps7_uniform/embeddings.pt"
  --gps9-emb "$RUN_ROOT/gps9_uniform/embeddings.pt"
  --csv data/raw/phase8_expansion_1m_broad_residual.csv
  --tag broad_residual98k_uniform --out "$RUN_DIR/metrics.json"
  --model-out models/scnet/phase8_broad_residual98k_dual2d.pt
  --checkpoint-out "$RUN_DIR/training_state.pt"
  --epochs 150 --patience 25 --batch-size 4096 --eval-batch-size 8192
  --lr 0.00054 --weight-decay 0.00001 --hidden 192
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then ARGS+=(--resume-from "$RUN_DIR/training_state.pt"); fi
python scripts/phase8/training/train_dual_gps_2d_head.py "${ARGS[@]}"
