#!/usr/bin/env bash
# Warm-start and rescue-train the dual-GPS head on aligned 2.02M embeddings.
#SBATCH --job-name=molgap-2m-hard20k-head
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
RUN_ROOT="results/phase8/multi2d_2m_hard20k_scnet"
RUN_DIR="$RUN_ROOT/head"
mkdir -p "$RUN_DIR" results/scnet/logs models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--gps7-emb "$RUN_ROOT/gps7/embeddings.pt" --gps9-emb "$RUN_ROOT/gps9/embeddings.pt"
  --csv data/raw/phase8_multi2d_coverage_2m_hard20k.csv --tag multi2d_2m_hard20k
  --out "$RUN_DIR/metrics.json" --model-out models/scnet/phase8_multi2d_2m_hard20k_dual2d.pt
  --checkpoint-out "$RUN_DIR/training_state.pt" --epochs 30 --patience 8
  --batch-size 4096 --eval-batch-size 8192 --lr 0.00005 --weight-decay 0.00001 --hidden 192
  --replay-boundary 2000000 --replay-weight 0.09)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then
  ARGS+=(--resume-from "$RUN_DIR/training_state.pt")
else
  ARGS+=(--init-from models/scnet/phase8_multi2d_coverage_2m_dual2d.pt)
fi
python scripts/phase8/training/train_dual_gps_2d_head.py "${ARGS[@]}"
