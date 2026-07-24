#!/usr/bin/env bash
# Coverage expert GPS9, continued from the aligned 1.5M model.
#SBATCH --job-name=molgap-multi2d-2m-gps9
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/multi2d_2m_scnet"
RUN_DIR="$RUN_ROOT/gps9"
mkdir -p results/scnet/logs "$RUN_DIR" models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --kind gps
  --graphs "$RUN_ROOT/pyg_2d_graphs_multi2d_coverage_2m.pt"
  --init-from models/scnet/phase8_repair_v3_1p5m_gps9.pt --num-layers 9
  --epochs 40 --patience 12 --lr 0.00008
  --batch-size 256 --eval-batch-size 256 --embedding-batch-size 256
  --checkpoint-out "$RUN_DIR/training_state.pt" --checkpoint-every 1
  --model-out models/scnet/phase8_multi2d_coverage_2m_gps9.pt
  --metrics-out "$RUN_DIR/metrics.json" --embeddings-out "$RUN_DIR/embeddings.pt"
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then ARGS+=(--resume-from "$RUN_DIR/training_state.pt"); fi
python scripts/phase8/training/train_encoder.py "${ARGS[@]}"
