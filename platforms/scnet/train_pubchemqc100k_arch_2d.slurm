#!/usr/bin/env bash
# Submit with CANDIDATE=gps7, gps9, or gps11_160.
#SBATCH --job-name=molgap-pc100k-2d
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
RUN_ROOT="${RUN_ROOT:-$TEAM_ROOT/molgap-results/phase8/pubchemqc100k_architecture}"
CANDIDATE="${CANDIDATE:?Set CANDIDATE to gps7, gps9, or gps11_160}"
GRAPH="$RUN_ROOT/pyg_2d_graphs_pubchemqc100k_architecture.pt"
SPLIT="$RUN_ROOT/split.csv"
RUN_DIR="$RUN_ROOT/$CANDIDATE/seed42"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$RUN_DIR"

case "$CANDIDATE" in
  gps7) LAYERS=7; HIDDEN=192; BATCH=256 ;;
  gps9) LAYERS=9; HIDDEN=192; BATCH=256 ;;
  gps11_160) LAYERS=11; HIDDEN=160; BATCH=256 ;;
  *) echo "Unknown CANDIDATE=$CANDIDATE" >&2; exit 2 ;;
esac

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --kind gps
  --graphs "$GRAPH"
  --split-csv "$SPLIT"
  --hidden-channels "$HIDDEN" --num-layers "$LAYERS" --num-heads 4 --dropout 0.05
  --seed 42 --split-seed 42
  --epochs 40 --patience 10 --lr 0.0004 --weight-decay 0.00001
  --batch-size "$BATCH" --eval-batch-size "$BATCH" --embedding-batch-size "$BATCH"
  --checkpoint-out "$RUN_DIR/training_state.pt" --checkpoint-every 1
  --model-out "$RUN_DIR/model.pt"
  --metrics-out "$RUN_DIR/metrics.json"
  --embeddings-out "$RUN_DIR/embeddings.pt"
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then
  ARGS+=(--resume-from "$RUN_DIR/training_state.pt")
fi
python scripts/phase8/training/train_encoder.py "${ARGS[@]}"
