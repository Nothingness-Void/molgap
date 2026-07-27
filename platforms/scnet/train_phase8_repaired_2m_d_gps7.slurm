#!/usr/bin/env bash
#SBATCH --job-name=molgap-rep2m-D-gps7
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
STAGE_ROOT="${STAGE_ROOT:-$TEAM_ROOT/molgap-results/phase8/repaired_2m_d}"
RUN_SEED="${RUN_SEED:-42}"
GRAPH="$STAGE_ROOT/pyg_2d_graphs_repaired_2m.pt"
RUN_ROOT="$STAGE_ROOT/gps7_seed${RUN_SEED}"
MODEL="$STAGE_ROOT/phase8_repaired_2m_d_gps7_seed${RUN_SEED}.pt"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$RUN_ROOT"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --kind gps
  --graphs "$GRAPH"
  --init-from models/scnet/phase8_repair_v3_1p5m_gps7.pt
  --seed "$RUN_SEED" --split-seed 42
  --epochs 40 --patience 12 --lr 0.00008
  --batch-size 256 --eval-batch-size 256 --embedding-batch-size 256
  --replay-boundary 500000 --replay-weight 3.0
  --checkpoint-out "$RUN_ROOT/training_state.pt" --checkpoint-every 1
  --model-out "$MODEL"
  --metrics-out "$RUN_ROOT/metrics.json"
  --embeddings-out "$RUN_ROOT/embeddings.pt"
  --no-embeddings
)
if [[ -f "$RUN_ROOT/training_state.pt" ]]; then
  ARGS+=(--resume-from "$RUN_ROOT/training_state.pt")
fi
python scripts/phase8/training/train_encoder.py "${ARGS[@]}"
