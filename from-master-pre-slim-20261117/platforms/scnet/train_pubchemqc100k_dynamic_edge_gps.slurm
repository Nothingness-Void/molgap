#!/usr/bin/env bash
# One fixed-contract pure-2D architecture candidate; submit only once.
#SBATCH --job-name=molgap-pc100k-edgegps
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
GRAPH="$RUN_ROOT/pyg_2d_graphs_pubchemqc100k_architecture.pt"
SPLIT="$RUN_ROOT/split.csv"
RUN_DIR="$RUN_ROOT/dynamic_edge_gps_seed42"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$RUN_DIR"
test -s "$GRAPH"
test -s "$SPLIT"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --graphs "$GRAPH"
  --split-csv "$SPLIT"
  --output-dir "$RUN_DIR"
  --epochs 40 --patience 10 --lr 0.0004 --weight-decay 0.00001
  --batch-size 256 --seed 42 --split-seed 42
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then
  ARGS+=(--resume-from "$RUN_DIR/training_state.pt")
fi
python "$PROJECT_DIR/experiments/pubchemqc100k_architecture/train_dynamic_edge_gps.py" "${ARGS[@]}"

