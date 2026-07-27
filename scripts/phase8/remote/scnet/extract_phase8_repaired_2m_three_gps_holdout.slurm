#!/usr/bin/env bash
# Extract leakage-bounded validation/test predictions for the learned 3-GPS gate.
#SBATCH --job-name=molgap-rep2m-3gps-holdout
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
STAGE_ROOT="${STAGE_ROOT:-$TEAM_ROOT/molgap-results/phase8/repaired_2m_d}"
OUT_DIR="$STAGE_ROOT/three_gps_router_seed42"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$OUT_DIR"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/extract_three_gps_holdout.py \
  --graphs "$STAGE_ROOT/pyg_2d_graphs_repaired_2m.pt" \
  --gps7 "$STAGE_ROOT/phase8_repaired_2m_d_gps7_seed42.pt" \
  --gps9 "$STAGE_ROOT/phase8_repaired_2m_d_gps9_seed42.pt" \
  --gps11-160 "$STAGE_ROOT/phase8_repaired_2m_d_gps11_160_seed42.pt" \
  --out "$OUT_DIR/holdout_predictions.pt" \
  --manifest-out "$OUT_DIR/completion_manifest.json" \
  --split-seed 42 --batch-size 256
