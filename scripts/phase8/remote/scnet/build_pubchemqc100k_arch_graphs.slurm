#!/usr/bin/env bash
#SBATCH --job-name=molgap-pc100k-subset
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
RUN_ROOT="${RUN_ROOT:-$TEAM_ROOT/molgap-results/phase8/pubchemqc100k_architecture}"
SOURCE_GRAPH="$PROJECT_DIR/results/phase8/pyg_2d_graphs_bond_expansion_1m.pt"
SPLIT="$RUN_ROOT/split.csv"
OUTPUT="$RUN_ROOT/pyg_2d_graphs_pubchemqc100k_architecture.pt"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$RUN_ROOT"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/data/subset_graph_cache_by_split.py \
  --graphs "$SOURCE_GRAPH" \
  --split-csv "$SPLIT" \
  --output "$OUTPUT" \
  --report "$RUN_ROOT/graph_subset_report.json"
