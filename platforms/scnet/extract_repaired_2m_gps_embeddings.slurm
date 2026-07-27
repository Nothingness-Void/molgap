#!/usr/bin/env bash
#SBATCH --job-name=molgap-rep2m-gps-emb
#SBATCH --partition=hx1hdnormal
#SBATCH --array=0-1
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%A_%a.out
#SBATCH --error=results/scnet/logs/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
STAGE_ROOT="${STAGE_ROOT:-$TEAM_ROOT/molgap-results/phase8/repaired_2m_d}"
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  NAME=gps7
  LAYERS=7
  HIDDEN=192
else
  NAME=gps9
  LAYERS=9
  HIDDEN=192
fi
MODEL="$STAGE_ROOT/phase8_repaired_2m_d_${NAME}_seed42.pt"
OUT_DIR="$STAGE_ROOT/${NAME}_seed42/embedding_parts"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$OUT_DIR"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/extract_gps_embeddings_chunked.py \
  --graphs "$STAGE_ROOT/pyg_2d_graphs_repaired_2m.pt" \
  --model "$MODEL" --name "$NAME" --layers "$LAYERS" \
  --hidden-channels "$HIDDEN" --out-dir "$OUT_DIR" \
  --batch-size 256 --part-rows 50000 --expected-rows 2000000 \
  --graph-sha256 2c7011880457e5583899217f05fafe81c27aeba13dde0a8bd540346f4705e9f8
