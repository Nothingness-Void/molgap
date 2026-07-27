#!/usr/bin/env bash
#SBATCH --job-name=molgap-rep2m-3gps-res
#SBATCH --partition=hx1hdnormal
#SBATCH --array=0-2
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%A_%a.out
#SBATCH --error=results/scnet/logs/%x-%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
STAGE_ROOT="${STAGE_ROOT:-$TEAM_ROOT/molgap-results/phase8/repaired_2m_d}"
SEED=$((42 + SLURM_ARRAY_TASK_ID))
OUT_DIR="$STAGE_ROOT/three_gps_embedding_residual/seed${SEED}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$OUT_DIR"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/experiments/train_three_gps_embedding_residual.py \
  --gps7-dir "$STAGE_ROOT/gps7_seed42/embedding_parts" \
  --gps9-dir "$STAGE_ROOT/gps9_seed42/embedding_parts" \
  --gps11-embeddings "$STAGE_ROOT/gps11_160_seed42/embeddings.pt" \
  --gps11-model "$STAGE_ROOT/phase8_repaired_2m_d_gps11_160_seed42.pt" \
  --out-dir "$OUT_DIR" --rows 2000000 --seed "$SEED"
