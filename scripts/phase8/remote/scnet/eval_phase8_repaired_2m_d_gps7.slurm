#!/usr/bin/env bash
#SBATCH --job-name=molgap-rep2m-D-eval
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TEAM_ROOT="${TEAM_ROOT:-/work1/share/$USER}"
STAGE_ROOT="${STAGE_ROOT:-$TEAM_ROOT/molgap-results/phase8/repaired_2m_d}"
RUN_SEED="${RUN_SEED:-42}"
MODEL="$STAGE_ROOT/phase8_repaired_2m_d_gps7_seed${RUN_SEED}.pt"
OUT_DIR="$STAGE_ROOT/external_eval_seed${RUN_SEED}"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$OUT_DIR"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/eval_single_gps_vs_routed_v4.py \
  --common-baseline-csv results/phase8/expansion_1m/common_eval_kaggle_predictions.csv \
  --pcqm-baseline-csv results/phase8/expansion_1m/pcqm4mv2_valid_5k_component_predictions.csv \
  --candidate "$MODEL" \
  --candidate-name "repaired_2m_d_gps7_seed${RUN_SEED}" \
  --out-dir "$OUT_DIR" --batch-size 256
