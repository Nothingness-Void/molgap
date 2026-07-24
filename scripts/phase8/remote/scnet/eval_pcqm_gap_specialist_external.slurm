#!/usr/bin/env bash
#SBATCH --job-name=molgap-pcqm-gap-eval
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/pcqm_gap_head_pilot"
OUT="$RUN_ROOT/b_replay_external_eval"
mkdir -p results/scnet/logs "$OUT"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/eval_single_gps_vs_routed_v4.py \
  --common-baseline-csv results/phase8/expansion_1m/common_eval_kaggle_predictions.csv \
  --pcqm-baseline-csv results/phase8/expansion_1m/pcqm4mv2_valid_5k_component_predictions.csv \
  --candidate models/scnet/phase8_pcqm_gap_head_b_replay_w30_gps7.pt \
  --candidate-name pcqm_gap_head_b_replay_w30 \
  --out-dir "$OUT" --batch-size 256
