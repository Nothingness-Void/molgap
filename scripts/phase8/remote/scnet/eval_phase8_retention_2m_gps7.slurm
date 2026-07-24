#!/usr/bin/env bash
# Submit after the matching retention training job.
#SBATCH --job-name=molgap-ret2m-eval
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RETENTION_VARIANT="${RETENTION_VARIANT:?set RETENTION_VARIANT=replay or replay_distill}"
cd "$PROJECT_DIR"
OUT_DIR="results/phase8/retention_2m_external_eval/$RETENTION_VARIANT"
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
  --candidate "models/scnet/phase8_retention_2m_${RETENTION_VARIANT}_gps7.pt" \
  --candidate-name "retention_2m_${RETENTION_VARIANT}_gps7" \
  --out-dir "$OUT_DIR" \
  --batch-size 256
