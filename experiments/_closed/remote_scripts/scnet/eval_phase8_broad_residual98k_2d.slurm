#!/usr/bin/env bash
#SBATCH --job-name=molgap-br98k-eval
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/eval_dual_gps_2d_ab.py \
  --common-csv results/phase8/remote/kaggle/external_eval_1m/datasets/labels/external_common_labels.csv \
  --pcqm-csv results/kaggle/evaluation/reference/molgap_pcqm4mv2_valid_5k/pcqm4mv2_valid_5k.csv \
  --sealed-csv results/phase8/broad_residual_pilot/sealed_scaffold_5k.csv \
  --baseline-gps7 models/scnet/phase8_residual40k_control_gps7.pt \
  --baseline-gps9 models/scnet/phase8_residual40k_control_gps9.pt \
  --baseline-head models/scnet/phase8_residual40k_control_dual2d.pt \
  --candidate-gps7 models/scnet/phase8_broad_residual98k_gps7.pt \
  --candidate-gps9 models/scnet/phase8_broad_residual98k_gps9.pt \
  --candidate-head models/scnet/phase8_broad_residual98k_dual2d.pt \
  --baseline-name original1m_equal_cost_control \
  --candidate-name broad_residual98k_uniform \
  --out-dir results/phase8/broad_residual98k_external_eval \
  --bootstrap-draws 10000
