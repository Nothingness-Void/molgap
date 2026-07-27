#!/usr/bin/env bash
#SBATCH --job-name=molgap-kd-ext-eval
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
mkdir -p results/scnet/logs results/phase8/distilled_2m_external_eval

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/eval_distilled_gps_external.py \
  --common-csv results/phase8/remote/kaggle/external_eval_1m/datasets/labels/external_common_labels.csv \
  --pcqm-csv results/kaggle/molgap_pcqm4mv2_valid_5k/pcqm4mv2_valid_5k.csv \
  --student models/scnet/phase8_distilled_teacher_w30_2m_gps7.pt \
  --control-gps7 models/scnet/phase8_residual40k_control_gps7.pt \
  --control-gps9 models/scnet/phase8_residual40k_control_gps9.pt \
  --control-head models/scnet/phase8_residual40k_control_dual2d.pt \
  --repair-gps7 models/scnet/phase8_repair_v2_1m_gps7_control.pt \
  --repair-gps9 models/scnet/phase8_repair_v2_1m_gps9_control.pt \
  --repair-head models/scnet/phase8_dual_gps_2d_v2_1m.pt \
  --out-dir results/phase8/distilled_2m_external_eval \
  --graph-batch-size 256 --bootstrap-draws 10000
