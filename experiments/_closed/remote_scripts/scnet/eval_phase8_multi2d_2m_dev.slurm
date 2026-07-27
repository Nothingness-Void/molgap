#!/usr/bin/env bash
# Development-only comparison; the future 20K sealed set is deliberately absent.
#SBATCH --job-name=molgap-multi2d-2m-dev
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

python scripts/phase8/evaluation/eval_multi2d_experts.py \
  --common-csv results/phase8/remote/kaggle/external_eval_1m/datasets/labels/external_common_labels.csv \
  --pcqm-csv results/kaggle/evaluation/reference/molgap_pcqm4mv2_valid_5k/pcqm4mv2_valid_5k.csv \
  --sealed-csv results/phase8/multi2d_final_sealed/sealed_10k.csv \
  --expert anchor \
    models/scnet/phase8_residual40k_control_gps7.pt \
    models/scnet/phase8_residual40k_control_gps9.pt \
    models/scnet/phase8_residual40k_control_dual2d.pt \
  --expert repair \
    models/scnet/phase8_repair_v2_1m_gps7_control.pt \
    models/scnet/phase8_repair_v2_1m_gps9_control.pt \
    models/scnet/phase8_dual_gps_2d_v2_1m.pt \
  --expert coverage2m \
    models/scnet/phase8_multi2d_coverage_2m_gps7.pt \
    models/scnet/phase8_multi2d_coverage_2m_gps9.pt \
    models/scnet/phase8_multi2d_coverage_2m_dual2d.pt \
  --ensemble incumbent=anchor,repair \
  --ensemble anchor_coverage=anchor,coverage2m \
  --ensemble tri_expert=anchor,repair,coverage2m \
  --baseline-name incumbent \
  --out-dir results/phase8/multi2d_2m_dev_eval \
  --bootstrap-draws 10000
