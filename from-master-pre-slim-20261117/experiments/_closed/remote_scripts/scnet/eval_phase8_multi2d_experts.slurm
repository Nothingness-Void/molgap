#!/usr/bin/env bash
#SBATCH --job-name=molgap-multi2d-eval
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

SEALED_CSV="${SEALED_CSV:-results/phase8/broad_residual_pilot/sealed_scaffold_5k.csv}"
OUT_DIR="${OUT_DIR:-results/phase8/multi2d_expert_ensemble}"

python scripts/phase8/evaluation/eval_multi2d_experts.py \
  --common-csv results/phase8/remote/kaggle/external_eval_1m/datasets/labels/external_common_labels.csv \
  --pcqm-csv results/kaggle/evaluation/reference/molgap_pcqm4mv2_valid_5k/pcqm4mv2_valid_5k.csv \
  --sealed-csv "$SEALED_CSV" \
  --expert control_a \
    models/scnet/phase8_residual40k_control_gps7.pt \
    models/scnet/phase8_residual40k_control_gps9.pt \
    models/scnet/phase8_residual40k_control_dual2d.pt \
  --expert control_b \
    models/scnet/phase8_gps_expansion_1m.pt \
    models/scnet/phase8_gps_expansion_1m_depth9.pt \
    models/scnet/phase8_dual_gps_2d_v1_1m.pt \
  --expert repair_v2 \
    models/scnet/phase8_repair_v2_1m_gps7_control.pt \
    models/scnet/phase8_repair_v2_1m_gps9_control.pt \
    models/scnet/phase8_dual_gps_2d_v2_1m.pt \
  --expert broad98k \
    models/scnet/phase8_broad_residual98k_gps7.pt \
    models/scnet/phase8_broad_residual98k_gps9.pt \
    models/scnet/phase8_broad_residual98k_dual2d.pt \
  --ensemble mean_control_repair=control_a,repair_v2 \
  --ensemble mean_controls_repair=control_a,control_b,repair_v2 \
  --ensemble mean_control_broad_repair=control_a,broad98k,repair_v2 \
  --ensemble mean_all4=control_a,control_b,repair_v2,broad98k \
  --baseline-name control_a \
  --out-dir "$OUT_DIR" \
  --bootstrap-draws 10000
