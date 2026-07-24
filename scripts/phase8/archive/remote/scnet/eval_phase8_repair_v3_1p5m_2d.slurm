#!/usr/bin/env bash
#SBATCH --job-name=molgap-repairv3-eval
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
mkdir -p results/scnet/logs results/phase8/repair_v3_1p5m_external_eval
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/eval_dual_gps_2d_ab.py \
  --common-csv results/phase8/gps_arch_dualgps_common_eval_predictions.csv \
  --pcqm-csv results/phase8/gps_arch_dualgps_pcqm_proxy_predictions.csv \
  --baseline-gps7 models/scnet/phase8_gps_expansion_1m.pt \
  --baseline-gps9 models/scnet/phase8_gps_expansion_1m_depth9.pt \
  --baseline-head models/scnet/phase8_dual_gps_2d_v1_1m.pt \
  --candidate-gps7 models/scnet/phase8_repair_v3_1p5m_gps7.pt \
  --candidate-gps9 models/scnet/phase8_repair_v3_1p5m_gps9.pt \
  --candidate-head models/scnet/phase8_dual_gps_2d_v3_1p5m_1p5m.pt \
  --baseline-name original_1m --candidate-name additive_1p5m \
  --out-dir results/phase8/repair_v3_1p5m_external_eval
