#!/usr/bin/env bash
#SBATCH --job-name=molgap-pcqm-train-scan
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SHARE_ROOT="${SHARE_ROOT:-/work1/share/$USER}"
DATA_ROOT="$SHARE_ROOT/molgap-data/pcqm4m-v2"
RUN_ROOT="$SHARE_ROOT/molgap-results/phase8/pcqm4mv2_train_residual_scan"
cd "$PROJECT_DIR"
mkdir -p results/scnet/logs "$RUN_ROOT"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/evaluation/scan_pcqm4mv2_train_residuals.py \
  --pcqm-csv "$DATA_ROOT/raw/data.csv.gz" \
  --split "$DATA_ROOT/split_dict.pt" \
  --historical-csv data/raw/phase8_repair_v3_1p5m.csv \
  --out-dir "$RUN_ROOT" \
  --chunk-size 25000 --graph-batch-size 512 --hard-pool-size 200000 \
  --control-gps7 models/scnet/phase8_residual40k_control_gps7.pt \
  --control-gps9 models/scnet/phase8_residual40k_control_gps9.pt \
  --control-head models/scnet/phase8_residual40k_control_dual2d.pt \
  --repair-gps7 models/scnet/phase8_repair_v2_1m_gps7_control.pt \
  --repair-gps9 models/scnet/phase8_repair_v2_1m_gps9_control.pt \
  --repair-head models/scnet/phase8_dual_gps_2d_v2_1m.pt
