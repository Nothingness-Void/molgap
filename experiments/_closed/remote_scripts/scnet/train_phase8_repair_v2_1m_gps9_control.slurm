#!/usr/bin/env bash
# Controlled repair-v2 GPS9 run: only the appended 500K differs from v1.
#SBATCH --job-name=molgap-repairv2-gps9ctl
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_DIR="results/phase8/repair_v2_scnet/gps9_control_uniform"
mkdir -p results/scnet/logs "$RUN_DIR" models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --kind gps
  --graphs results/phase8/repair_v2_scnet/pyg_2d_graphs_bond_repair_v2_1m.pt
  --init-from models/phase8_gps_expansion_500k_depth9.pt --num-layers 9
  --epochs 40 --patience 12 --lr 0.0001
  --batch-size 256 --eval-batch-size 256 --embedding-batch-size 256
  --checkpoint-out "$RUN_DIR/training_state.pt" --checkpoint-every 1
  --model-out models/scnet/phase8_repair_v2_1m_gps9_control.pt
  --metrics-out "$RUN_DIR/metrics.json" --embeddings-out "$RUN_DIR/embeddings.pt"
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then args+=(--resume-from "$RUN_DIR/training_state.pt"); fi
python scripts/phase8/training/train_encoder.py "${args[@]}"
