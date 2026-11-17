#!/usr/bin/env bash
# Low-LR GPS7 rescue continuation with a 10% expected hard20K draw rate.
#SBATCH --job-name=molgap-2m-hard20k-gps7
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/multi2d_2m_hard20k_scnet"
RUN_DIR="$RUN_ROOT/gps7"
mkdir -p "$RUN_DIR" results/scnet/logs models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--kind gps --graphs "$RUN_ROOT/pyg_2d_graphs_multi2d_2m_hard20k.pt"
  --init-from models/scnet/phase8_multi2d_coverage_2m_gps7.pt
  --epochs 8 --patience 8 --lr 0.00002 --replay-boundary 2000000 --replay-weight 0.09
  --batch-size 256 --eval-batch-size 256 --embedding-batch-size 256
  --checkpoint-out "$RUN_DIR/training_state.pt" --checkpoint-every 1
  --model-out models/scnet/phase8_multi2d_2m_hard20k_gps7.pt
  --metrics-out "$RUN_DIR/metrics.json" --embeddings-out "$RUN_DIR/embeddings.pt")
if [[ -f "$RUN_DIR/training_state.pt" ]]; then ARGS+=(--resume-from "$RUN_DIR/training_state.pt"); fi
python scripts/phase8/training/train_encoder.py "${ARGS[@]}"
