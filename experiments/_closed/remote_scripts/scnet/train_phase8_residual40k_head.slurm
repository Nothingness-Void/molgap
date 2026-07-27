#!/usr/bin/env bash
#SBATCH --job-name=molgap-res40k-head
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
VARIANT="${PILOT_VARIANT:?set PILOT_VARIANT=control or target}"
RUN_ROOT="results/phase8/residual_target_40k_scnet"
RUN_DIR="$RUN_ROOT/${VARIANT}_head"
mkdir -p "$RUN_DIR" results/scnet/logs models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$VARIANT" == control ]]; then
  CSV="data/raw/phase8_expansion_1m.csv"
elif [[ "$VARIANT" == target ]]; then
  CSV="data/raw/phase8_expansion_1m_residual40k.csv"
else
  echo "invalid PILOT_VARIANT=$VARIANT" >&2; exit 2
fi
ARGS=(
  --gps7-emb "$RUN_ROOT/${VARIANT}_gps7/embeddings.pt"
  --gps9-emb "$RUN_ROOT/${VARIANT}_gps9/embeddings.pt"
  --csv "$CSV" --tag "residual40k_${VARIANT}"
  --out "$RUN_DIR/metrics.json"
  --model-out "models/scnet/phase8_residual40k_${VARIANT}_dual2d.pt"
  --checkpoint-out "$RUN_DIR/training_state.pt"
  --epochs 150 --patience 25 --batch-size 4096 --eval-batch-size 8192
  --lr 0.00054 --weight-decay 0.00001 --hidden 192
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then ARGS+=(--resume-from "$RUN_DIR/training_state.pt"); fi
python scripts/phase8/training/train_dual_gps_2d_head.py "${ARGS[@]}"
