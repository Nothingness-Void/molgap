#!/usr/bin/env bash
# Submit with PILOT_VARIANT=control|target and GPS_DEPTH=7|9.
#SBATCH --job-name=molgap-res40k-gps
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
VARIANT="${PILOT_VARIANT:?set PILOT_VARIANT=control or target}"
DEPTH="${GPS_DEPTH:?set GPS_DEPTH=7 or 9}"
[[ "$DEPTH" == 7 || "$DEPTH" == 9 ]] || { echo "invalid GPS_DEPTH" >&2; exit 2; }
RUN_DIR="results/phase8/residual_target_40k_scnet/${VARIANT}_gps${DEPTH}"
mkdir -p "$RUN_DIR" results/scnet/logs models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$DEPTH" == 7 ]]; then
  INIT="models/scnet/phase8_gps_expansion_1m.pt"
  DEPTH_ARGS=()
else
  INIT="models/scnet/phase8_gps_expansion_1m_depth9.pt"
  DEPTH_ARGS=(--num-layers 9)
fi
if [[ "$VARIANT" == control ]]; then
  GRAPHS="results/phase8/pyg_2d_graphs_bond_expansion_1m.pt"
  REPLAY_ARGS=()
elif [[ "$VARIANT" == target ]]; then
  GRAPHS="results/phase8/residual_target_40k_scnet/pyg_2d_graphs_expansion_1m_residual40k.pt"
  REPLAY_ARGS=(--replay-boundary 1000000 --replay-weight 0.16)
else
  echo "invalid PILOT_VARIANT=$VARIANT" >&2; exit 2
fi

ARGS=(
  --kind gps --graphs "$GRAPHS" --init-from "$INIT" "${DEPTH_ARGS[@]}"
  --epochs 12 --patience 12 --lr 0.00002 --batch-size 256
  --eval-batch-size 256 --embedding-batch-size 256 "${REPLAY_ARGS[@]}"
  --checkpoint-out "$RUN_DIR/training_state.pt" --checkpoint-every 1
  --model-out "models/scnet/phase8_residual40k_${VARIANT}_gps${DEPTH}.pt"
  --metrics-out "$RUN_DIR/metrics.json" --embeddings-out "$RUN_DIR/embeddings.pt"
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then ARGS+=(--resume-from "$RUN_DIR/training_state.pt"); fi
python scripts/phase8/training/train_encoder.py "${ARGS[@]}"
