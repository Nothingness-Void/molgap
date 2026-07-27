#!/usr/bin/env bash
# Submit with RETENTION_VARIANT=replay or replay_distill.
#SBATCH --job-name=molgap-ret2m-gps7
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RETENTION_VARIANT="${RETENTION_VARIANT:?set RETENTION_VARIANT=replay or replay_distill}"
case "$RETENTION_VARIANT" in
  replay|replay_distill) ;;
  *) echo "Unsupported RETENTION_VARIANT=$RETENTION_VARIANT" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/retention_2m_scnet/$RETENTION_VARIANT"
mkdir -p results/scnet/logs "$RUN_ROOT" models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --kind gps
  --graphs results/phase8/multi2d_2m_scnet/pyg_2d_graphs_multi2d_coverage_2m.pt
  --init-from models/scnet/phase8_repair_v3_1p5m_gps7.pt
  --epochs 40 --patience 12 --lr 0.00008
  --batch-size 256 --eval-batch-size 256 --embedding-batch-size 256
  --replay-boundary 500000 --replay-weight 3.0
  --checkpoint-out "$RUN_ROOT/training_state.pt" --checkpoint-every 1
  --model-out "models/scnet/phase8_retention_2m_${RETENTION_VARIANT}_gps7.pt"
  --metrics-out "$RUN_ROOT/metrics.json"
  --embeddings-out "$RUN_ROOT/embeddings.pt"
  --no-embeddings
)
if [[ "$RETENTION_VARIANT" == "replay_distill" ]]; then
  ARGS+=(
    --retention-teacher models/phase8_gps_expansion_500k.pt
    --retention-boundary 500000
    --retention-weight 0.25
    --retention-targets-cache "$RUN_ROOT/retention_targets_500k.pt"
  )
fi
if [[ -f "$RUN_ROOT/training_state.pt" ]]; then
  ARGS+=(--resume-from "$RUN_ROOT/training_state.pt")
fi
python scripts/phase8/training/train_encoder.py "${ARGS[@]}"
