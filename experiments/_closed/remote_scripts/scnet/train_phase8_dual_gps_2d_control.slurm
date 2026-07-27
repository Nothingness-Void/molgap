#!/usr/bin/env bash
# Train the same GPS7+GPS9-only gated head for a controlled dataset variant.
# Submit examples:
#   sbatch --export=ALL,DATASET_VARIANT=v1 scripts/phase8/archive/remote/scnet/train_phase8_dual_gps_2d_control.slurm
#   sbatch --export=ALL,DATASET_VARIANT=v2 scripts/phase8/archive/remote/scnet/train_phase8_dual_gps_2d_control.slurm
#SBATCH --job-name=molgap-dual2d-control
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
VARIANT="${DATASET_VARIANT:?set DATASET_VARIANT to v1 or v2}"
RUN_DIR="results/phase8/dual_gps_2d_control/$VARIANT"
mkdir -p results/scnet/logs "$RUN_DIR" models/scnet
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

case "$VARIANT" in
  v1)
    csv="data/raw/phase8_expansion_1m.csv"
    gps7="results/phase8/gps_expansion_1m_embeddings.pt"
    gps9="results/phase8/gps_expansion_1m_depth9_embeddings.pt"
    rows_tag="1m"
    ;;
  v2)
    csv="data/raw/phase8_repair_v2_1m.csv"
    gps7="results/phase8/repair_v2_scnet/gps7_control_uniform/embeddings.pt"
    gps9="results/phase8/repair_v2_scnet/gps9_control_uniform/embeddings.pt"
    rows_tag="1m"
    ;;
  v3_1p5m)
    csv="data/raw/phase8_repair_v3_1p5m.csv"
    gps7="results/phase8/repair_v3_1p5m_scnet/gps7_uniform/embeddings.pt"
    gps9="results/phase8/repair_v3_1p5m_scnet/gps9_uniform/embeddings.pt"
    rows_tag="1p5m"
    ;;
  *) echo "Unsupported DATASET_VARIANT=$VARIANT" >&2; exit 2 ;;
esac

args=(
  --gps7-emb "$gps7" --gps9-emb "$gps9" --csv "$csv" --tag "dual_gps_2d_${VARIANT}_${rows_tag}"
  --out "$RUN_DIR/metrics.json" --model-out "models/scnet/phase8_dual_gps_2d_${VARIANT}_${rows_tag}.pt"
  --checkpoint-out "$RUN_DIR/training_state.pt"
  --epochs 150 --patience 25 --batch-size 4096 --eval-batch-size 8192
  --lr 0.00054 --weight-decay 0.00001 --hidden 192
)
if [[ -f "$RUN_DIR/training_state.pt" ]]; then args+=(--resume-from "$RUN_DIR/training_state.pt"); fi
python scripts/phase8/training/train_dual_gps_2d_head.py "${args[@]}"
