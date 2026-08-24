#!/usr/bin/env bash
#SBATCH --job-name=molgap-arch-screen
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --array=0-5

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${CODE:?submit with CODE}"
cd "$ROOT"
test "$(realpath "$ROOT")" = "$ROOT"
test "$(realpath "$CODE")" = "$CODE"
test -s "$ROOT/preflight.json"

KINDS=(gps gps gps structural_gps structural_gps structural_gps)
NAMES=(gps9_control gps9_control gps9_control structural_gps9 structural_gps9 structural_gps9)
SEEDS=(42 43 44 42 43 44)
KIND="${KINDS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$ROOT/runs/$NAME/seed$SEED"
mkdir -p "$RUN_DIR"

if [[ "$KIND" == "structural_gps" ]]; then
  GRAPH="$ROOT/inputs/pyg_2d_graphs_pubchemqc100k_rwse16.pt"
else
  GRAPH="$ROOT/inputs/pyg_2d_graphs_pubchemqc100k_architecture.pt"
fi
if [[ -s "$RUN_DIR/completion_manifest.json" ]]; then
  echo "accepted run already exists: $RUN_DIR"
  exit 0
fi

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$CODE/src:$HOME/molgap/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

ARGS=(
  --kind "$KIND"
  --graphs "$GRAPH"
  --split-csv "$ROOT/inputs/split.csv"
  --hidden-channels 192 --num-layers 9 --num-heads 4 --dropout 0.05
  --seed "$SEED" --split-seed 42
  --epochs 40 --patience 10 --lr 0.0004 --weight-decay 0.00001
  --batch-size 256 --eval-batch-size 256 --embedding-batch-size 256
  --rwse-dim 16 --no-embeddings
  --checkpoint-out "$RUN_DIR/training_state.pt" --checkpoint-every 1
  --model-out "$RUN_DIR/model.pt"
  --metrics-out "$RUN_DIR/metrics.json"
  --predictions-out "$RUN_DIR/test_predictions.pt"
)
if [[ -s "$RUN_DIR/training_state.pt" ]]; then
  ARGS+=(--resume-from "$RUN_DIR/training_state.pt")
fi
python "$CODE/production/03_train/scripts/training/train_encoder.py" "${ARGS[@]}"
python "$CODE/experiments/resource_bounded_architecture/accept_screen_run.py" \
  --run-dir "$RUN_DIR" --kind "$KIND" --seed "$SEED" \
  --graph "$GRAPH" --split-csv "$ROOT/inputs/split.csv"
