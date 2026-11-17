#!/usr/bin/env bash
#SBATCH --job-name=molgap-kd-teacher
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --array=0-3%2
#SBATCH --output=results/scnet/logs/%x-%A_%a.out
#SBATCH --error=results/scnet/logs/%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/distilled_2m_scnet"
GRAPH_CACHE="results/phase8/multi2d_2m_scnet/pyg_2d_graphs_multi2d_coverage_2m.pt"
mkdir -p results/scnet/logs "$RUN_ROOT/teacher_embeddings"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

NAMES=(control_a_gps7 control_a_gps9 repair_v2_gps7 repair_v2_gps9)
LAYERS=(7 9 7 9)
MODELS=(
  models/scnet/phase8_residual40k_control_gps7.pt
  models/scnet/phase8_residual40k_control_gps9.pt
  models/scnet/phase8_repair_v2_1m_gps7_control.pt
  models/scnet/phase8_repair_v2_1m_gps9_control.pt
)
INDEX="${SLURM_ARRAY_TASK_ID}"

python scripts/phase8/training/extract_distillation_teacher_embeddings.py \
  --graphs "$GRAPH_CACHE" \
  --model "${MODELS[$INDEX]}" \
  --num-layers "${LAYERS[$INDEX]}" \
  --out-dir "$RUN_ROOT/teacher_embeddings/${NAMES[$INDEX]}" \
  --batch-size 256 \
  --chunk-size 50000
