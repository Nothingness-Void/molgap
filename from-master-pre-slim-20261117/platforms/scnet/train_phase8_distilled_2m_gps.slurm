#!/usr/bin/env bash
#SBATCH --job-name=molgap-kd-student
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/distilled_2m_scnet"
mkdir -p results/scnet/logs "$RUN_ROOT" models/scnet

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/training/train_distilled_gps.py \
  --graphs results/phase8/multi2d_2m_scnet/pyg_2d_graphs_multi2d_coverage_2m.pt \
  --student-init models/scnet/phase8_multi2d_coverage_2m_gps7.pt \
  --teacher control_a \
    "$RUN_ROOT/teacher_embeddings/control_a_gps7" \
    "$RUN_ROOT/teacher_embeddings/control_a_gps9" \
    models/scnet/phase8_residual40k_control_dual2d.pt \
  --teacher repair_v2 \
    "$RUN_ROOT/teacher_embeddings/repair_v2_gps7" \
    "$RUN_ROOT/teacher_embeddings/repair_v2_gps9" \
    models/scnet/phase8_dual_gps_2d_v2_1m.pt \
  --run-dir "$RUN_ROOT/student_gps7" \
  --model-out models/scnet/phase8_distilled_teacher_2m_gps7.pt \
  --epochs 30 --patience 10 \
  --batch-size 256 --eval-batch-size 256 \
  --lr 0.00004 --weight-decay 0.00001 \
  --teacher-weight 0.70 \
  --fusion-prefix-rows 997445
