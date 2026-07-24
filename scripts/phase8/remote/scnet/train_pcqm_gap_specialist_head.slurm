#!/usr/bin/env bash
#SBATCH --job-name=molgap-pcqm-gap-head
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/pcqm_gap_head_pilot"
mkdir -p results/scnet/logs "$RUN_ROOT" models/scnet

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/training/train_gap_specialist_head.py \
  --base-model models/scnet/phase8_retention_2m_replay_gps7.pt \
  --b3-embeddings "$RUN_ROOT/b3_rehearsal_embeddings/manifest.json" \
  --b3-graphs results/phase8/multi2d_2m_scnet/pyg_2d_graphs_multi2d_coverage_2m.pt \
  --pcqm-embeddings "$RUN_ROOT/pcqm_embeddings/manifest.json" \
  --pcqm-table "$RUN_ROOT/pcqm4mv2_train_hard_pool_clean.parquet" \
  --run-dir "$RUN_ROOT/b_replay_head_w30" \
  --model-out models/scnet/phase8_pcqm_gap_head_b_replay_w30_gps7.pt \
  --pcqm-weight 0.30 --epochs 50 --patience 10 \
  --batch-size 2048 --lr 0.0002 --weight-decay 0.00001 \
  --b3-train-rows 200000 --b3-validation-rows 20000 \
  --seed 42 --scaffold-workers 8
