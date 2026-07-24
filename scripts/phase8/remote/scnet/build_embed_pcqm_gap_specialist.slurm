#!/usr/bin/env bash
#SBATCH --job-name=molgap-pcqm-gap-emb
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
TABLE="$RUN_ROOT/pcqm4mv2_train_hard_pool_clean.parquet"
GRAPHS="$RUN_ROOT/pcqm_clean_95909_2d_graphs.pt"
B3_GRAPHS="results/phase8/multi2d_2m_scnet/pyg_2d_graphs_multi2d_coverage_2m.pt"
BASE_MODEL="models/scnet/phase8_retention_2m_replay_gps7.pt"
mkdir -p results/scnet/logs "$RUN_ROOT"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/data/materialize_clean_pcqm_hard_pool.py \
  --input /work1/share/acf9jvb3sm/molgap-results/phase8/pcqm4mv2_train_residual_scan/pcqm4mv2_train_hard_pool.parquet \
  --output "$TABLE" \
  --report "$RUN_ROOT/clean_pool_report.json"

python scripts/phase8/data/build_gap_only_graphs.py \
  --input "$TABLE" --out "$GRAPHS" --shard-size 10000

python scripts/phase8/training/extract_distillation_teacher_embeddings.py \
  --graphs "$GRAPHS" \
  --model "$BASE_MODEL" \
  --num-layers 7 \
  --out-dir "$RUN_ROOT/pcqm_embeddings" \
  --batch-size 256 --chunk-size 10000

python scripts/phase8/training/extract_distillation_teacher_embeddings.py \
  --graphs "$B3_GRAPHS" \
  --model "$BASE_MODEL" \
  --num-layers 7 \
  --out-dir "$RUN_ROOT/b3_rehearsal_embeddings" \
  --batch-size 256 --chunk-size 50000 --max-rows 220000
