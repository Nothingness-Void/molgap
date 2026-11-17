#!/usr/bin/env bash
# Stage two independently retrievable FP16 expert chunks for Kaggle fusion.
#SBATCH --job-name=molgap-stage-2m1m-fusion
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_ROOT="results/phase8/multi2d_2m_1m3d_fusion_staging"
mkdir -p "$RUN_ROOT" results/scnet/logs
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase8/archive/scaleup/data/stage_multi2d_fusion_embeddings.py \
  --expert coverage2m=results/phase8/multi2d_2m_scnet/gps7/embeddings.pt,results/phase8/multi2d_2m_scnet/gps9/embeddings.pt \
  --expert hard20k=results/phase8/multi2d_2m_hard20k_scnet/gps7/embeddings.pt,results/phase8/multi2d_2m_hard20k_scnet/gps9/embeddings.pt \
  --rows 1000000 --out-dir "$RUN_ROOT" --report "$RUN_ROOT/staging_report.json"

cp models/scnet/phase8_multi2d_coverage_2m_gps7.pt "$RUN_ROOT/"
cp models/scnet/phase8_multi2d_coverage_2m_gps9.pt "$RUN_ROOT/"
cp models/scnet/phase8_multi2d_coverage_2m_dual2d.pt "$RUN_ROOT/"
cp models/scnet/phase8_multi2d_2m_hard20k_gps7.pt "$RUN_ROOT/"
cp models/scnet/phase8_multi2d_2m_hard20k_gps9.pt "$RUN_ROOT/"
cp models/scnet/phase8_multi2d_2m_hard20k_dual2d.pt "$RUN_ROOT/"
sha256sum "$RUN_ROOT"/*.pt > "$RUN_ROOT/SHA256SUMS"
