#!/usr/bin/env bash
# Append the leakage-safe hard20K and build a durable 2.02M graph cache.
#SBATCH --job-name=molgap-2m-hard20k-build
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
RUN_ROOT="results/phase8/multi2d_2m_hard20k_scnet"
mkdir -p "$RUN_ROOT" results/scnet/logs
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"

FULL_CSV="data/raw/phase8_multi2d_coverage_2m_hard20k.csv"
python scripts/phase8/archive/scaleup/data/assemble_repair_v2_1m.py \
  --base-csv data/raw/phase8_multi2d_coverage_2m.csv \
  --topup-csv data/raw/phase8_multi2d_2m_hard20k.csv \
  --out-csv "$FULL_CSV" --report-json "$RUN_ROOT/full_csv_assembly_report.json" \
  --dataset-name phase8_multi2d_2m_hard20k --expected-base-rows 2000000 \
  --expected-topup-rows 20000

TOPUP_CACHE="$RUN_ROOT/pyg_2d_graphs_hard20k.pt"
if [[ ! -s "$TOPUP_CACHE" ]]; then
  python scripts/phase8/data/build_graphs.py \
    --csv "$FULL_CSV" --which 2d --tag multi2d_2m_hard20k \
    --start-row 2000000 --max-rows 20000 --resume --out-2d "$TOPUP_CACHE"
fi

FULL_CACHE="$RUN_ROOT/pyg_2d_graphs_multi2d_2m_hard20k.pt"
if [[ ! -s "$FULL_CACHE" ]]; then
  /usr/bin/time -v python scripts/phase8/data/merge_2d_graph_caches.py \
    --base results/phase8/multi2d_2m_scnet/pyg_2d_graphs_multi2d_coverage_2m.pt \
    --append "$TOPUP_CACHE" --out "$FULL_CACHE" \
    --report "$RUN_ROOT/graph_merge_report.json" \
    --expected-base 2000000 --expected-append 20000
fi

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("results/phase8/multi2d_2m_hard20k_scnet/graph_merge_report.json").read_text())
assert report["total_graphs"] == 2_020_000
assert report["contiguous_source_idx"] is True
assert report["source_idx_min"] == 0 and report["source_idx_max"] == 2_019_999
assert Path(report["out"]).stat().st_size == report["output_bytes"]
print("validated_2d_graphs=2,020,000", flush=True)
PY
