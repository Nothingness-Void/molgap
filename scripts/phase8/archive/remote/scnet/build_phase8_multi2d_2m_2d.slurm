#!/usr/bin/env bash
# Assemble the exact 2M table, reuse/build its new 500K graphs, then merge with extra host memory.
#SBATCH --job-name=molgap-multi2d-2m-build
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G

set -euxo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
RUN_DIR="results/phase8/multi2d_2m_scnet"
mkdir -p results/scnet/logs "$RUN_DIR"
echo "stage=environment started=$(date -Is) host=$(hostname)"
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
echo "stage=environment_complete at=$(date -Is)"

FULL_CSV="data/raw/phase8_multi2d_coverage_2m.csv"
ASSEMBLY_REPORT="$RUN_DIR/full_csv_assembly_report.json"
if [[ -s "$FULL_CSV" && -s "$ASSEMBLY_REPORT" ]] \
  && grep -q '"assembled_rows": 2000000' "$ASSEMBLY_REPORT" \
  && grep -q 'ec8721bcd88a8e29ea9fe6b081565e8d1f1669ec895945f48a2a9efcc8026dd6' "$ASSEMBLY_REPORT"; then
  echo "stage=assembly_reused csv=$FULL_CSV"
else
  python scripts/phase8/archive/scaleup/data/assemble_repair_v2_1m.py \
    --base-csv data/raw/phase8_repair_v3_1p5m.csv \
    --topup-csv data/raw/phase8_multi2d_2m_topup_500k.csv \
    --out-csv "$FULL_CSV" \
    --report-json "$ASSEMBLY_REPORT" \
    --dataset-name phase8_multi2d_coverage_expert_2m \
    --expected-base-rows 1500000 --expected-topup-rows 500000
fi

TOPUP_CACHE="$RUN_DIR/pyg_2d_graphs_multi2d_2m_topup_500k.pt"
if [[ ! -s "$TOPUP_CACHE" ]]; then
  python scripts/phase8/data/build_graphs.py \
    --csv data/raw/phase8_multi2d_coverage_2m.csv \
    --which 2d --tag multi2d_2m_topup_500k \
    --start-row 1500000 --max-rows 500000 --resume \
    --out-2d "$TOPUP_CACHE"
else
  echo "Reusing completed top-up cache: $TOPUP_CACHE"
fi

FULL_CACHE="$RUN_DIR/pyg_2d_graphs_multi2d_coverage_2m.pt"
if [[ ! -s "$FULL_CACHE" ]]; then
  echo "stage=merge_started at=$(date -Is)"
  /usr/bin/time -v python scripts/phase8/data/merge_2d_graph_caches.py \
    --base results/phase8/repair_v3_1p5m_scnet/pyg_2d_graphs_bond_repair_v3_1p5m.pt \
    --append "$TOPUP_CACHE" --out "$FULL_CACHE" \
    --report "$RUN_DIR/graph_merge_report.json" \
    --expected-base 1500000 --expected-append 500000
else
  echo "Reusing completed full cache: $FULL_CACHE"
fi

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("results/phase8/multi2d_2m_scnet/graph_merge_report.json").read_text())
path = Path(report["out"])
assert report["total_graphs"] == 2_000_000
assert report["contiguous_source_idx"] is True
assert report["source_idx_min"] == 0 and report["source_idx_max"] == 1_999_999
assert path.stat().st_size == report["output_bytes"]
print(f"validated_2d_graphs=2,000,000 path={path}", flush=True)
PY
echo "stage=complete at=$(date -Is)"
