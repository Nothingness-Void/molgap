#!/usr/bin/env bash
# Fetch disjoint 1M-v2 PubChemQC CSV candidates only; no graphs or accelerators.
# Submit with: sbatch scripts/phase8/archive/remote/scnet/fetch_phase8_repair_v2_array.slurm
# The four groups run independently so rare chemistry does not block general rows.
#SBATCH --job-name=molgap-p8-v2-fetch
#SBATCH --partition=hx1hdnormal
# The account QoS rejects CPU-only jobs; this task does not use the allocated DCU.
#SBATCH --gres=dcu:Hygon:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=results/phase8/remote/repair_1m_v2_fetch/%x-%A_%a.out
#SBATCH --error=results/phase8/remote/repair_1m_v2_fetch/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/molgap}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.12/bin/python3}"
FETCH_SITE="$PROJECT_DIR/.scnet/fetch-py312"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    GROUP="rare"
    MAX_KEPT=10000
    BUCKETS=(very_low_gap low_gap_aromatic_edge)
    ;;
  1)
    GROUP="aromatic_large"
    MAX_KEPT=80000
    BUCKETS=(large_aromatic_edge very_large_general aromatic_edge_general)
    ;;
  2)
    GROUP="topology_elements"
    MAX_KEPT=120000
    BUCKETS=(s_or_cl_hard flexible_hard large_mw_500_700)
    ;;
  3)
    GROUP="balanced"
    MAX_KEPT=390000
    BUCKETS=(balanced_general)
    ;;
  *)
    echo "Unsupported array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
mkdir -p results/phase8/remote/repair_1m_v2_fetch data/raw
export PYTHONPATH="$PROJECT_DIR/src:$FETCH_SITE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT="data/raw/phase8_repair_v2_${GROUP}.csv"
REPORT="results/phase8/repair_1m_v2_${GROUP}_report.json"

echo "host=$(hostname) group=$GROUP start=$(date -Is)"
"$PYTHON_BIN" -c "import ijson, pandas, rdkit; print('fetch_deps=ok')"
"$PYTHON_BIN" scripts/phase8/archive/scaleup/data/fetch_repair_candidates.py \
  --train-csv data/raw/phase8_expansion_1m.csv \
  --include-buckets "${BUCKETS[@]}" \
  --max-kept "$MAX_KEPT" \
  --windows-per-file 16 \
  --chunk-bytes 16000000 \
  --download-workers 4 \
  --out-csv "$OUT" \
  --report-json "$REPORT" \
  --overwrite
echo "host=$(hostname) group=$GROUP end=$(date -Is)"
