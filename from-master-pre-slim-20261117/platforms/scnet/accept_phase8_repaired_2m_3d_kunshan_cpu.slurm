#!/bin/bash
# Strictly accept the existing repaired-2M primary graph shards.
# This job never builds or modifies graph shards.
#SBATCH --job-name=molgap-rep2m-3d-accept
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14200M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/molgap}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/molgap-3d-env/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/molgap-inputs/repaired_2m_3d}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/molgap-results/phase8/repaired_2m_3d}"

SOURCE_CSV="$INPUT_ROOT/phase8_repaired_2m.csv"
GRAPH_DIR="$OUTPUT_ROOT/graph_shards"
ACCEPTANCE="$OUTPUT_ROOT/primary_acceptance.json"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_DIR"
mkdir -p logs
for path in "$PYTHON_BIN" "$SOURCE_CSV" "$GRAPH_DIR"; do
  test -e "$path" || { echo "missing input: $path" >&2; exit 2; }
done

echo "host=$(hostname)"
echo "python=$($PYTHON_BIN --version)"
echo "source_csv=$SOURCE_CSV"
echo "graph_dir=$GRAPH_DIR"
echo "start=$(date -Is)"

"$PYTHON_BIN" scripts/phase8/evaluation/accept_repaired_2m_3d_graphs.py \
  --shard-dir "$GRAPH_DIR" \
  --source-csv "$SOURCE_CSV" \
  --output "$ACCEPTANCE" \
  --expected-shards 100

"$PYTHON_BIN" - <<'PY' "$ACCEPTANCE"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if (
    not report.get("accepted")
    or report.get("expected_shards") != 100
    or report.get("source_rows") != 2_000_000
    or report.get("accepted_rows") != 1_989_116
    or report.get("failed") != 10_884
    or report.get("unique_source_idx") != report.get("accepted_rows")
    or report.get("unique_cid") != report.get("accepted_rows")
    or not report.get("target_alignment")
):
    raise SystemExit("primary acceptance contract failed: " + repr(report))
print(json.dumps({
    "accepted": True,
    "shards": report["expected_shards"],
    "source_rows": report["source_rows"],
    "accepted_rows": report["accepted_rows"],
    "failed": report["failed"],
    "source_csv_sha256": report["source_csv_sha256"],
}, indent=2))
PY

echo "end=$(date -Is)"
