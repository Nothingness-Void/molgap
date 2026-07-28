#!/bin/bash
# Strictly accept the independently seeded secondary repaired-2M graph shards.
# This job never builds or modifies graph shards. It also proves the secondary
# coordinates differ from the accepted primary view, which no count or hash check
# can establish on its own.
#SBATCH --job-name=molgap-rep2m-3d-accept-secondary
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14200M
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/molgap}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/molgap-3d-env/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/molgap-inputs/repaired_2m_3d}"
PRIMARY_ROOT="${PRIMARY_ROOT:-$HOME/molgap-results/phase8/repaired_2m_3d}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/molgap-results/phase8/repaired_2m_3d_secondary}"

SOURCE_CSV="$INPUT_ROOT/phase8_repaired_2m.csv"
GRAPH_DIR="$OUTPUT_ROOT/graph_shards"
PRIMARY_MANIFEST="$PRIMARY_ROOT/primary_acceptance.json"
PRIMARY_GRAPH_DIR="$PRIMARY_ROOT/graph_shards"
ACCEPTANCE="$OUTPUT_ROOT/secondary_acceptance.json"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_DIR"
mkdir -p logs
for path in "$PYTHON_BIN" "$SOURCE_CSV" "$GRAPH_DIR" "$PRIMARY_MANIFEST" "$PRIMARY_GRAPH_DIR"; do
  test -e "$path" || { echo "missing input: $path" >&2; exit 2; }
done

echo "host=$(hostname)"
echo "python=$($PYTHON_BIN --version)"
echo "source_csv=$SOURCE_CSV"
echo "graph_dir=$GRAPH_DIR"
echo "primary_manifest=$PRIMARY_MANIFEST"
echo "start=$(date -Is)"

"$PYTHON_BIN" production/04_evaluate/scripts/evaluation/accept_repaired_2m_secondary_graphs.py \
  --shard-dir "$GRAPH_DIR" \
  --source-csv "$SOURCE_CSV" \
  --primary-manifest "$PRIMARY_MANIFEST" \
  --primary-shard-dir "$PRIMARY_GRAPH_DIR" \
  --output "$ACCEPTANCE" \
  --expected-shards 100 \
  --expected-seed 314159

"$PYTHON_BIN" - <<'PY' "$ACCEPTANCE"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if (
    not report.get("accepted")
    or report.get("view") != "independent_secondary_etkdg"
    or report.get("seed") != 314159
    or report.get("expected_shards") != 100
    or report.get("source_rows") != 2_000_000
    or report.get("requested_rows") != 1_989_116
    or report.get("accepted_rows") != 1_986_868
    or report.get("failed") != 2_248
    or report.get("requested_rows") != report.get("primary_accepted_rows")
    or report.get("unique_source_idx") != report.get("accepted_rows")
    or report.get("unique_cid") != report.get("accepted_rows")
    or report.get("coordinates_compared") != report.get("accepted_rows")
    or report.get("coordinates_distinct") != report.get("coordinates_compared")
    or not report.get("target_alignment")
):
    raise SystemExit("secondary acceptance contract failed: " + repr(report))
print(json.dumps({
    "accepted": True,
    "view": report["view"],
    "seed": report["seed"],
    "shards": report["expected_shards"],
    "requested_rows": report["requested_rows"],
    "accepted_rows": report["accepted_rows"],
    "failed": report["failed"],
    "paired_rows": report["paired_rows"],
    "coordinates_distinct_fraction": report["coordinates_distinct_fraction"],
    "primary_manifest_sha256": report["primary_manifest_sha256"],
    "source_csv_sha256": report["source_csv_sha256"],
}, indent=2))
PY

echo "end=$(date -Is)"
