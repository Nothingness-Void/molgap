#!/bin/bash
# Repaired-2M primary ETKDG construction on the Kunshan CPU-only partition.
#SBATCH --job-name=molgap-rep2m-3d
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/molgap}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/molgap-3d-env/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/molgap-inputs/repaired_2m_3d}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/molgap-results/phase8/repaired_2m_3d}"
WORKERS="${WORKERS:-28}"

REPAIRED_CSV="$INPUT_ROOT/phase8_repaired_2m.csv"
ORIGINAL_CSV="$INPUT_ROOT/phase8_expansion_1m.csv"
ORIGINAL_GRAPHS="$INPUT_ROOT/pyg_3d_graphs_etkdg_expansion_1m.pt"
GRAPH_DIR="$OUTPUT_ROOT/graph_shards"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_DIR"
mkdir -p logs "$GRAPH_DIR"
for path in "$PYTHON_BIN" "$REPAIRED_CSV" "$ORIGINAL_CSV" "$ORIGINAL_GRAPHS"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done

echo "host=$(hostname)"
echo "python=$($PYTHON_BIN --version)"
echo "workers=$WORKERS"
echo "start=$(date -Is)"
"$PYTHON_BIN" - <<'PY'
import pandas
import rdkit
import torch
import torch_geometric
print(
    "runtime",
    "torch=" + torch.__version__,
    "pyg=" + torch_geometric.__version__,
    "rdkit=" + rdkit.__version__,
    "pandas=" + pandas.__version__,
)
PY

"$PYTHON_BIN" scripts/phase8/data/build_repaired_2m_3d_shards.py \
  --repaired-csv "$REPAIRED_CSV" \
  --original-csv "$ORIGINAL_CSV" \
  --original-graphs "$ORIGINAL_GRAPHS" \
  --output-dir "$GRAPH_DIR" \
  --shard-size 20000 \
  --workers "$WORKERS" \
  --seed 42

"$PYTHON_BIN" - <<'PY' "$GRAPH_DIR"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
completion = json.loads((root / "build_completion.json").read_text())
if completion["status"] != "complete" or completion["shards"] != 100:
    raise SystemExit(f"incomplete graph build: {completion}")
print(json.dumps(completion, indent=2))
PY
echo "end=$(date -Is)"
