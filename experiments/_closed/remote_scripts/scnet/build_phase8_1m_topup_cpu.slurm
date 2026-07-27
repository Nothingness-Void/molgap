#!/bin/bash
#SBATCH --job-name=molgap-3d-topup
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# This is CPU-only ETKDG graph construction. It does not train SchNet or use a DCU.
PROJECT_DIR="${PROJECT_DIR:-$HOME/molgap}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/molgap-3d-env/bin/python}"
GRAPH_TAG="${GRAPH_TAG:-expansion_1m_topup_full}"
START_ROW="${START_ROW:-500000}"
N_JOBS="${N_JOBS:-28}"
MAX_ROWS="${MAX_ROWS:-}"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT_DIR"
mkdir -p logs results/phase8

echo "host=$(hostname)"
echo "python=$($PYTHON_BIN --version)"
echo "tag=$GRAPH_TAG start_row=$START_ROW max_rows=${MAX_ROWS:-all} n_jobs=$N_JOBS"
echo "start=$(date -Is)"

args=(
  --csv data/raw/phase8_expansion_1m.csv
  --which 3d
  --tag "$GRAPH_TAG"
  --start-row "$START_ROW"
  --n-jobs "$N_JOBS"
  --resume
)
if [[ -n "$MAX_ROWS" ]]; then
  args+=(--max-rows "$MAX_ROWS")
fi

"$PYTHON_BIN" scripts/phase8/data/build_graphs.py "${args[@]}"

echo "end=$(date -Is)"
