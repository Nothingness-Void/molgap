#!/bin/bash
# Repaired-2M independent secondary ETKDG view on Kunshan CPU.
# Submit with --dependency=afterok:<primary-job-id>.
#SBATCH --job-name=molgap-rep2m-3d-b
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/molgap}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/molgap-3d-env/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-$HOME/molgap-inputs/repaired_2m_3d}"
PRIMARY_ROOT="${PRIMARY_ROOT:-$HOME/molgap-results/phase8/repaired_2m_3d}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/molgap-results/phase8/repaired_2m_3d_secondary}"
WORKERS="${WORKERS:-28}"

REPAIRED_CSV="$INPUT_ROOT/phase8_repaired_2m.csv"
PRIMARY_GRAPH_DIR="$PRIMARY_ROOT/graph_shards"
SECONDARY_GRAPH_DIR="$OUTPUT_ROOT/graph_shards"
PRIMARY_ACCEPTANCE="$PRIMARY_ROOT/primary_acceptance.json"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$PROJECT_DIR"
mkdir -p logs "$SECONDARY_GRAPH_DIR"
for path in "$PYTHON_BIN" "$REPAIRED_CSV" "$PRIMARY_GRAPH_DIR"; do
  test -e "$path" || { echo "missing input: $path" >&2; exit 2; }
done

echo "host=$(hostname)"
echo "python=$($PYTHON_BIN --version)"
echo "workers=$WORKERS"
echo "seed=314159"
echo "start=$(date -Is)"

# The dependency checks primary process success; this gate checks scientific
# completeness and identity before spending CPU on a second conformer.
"$PYTHON_BIN" scripts/phase8/evaluation/accept_repaired_2m_3d_graphs.py \
  --shard-dir "$PRIMARY_GRAPH_DIR" \
  --source-csv "$REPAIRED_CSV" \
  --output "$PRIMARY_ACCEPTANCE" \
  --expected-shards 100

"$PYTHON_BIN" scripts/phase8/data/build_repaired_2m_secondary_3d_shards.py \
  --repaired-csv "$REPAIRED_CSV" \
  --primary-graph-dir "$PRIMARY_GRAPH_DIR" \
  --output-dir "$SECONDARY_GRAPH_DIR" \
  --workers "$WORKERS" \
  --seed 314159

test -s "$SECONDARY_GRAPH_DIR/build_completion.json"
test -s "$SECONDARY_GRAPH_DIR/validation.json"
echo "end=$(date -Is)"
