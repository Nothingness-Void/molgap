#!/usr/bin/env bash
#SBATCH --job-name=pcqm-hierarchy-labels
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
: "${CODE:?submit with CODE}"
: "${GRAPH_ROOT:?submit with GRAPH_ROOT}"
: "${SOURCE_CSV:?submit with SOURCE_CSV}"
: "${OUTPUT_ROOT:?submit with OUTPUT_ROOT}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"
: "${PYTHON:?submit with the accepted CPU Python path}"
: "${DEPS:?submit with the accepted CPU dependency path}"

for path in "$CODE" "$GRAPH_ROOT" "$SOURCE_CSV" "$OUTPUT_ROOT" "$PYTHON" "$DEPS"; do
  case "$(realpath -m "$path")" in "$HOME"/*) ;; *) exit 2 ;; esac
done
test -s "$GRAPH_ROOT/manifest.json"
test -s "$SOURCE_CSV"
test -x "$PYTHON"
test -d "$DEPS"

module purge
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$DEPS:$CODE/src"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
mkdir -p "$OUTPUT_ROOT"
"$PYTHON" "$CODE/platforms/scnet/run_pcqm_local_hierarchy.py" build-labels \
  --source-csv "$SOURCE_CSV" --graph-root "$GRAPH_ROOT" \
  --output-root "$OUTPUT_ROOT" --source-commit "$SOURCE_COMMIT"
"$PYTHON" "$CODE/platforms/scnet/run_pcqm_local_hierarchy.py" accept-labels \
  --output-root "$OUTPUT_ROOT" --source-commit "$SOURCE_COMMIT"
