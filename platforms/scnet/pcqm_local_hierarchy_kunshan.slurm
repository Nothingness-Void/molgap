#!/usr/bin/env bash
#SBATCH --job-name=pcqm-hierarchy-ksh
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --time=12:00:00
set -euo pipefail
: "${CODE:?submit with CODE}"
: "${GRAPH_ROOT:?submit with GRAPH_ROOT}"
: "${LABEL_ROOT:?submit with LABEL_ROOT}"
: "${OUTPUT_ROOT:?submit with OUTPUT_ROOT}"
: "${LABEL_SHA256:?submit with LABEL_SHA256}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"
: "${ROLE:?submit with ROLE=scratch or pretrained}"
: "${PYTHON:?submit with the accepted DTK Python path}"

for path in "$CODE" "$GRAPH_ROOT" "$LABEL_ROOT" "$OUTPUT_ROOT"; do
  case "$(realpath -m "$path")" in "$HOME"/*) ;; *) exit 2 ;; esac
done
test "$ROLE" = scratch || test "$ROLE" = pretrained
test -s "$GRAPH_ROOT/manifest.json"
test -s "$LABEL_ROOT/manifest.json"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$CODE/src"
export MOLGAP_EXECUTION_PLATFORM=scnet-kunshan-dtk2310
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$OUTPUT_ROOT/$ROLE"
"$PYTHON" "$CODE/platforms/scnet/run_pcqm_local_hierarchy.py" worker \
  --role "$ROLE" --graph-root "$GRAPH_ROOT" --label-root "$LABEL_ROOT" \
  --output-root "$OUTPUT_ROOT" --label-sha256 "$LABEL_SHA256" \
  --source-commit "$SOURCE_COMMIT"
