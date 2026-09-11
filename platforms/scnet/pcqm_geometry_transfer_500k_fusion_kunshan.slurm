#!/usr/bin/env bash
#SBATCH --partition=kshctest02
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"

ROOT="$(realpath "$ROOT")"
case "$ROOT" in
  /public/home/scnaqkfcy3/*) ;;
  *) echo "refusing root outside the account home: $ROOT" >&2; exit 2 ;;
esac
CODE="$ROOT/code"
OUTPUT="$ROOT/output"
RUNNER="$CODE/platforms/scnet/run_pcqm_geometry_transfer_500k.py"
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python

test -f "$RUNNER"
mkdir -p "$OUTPUT" "$ROOT/logs"
cd "$ROOT"

module purge
module load compiler/dtk/23.10
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON" -u "$RUNNER" fusion --output "$OUTPUT"
