#!/usr/bin/env bash
#SBATCH --partition=kshctest02
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
: "${ROOT:?submit with ROOT}"

ROOT="$(realpath "$ROOT")"
case "$ROOT" in
  /public/home/scnaqkfcy3/*) ;;
  *) echo "refusing root outside the account home: $ROOT" >&2; exit 2 ;;
esac
CACHE="$ROOT/input/cache"
CODE="$ROOT/code"
OUTPUT="$ROOT/output/acceptance"
mkdir -p "$OUTPUT" "$ROOT/logs"
cd "$ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/cpu-cache-py311/bin/python
RUNNER="$CODE/platforms/scnet/run_pcqm_distance_angle_500k.py"

"$PYTHON" -u "$RUNNER" accept \
  --cache-root "$CACHE" \
  --output "$OUTPUT"
