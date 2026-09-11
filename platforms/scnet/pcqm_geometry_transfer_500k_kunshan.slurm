#!/usr/bin/env bash
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${MODE:?submit with MODE=preflight|train}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"

ROOT="$(realpath "$ROOT")"
case "$ROOT" in
  /public/home/scnaqkfcy3/*) ;;
  *) echo "refusing root outside the account home: $ROOT" >&2; exit 2 ;;
esac
CACHE="$ROOT/input/cache"
CODE="$ROOT/code"
OUTPUT="$ROOT/output"
RUNNER="$CODE/platforms/scnet/run_pcqm_geometry_transfer_500k.py"
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python

test -f "$RUNNER"
mkdir -p "$OUTPUT" "$ROOT/logs"
cd "$ROOT"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

case "$MODE" in
  preflight)
    : "${MODEL_ID:?preflight requires MODEL_ID}"
    test -f "$CACHE/manifest.json"
    test -f "$CACHE/acceptance.json"
    "$PYTHON" -u "$RUNNER" preflight \
      --cache-root "$CACHE" \
      --output "$OUTPUT/$MODEL_ID/preflight" \
      --model-id "$MODEL_ID"
    ;;
  train)
    : "${MODEL_ID:?train requires MODEL_ID}"
    test -f "$CACHE/manifest.json"
    test -f "$CACHE/acceptance.json"
    "$PYTHON" -u "$RUNNER" train \
      --cache-root "$CACHE" \
      --output "$OUTPUT/$MODEL_ID" \
      --model-id "$MODEL_ID" \
      --source-commit "$SOURCE_COMMIT"
    ;;
  *)
    echo "unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac
