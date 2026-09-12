#!/usr/bin/env bash
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --time=08:00:00

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${MODE:?submit with MODE=preflight|train}"
: "${PYTHON_BIN:?submit with PYTHON_BIN}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"
: "${SOURCE_ARCHIVE:?submit with SOURCE_ARCHIVE}"
: "${SOURCE_ARCHIVE_SHA256:?submit with SOURCE_ARCHIVE_SHA256}"

ROOT="$(realpath "$ROOT")"
ACCOUNT_HOME="$(realpath "${ACCOUNT_HOME:-$HOME}")"
case "$ROOT" in
  "$ACCOUNT_HOME"/*) ;;
  *) echo "refusing ROOT outside account home: $ROOT" >&2; exit 2 ;;
esac

CODE="$ROOT/code"
DATASET="$ROOT/input/pcqm4mv2-ogb-fixed-100k-v1"
MANIFEST="$DATASET/manifest.json"
OUTPUT="$ROOT/output"
PREFLIGHT="$OUTPUT/preflight/preflight.json"
RUNNER="$CODE/experiments/pcqm_gptrans_t_100k_v4/run.py"
PLATFORM_ID="scnet-kunshan-dtk23.10-dcu66a1"

test -f "$RUNNER"
test -f "$MANIFEST"
test -f "$SOURCE_ARCHIVE"
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
    "$PYTHON_BIN" -u "$RUNNER" preflight \
      --dataset-root "$DATASET" \
      --manifest "$MANIFEST" \
      --source-archive "$SOURCE_ARCHIVE" \
      --source-archive-sha256 "$SOURCE_ARCHIVE_SHA256" \
      --source-commit "$SOURCE_COMMIT" \
      --output "$OUTPUT/preflight" \
      --platform-id "$PLATFORM_ID"
    ;;
  train)
    test -f "$PREFLIGHT"
    "$PYTHON_BIN" -u "$RUNNER" train \
      --dataset-root "$DATASET" \
      --manifest "$MANIFEST" \
      --preflight "$PREFLIGHT" \
      --source-archive "$SOURCE_ARCHIVE" \
      --source-archive-sha256 "$SOURCE_ARCHIVE_SHA256" \
      --source-commit "$SOURCE_COMMIT" \
      --output "$OUTPUT/training" \
      --platform-id "$PLATFORM_ID"
    ;;
  *)
    echo "unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac
