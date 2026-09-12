#!/usr/bin/env bash
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --time=00:05:00

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${PYTHON_BIN:?submit with PYTHON_BIN}"

ROOT="$(realpath "$ROOT")"
ACCOUNT_HOME="$(realpath "${ACCOUNT_HOME:-$HOME}")"
case "$ROOT" in
  "$ACCOUNT_HOME"/*) ;;
  *) echo "refusing ROOT outside account home: $ROOT" >&2; exit 2 ;;
esac

CODE="$ROOT/code"
mkdir -p "$ROOT/logs"
cd "$ROOT"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON_BIN" - <<'PY'
from molgap.pcqm_gptrans_v4 import (
    EXPECTED_INITIAL_MODEL_SHA256,
    _make_model,
    _state_sha256,
    configure_fp32_determinism,
)

configure_fp32_determinism(42)
model = _make_model().to("cuda")
print(f"observed={_state_sha256(model)}", flush=True)
print(f"expected={EXPECTED_INITIAL_MODEL_SHA256}", flush=True)
PY
