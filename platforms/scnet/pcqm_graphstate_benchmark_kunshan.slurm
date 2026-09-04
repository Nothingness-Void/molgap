#!/usr/bin/env bash
#SBATCH --job-name=molgap-gstate-kunshan
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${CODE:?submit with CODE}"
: "${CACHE:?submit with CACHE}"
: "${SOURCE:?submit with SOURCE}"

cd "$ROOT"
test "$(realpath "$ROOT")" = "$ROOT"
test "$(realpath "$CODE")" = "$CODE"
test "$(realpath "$CACHE")" = "$CACHE"
test "$(realpath "$SOURCE")" = "$SOURCE"
test -f "$CACHE/manifest.json"
test -f "$SOURCE/src/molgap/pcqm_gap_architecture.py"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python

"$PYTHON" "$CODE/pcqm_graphstate_benchmark.py" \
  --cache-root "$CACHE" \
  --output-root "$ROOT/output" \
  --model-source-commit 9068ddb82e6bdf16b841570abbff023b90c07f07 \
  --seed 42 --epochs 3 --batch-size 48
