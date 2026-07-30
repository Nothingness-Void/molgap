#!/usr/bin/env bash
#SBATCH --job-name=rb-pcqm-gps
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail

: "${ENCODER_NAME:?submit with ENCODER_NAME=gps9 or gps11_160}"
: "${ROOT:?submit with ROOT}"
: "${CODE:?submit with CODE}"
cd "$ROOT"
test "$(realpath "$ROOT")" = "$ROOT"
test "$(realpath "$CODE")" = "$CODE"
test -f "preflight/${ENCODER_NAME}.json"
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$CODE/src:$HOME/molgap/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python "$CODE/platforms/scnet/pcqm_route_b/run_tuned_gps.py" \
  --root "$ROOT" --encoder "$ENCODER_NAME"
