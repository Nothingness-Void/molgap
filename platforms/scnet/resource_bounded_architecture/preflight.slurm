#!/usr/bin/env bash
#SBATCH --job-name=molgap-arch-preflight
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${CODE:?submit with CODE}"
cd "$ROOT"
test "$(realpath "$ROOT")" = "$ROOT"
test "$(realpath "$CODE")" = "$CODE"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$CODE/src:$HOME/molgap/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python "$CODE/experiments/resource_bounded_architecture/preflight.py" \
  --base-graph "$ROOT/inputs/pyg_2d_graphs_pubchemqc100k_architecture.pt" \
  --rwse-graph "$ROOT/inputs/pyg_2d_graphs_pubchemqc100k_rwse16.pt" \
  --split-csv "$ROOT/inputs/split.csv" \
  --output "$ROOT/preflight.json"
