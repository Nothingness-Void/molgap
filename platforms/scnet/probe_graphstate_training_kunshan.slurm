#!/usr/bin/env bash
#SBATCH --job-name=molgap-graphstate-probe
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the isolated probe checkout}"
: "${PROBE_OUTPUT:?Set PROBE_OUTPUT to a dedicated JSON path}"

module purge
module load apps/PyTorch/1.10/dtk-22.04-py37

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
python "$PROJECT_DIR/platforms/scnet/probe_graphstate_training.py" \
  --output "$PROBE_OUTPUT"
