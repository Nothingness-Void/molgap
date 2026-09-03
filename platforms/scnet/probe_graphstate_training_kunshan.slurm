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
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python
"$PYTHON" "$PROJECT_DIR/platforms/scnet/probe_graphstate_training.py" \
  --output "$PROBE_OUTPUT"
