#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-vector-probe
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
: "${ROOT:?}"
: "${CODE:?}"
for path in "$ROOT" "$CODE"; do
  case "$(realpath "$path")" in /work/home/acf9jvb3sm/*) ;; *) exit 2 ;; esac
done
cd "$ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
PLATFORM_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$CODE/src:$PLATFORM_PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=/work/home/acf9jvb3sm/molgap-envs/xian-card2-torch110-pyg24/bin/python
"$PYTHON" "$CODE/tests/remote_pcqm_vector_state.py" \
  --output "$ROOT/model_preflight.json"
