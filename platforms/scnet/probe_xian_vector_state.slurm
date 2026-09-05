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
EXTRA_SITE=/work/home/acf9jvb3sm/molgap-envs/xian-card2-torch110-pyg24/lib/python3.9/site-packages
export PYTHONPATH="$CODE/src:$EXTRA_SITE:$PLATFORM_PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python
"$PYTHON" "$CODE/tests/remote_pcqm_vector_state.py" \
  --output "$ROOT/model_preflight.json"
