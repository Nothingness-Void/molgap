#!/usr/bin/env bash
#SBATCH --job-name=molgap-gstate-xian-probe
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
: "${ROOT:?}"
: "${CODE:?}"
: "${CACHE:?}"
: "${SOURCE:?}"
for path in "$ROOT" "$CODE" "$CACHE" "$SOURCE"; do
  case "$(realpath "$path")" in /work/home/acf9jvb3sm/*) ;; *) exit 2 ;; esac
done

cd "$ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1
export PYTHONNOUSERSITE=1
PLATFORM_PYTHONPATH="${PYTHONPATH:-}"
EXTRA_SITE=/work/home/acf9jvb3sm/molgap-envs/xian-card2-torch110-pyg24/lib/python3.9/site-packages
export PYTHONPATH="$SOURCE/src:$EXTRA_SITE:$PLATFORM_PYTHONPATH"
RUNTIME_ROOT=/work/home/acf9jvb3sm/molgap-results/xian-card2-compatibility-20260905
export LD_LIBRARY_PATH="$RUNTIME_ROOT/glog-0.3.5-install/lib:$RUNTIME_ROOT/gflags-2.1.2-install/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python

"$PYTHON" "$CODE/probe_xian_graphstate_throughput.py" \
  --cache-root "$CACHE" \
  --output "$ROOT/throughput.json" \
  --batch-sizes 48 96 192
