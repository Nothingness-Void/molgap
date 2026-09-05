#!/usr/bin/env bash
#SBATCH --job-name=molgap-gstate-xian
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
: "${ROOT:?submit with ROOT}"
: "${CODE:?submit with CODE}"
: "${CACHE:?submit with CACHE}"
: "${SOURCE:?submit with SOURCE}"

for path in "$ROOT" "$CODE" "$CACHE" "$SOURCE"; do
  case "$(realpath "$path")" in
    /work/home/acf9jvb3sm/*) ;;
    *) exit 2 ;;
  esac
done
test -f "$CACHE/manifest.json"
test -f "$SOURCE/src/molgap/pcqm_gap_architecture.py"

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

"$PYTHON" "$CODE/pcqm_graphstate_benchmark.py" \
  --cache-root "$CACHE" \
  --output-root "$ROOT/output" \
  --model-source-commit 9068ddb82e6bdf16b841570abbff023b90c07f07 \
  --seed 42 --epochs 3 --batch-size 48
