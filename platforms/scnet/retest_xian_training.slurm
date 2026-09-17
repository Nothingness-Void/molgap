#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-retest
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
: "${RUN_ROOT:?submit with RUN_ROOT below /work/home/acf9jvb3sm}"
: "${AUDIT_SOURCE:?submit with AUDIT_SOURCE below /work/home/acf9jvb3sm}"
: "${MODEL_SOURCE:?submit with MODEL_SOURCE below /work/home/acf9jvb3sm}"
: "${CACHE:?submit with CACHE below /work/home/acf9jvb3sm}"

for path in "$RUN_ROOT" "$AUDIT_SOURCE" "$MODEL_SOURCE" "$CACHE"; do
  case "$(realpath "$path")" in
    /work/home/acf9jvb3sm/*) ;;
    *) echo "path outside authorized boundary: $path" >&2; exit 2 ;;
  esac
done

BENCHMARK="$MODEL_SOURCE/platforms/scnet/pcqm_graphstate_benchmark.py"
test -f "$BENCHMARK"
test -f "$CACHE/manifest.json"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs"
cd "$RUN_ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1

export PYTHONNOUSERSITE=1
PLATFORM_PYTHONPATH="${PYTHONPATH:-}"
EXTRA_SITE=/work/home/acf9jvb3sm/molgap-envs/xian-card2-torch110-pyg24/lib/python3.9/site-packages
export PYTHONPATH="$AUDIT_SOURCE/src:$EXTRA_SITE:$PLATFORM_PYTHONPATH"
RUNTIME_ROOT=/work/home/acf9jvb3sm/molgap-results/xian-card2-compatibility-20260905
export LD_LIBRARY_PATH="$RUNTIME_ROOT/glog-0.3.5-install/lib:$RUNTIME_ROOT/gflags-2.1.2-install/lib:${LD_LIBRARY_PATH:-}"
export ROCBLAS_DEFAULT_ATOMICS_MODE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python
RUNNER="$AUDIT_SOURCE/platforms/scnet/run_xian_deterministic_training.py"

"$PYTHON" "$RUNNER" --model-source "$MODEL_SOURCE" \
  --benchmark-script "$BENCHMARK" --cache-root "$CACHE" \
  --output-root "$RUN_ROOT"
sha256sum "$RUN_ROOT/metrics.json" "$RUN_ROOT/best_model.pt" \
  "$RUN_ROOT/last_checkpoint.pt" "$RUN_ROOT/deterministic_runtime.json" \
  > "$RUN_ROOT/SHA256SUMS"
