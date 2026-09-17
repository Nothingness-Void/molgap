#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-model-det
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
: "${ROOT:?submit with ROOT below /work/home/acf9jvb3sm}"
: "${AUDIT_SOURCE:?submit with AUDIT_SOURCE below /work/home/acf9jvb3sm}"
: "${MODEL_SOURCE:?submit with MODEL_SOURCE below /work/home/acf9jvb3sm}"
: "${CACHE:?submit with CACHE below /work/home/acf9jvb3sm}"

for path in "$ROOT" "$AUDIT_SOURCE" "$MODEL_SOURCE" "$CACHE"; do
  case "$(realpath "$path")" in
    /work/home/acf9jvb3sm/*) ;;
    *) echo "path outside authorized boundary: $path" >&2; exit 2 ;;
  esac
done

test -f "$MODEL_SOURCE/src/molgap/pcqm_gap_architecture.py"
test -f "$CACHE/manifest.json"
mkdir -p "$ROOT/model-replay" "$ROOT/logs"
cd "$ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1

export PYTHONNOUSERSITE=1
PLATFORM_PYTHONPATH="${PYTHONPATH:-}"
EXTRA_SITE=/work/home/acf9jvb3sm/molgap-envs/xian-card2-torch110-pyg24/lib/python3.9/site-packages
export PYTHONPATH="$AUDIT_SOURCE/src:$EXTRA_SITE:$PLATFORM_PYTHONPATH"
RUNTIME_ROOT=/work/home/acf9jvb3sm/molgap-results/xian-card2-compatibility-20260905
export LD_LIBRARY_PATH="$RUNTIME_ROOT/glog-0.3.5-install/lib:$RUNTIME_ROOT/gflags-2.1.2-install/lib:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export ROCBLAS_DEFAULT_ATOMICS_MODE=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python
CLI="$AUDIT_SOURCE/platforms/scnet/audit_xian_reductions.py"

"$PYTHON" "$CLI" model-audit \
  --model-source "$MODEL_SOURCE" --cache-root "$CACHE" \
  --output "$ROOT/model-replay/first.json" --repetitions 3 --batch-size 128
"$PYTHON" "$CLI" model-audit \
  --model-source "$MODEL_SOURCE" --cache-root "$CACHE" \
  --output "$ROOT/model-replay/second.json" --repetitions 3 --batch-size 128
"$PYTHON" "$CLI" model-accept \
  --first "$ROOT/model-replay/first.json" \
  --second "$ROOT/model-replay/second.json" \
  --output "$ROOT/model-replay/acceptance.json"
sha256sum "$ROOT/model-replay/"*.json > "$ROOT/model-replay/SHA256SUMS"
