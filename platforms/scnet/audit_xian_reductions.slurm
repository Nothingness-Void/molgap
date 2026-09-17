#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-det
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
: "${ROOT:?submit with ROOT below /work/home/acf9jvb3sm}"
: "${SOURCE:?submit with SOURCE below /work/home/acf9jvb3sm}"

for path in "$ROOT" "$SOURCE"; do
  case "$(realpath "$path")" in
    /work/home/acf9jvb3sm/*) ;;
    *) echo "path outside authorized boundary: $path" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROOT/output" "$ROOT/logs"
cd "$ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1

export PYTHONNOUSERSITE=1
PLATFORM_PYTHONPATH="${PYTHONPATH:-}"
EXTRA_SITE=/work/home/acf9jvb3sm/molgap-envs/xian-card2-torch110-pyg24/lib/python3.9/site-packages
export PYTHONPATH="$SOURCE/src:$EXTRA_SITE:$PLATFORM_PYTHONPATH"
RUNTIME_ROOT=/work/home/acf9jvb3sm/molgap-results/xian-card2-compatibility-20260905
export LD_LIBRARY_PATH="$RUNTIME_ROOT/glog-0.3.5-install/lib:$RUNTIME_ROOT/gflags-2.1.2-install/lib:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python
CLI="$SOURCE/platforms/scnet/audit_xian_reductions.py"

"$PYTHON" "$CLI" audit --output "$ROOT/output/default.json" --repetitions 12
"$PYTHON" "$CLI" audit --strict --output "$ROOT/output/strict.json" --repetitions 12
"$PYTHON" "$CLI" audit --strict --output "$ROOT/output/strict-repeat.json" --repetitions 12
"$PYTHON" "$CLI" accept \
  --default "$ROOT/output/default.json" \
  --strict "$ROOT/output/strict.json" \
  --strict-repeat "$ROOT/output/strict-repeat.json" \
  --output "$ROOT/output/acceptance.json"

sha256sum "$ROOT/output/default.json" "$ROOT/output/strict.json" \
  "$ROOT/output/strict-repeat.json" \
  "$ROOT/output/acceptance.json" > "$ROOT/output/SHA256SUMS"
