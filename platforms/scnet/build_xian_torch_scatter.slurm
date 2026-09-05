#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-scatter-build
#SBATCH --partition=xahdtest
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

: "${ROOT:?}"
: "${TARGET_SITE:?}"
: "${SOURCE_TARBALL:?}"
: "${GFLAGS_TARBALL:?}"
for path in "$ROOT" "$TARGET_SITE" "$SOURCE_TARBALL" "$GFLAGS_TARBALL"; do
  case "$(realpath "$path")" in
    /work/home/acf9jvb3sm/*) ;;
    *) exit 2 ;;
  esac
done

cd "$ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1

export PYTHONNOUSERSITE=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx906}"

PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python
GFLAGS_PREFIX="$ROOT/gflags-2.1.2-install"
if [[ ! -f "$GFLAGS_PREFIX/include/gflags/gflags.h" ]]; then
  GFLAGS_BUILD="$ROOT/gflags-build-${SLURM_JOB_ID}"
  mkdir -p "$GFLAGS_BUILD"
  tar -xzf "$GFLAGS_TARBALL" -C "$GFLAGS_BUILD"
  mkdir -p "$GFLAGS_BUILD/build"
  cd "$GFLAGS_BUILD/build"
  cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX="$GFLAGS_PREFIX" \
    ../gflags-2.1.2
  make -j "${SLURM_CPUS_PER_TASK:-8}"
  make install
  cd "$ROOT"
fi

export CPLUS_INCLUDE_PATH="$GFLAGS_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$GFLAGS_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$GFLAGS_PREFIX/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" -m pip install \
  --no-deps \
  --no-build-isolation \
  --target "$TARGET_SITE" \
  --no-cache-dir \
  "$SOURCE_TARBALL"

PYTHONPATH="$TARGET_SITE:${PYTHONPATH:-}" "$PYTHON" - <<'PY'
import json
import pathlib
import torch
import torch_scatter

payload = {
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "torch_scatter": torch_scatter.__version__,
    "pytorch_rocm_arch": "gfx906",
}
path = pathlib.Path("torch_scatter_build.json")
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path.read_text(encoding="utf-8"))
PY
