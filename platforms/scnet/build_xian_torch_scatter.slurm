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
: "${GLOG_TARBALL:?}"
for path in "$ROOT" "$TARGET_SITE" "$SOURCE_TARBALL" "$GFLAGS_TARBALL" "$GLOG_TARBALL"; do
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
# torch-scatter 2.0.9 checks CUDA_HOME only; the vendor HIP build exposes
# ROCM_HOME instead, so request the CUDA-named extension path explicitly.
export FORCE_CUDA=1
# The DTK 22.10 Torch fork omits the upstream compile-time HIP version macro.
# Its reported HIP runtime is 5.2, whose upstream numeric encoding is 502.
export CFLAGS="${CFLAGS:-} -DTORCH_HIP_VERSION=502"
export CXXFLAGS="${CXXFLAGS:-} -DTORCH_HIP_VERSION=502"

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

GLOG_PREFIX="$ROOT/glog-0.3.5-install"
if [[ ! -f "$GLOG_PREFIX/include/glog/stl_logging.h" ]]; then
  GLOG_BUILD="$ROOT/glog-build-${SLURM_JOB_ID}"
  mkdir -p "$GLOG_BUILD"
  tar -xzf "$GLOG_TARBALL" -C "$GLOG_BUILD"
  cd "$GLOG_BUILD/glog-0.3.5"
  CPPFLAGS="-I$GFLAGS_PREFIX/include" \
  LDFLAGS="-L$GFLAGS_PREFIX/lib" \
    ./configure --prefix="$GLOG_PREFIX" --with-gflags="$GFLAGS_PREFIX"
  make -j "${SLURM_CPUS_PER_TASK:-8}"
  make install
  cd "$ROOT"
fi

export CPLUS_INCLUDE_PATH="$GLOG_PREFIX/include:$GFLAGS_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$GLOG_PREFIX/lib:$GFLAGS_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$GLOG_PREFIX/lib:$GFLAGS_PREFIX/lib:${LD_LIBRARY_PATH:-}"

SCATTER_BUILD="$ROOT/torch-scatter-build-${SLURM_JOB_ID}"
mkdir -p "$SCATTER_BUILD"
tar -xzf "$SOURCE_TARBALL" -C "$SCATTER_BUILD"
SCATTER_SOURCE="$SCATTER_BUILD/torch_scatter-2.0.9"
"$PYTHON" - "$SCATTER_SOURCE/setup.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "nvcc_flags += ['--expt-relaxed-constexpr', '-O2']"
new = "nvcc_flags += ([] if torch.version.hip else ['--expt-relaxed-constexpr']) + ['-O2']"
if old not in text:
    raise SystemExit("expected torch-scatter compiler flag line was not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
"$PYTHON" - "$SCATTER_SOURCE/csrc/cuda/utils.cuh" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
replacements = {
    "return __shfl_up_sync(mask, (__half)var, delta);":
        "return at::Half(__shfl_up((float)var, delta));",
    "return __shfl_down_sync(mask, (__half)var, delta);":
        "return at::Half(__shfl_down((float)var, delta));",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"expected torch-scatter half shuffle line was not found: {old}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
PY
"$PYTHON" - "$SCATTER_SOURCE/csrc/cuda/atomics.cuh" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || CUDA_VERSION < 10000)"
new = "#if defined(__HIP_PLATFORM_HCC__) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || CUDA_VERSION < 10000))"
if old not in text:
    raise SystemExit("expected torch-scatter half atomic guard was not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY

"$PYTHON" -m pip install \
  --no-deps \
  --no-build-isolation \
  --target "$TARGET_SITE" \
  --no-cache-dir \
  "$SCATTER_SOURCE"

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
