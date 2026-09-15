#!/usr/bin/env bash
# Build the account-local Kunshan DTK 23.10 runtime without using accelerator time.
#SBATCH --job-name=molgap-v4-env
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

: "${BOOTSTRAP_ROOT:?submit with BOOTSTRAP_ROOT}"
: "${RUNTIME_ROOT:?submit with RUNTIME_ROOT}"
: "${WHEELHOUSE:?submit with WHEELHOUSE}"

BOOTSTRAP_ROOT="$(realpath "$BOOTSTRAP_ROOT")"
ACCOUNT_HOME="$(realpath "${ACCOUNT_HOME:-$HOME}")"
mkdir -p "$RUNTIME_ROOT"
RUNTIME_ROOT="$(realpath "$RUNTIME_ROOT")"
WHEELHOUSE="$(realpath "$WHEELHOUSE")"
for path in "$BOOTSTRAP_ROOT" "$RUNTIME_ROOT" "$WHEELHOUSE"; do
  case "$path" in
    "$ACCOUNT_HOME"/*) ;;
    *) echo "refusing path outside account home: $path" >&2; exit 2 ;;
  esac
done

test -f "$BOOTSTRAP_ROOT/BOOTSTRAP_MANIFEST.json"
test -f "$BOOTSTRAP_ROOT/deployment/integrity.json"
test -f "$BOOTSTRAP_ROOT/input/pcqm4mv2-ogb-fixed-100k-v1/manifest.json"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export CONDARC=/dev/null
export CONDA_PKGS_DIRS="$RUNTIME_ROOT/conda-pkgs"
export TMPDIR="$RUNTIME_ROOT/tmp-${SLURM_JOB_ID}"
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR" "$RUNTIME_ROOT/logs"

INSTALLER="$RUNTIME_ROOT/Miniconda3-py310_25.7.0-2-Linux-x86_64.sh"
MINICONDA="$RUNTIME_ROOT/miniconda3-py310-25.7.0-2"
test -f "$INSTALLER"
test -d "$WHEELHOUSE"
if [[ ! -x "$MINICONDA/bin/python" ]]; then
  bash "$INSTALLER" -b -p "$MINICONDA"
fi

PYTHON="$MINICONDA/bin/python"
"$PYTHON" -m pip install --no-index --find-links "$WHEELHOUSE" --upgrade \
  "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"

TORCH_WHEEL="/public/software/apps/DeepLearning/whl/dtk-23.10/pytorch/torch1.13/py310/torch-1.13.1+git7d2dd01.abi0.dtk2310-cp310-cp310-manylinux2014_x86_64.whl"
test -r "$TORCH_WHEEL"
"$PYTHON" -m pip install --no-deps "$TORCH_WHEEL"
"$PYTHON" -m pip install --no-index --find-links "$WHEELHOUSE" \
  "numpy==1.26.4" \
  "scipy==1.15.3" \
  "pandas==2.3.2" \
  "scikit-learn==1.7.1" \
  "networkx==3.4.2" \
  "tqdm==4.67.1" \
  "psutil==7.0.0" \
  "packaging==25.0" \
  "typing-extensions==4.15.0" \
  "requests==2.32.5" \
  "littleutils==0.2.4" \
  "outdated==0.2.2"
"$PYTHON" -m pip install --no-index --find-links "$WHEELHOUSE" \
  "ogb==1.3.6" \
  "torch-geometric==2.5.3"

export PYTHONPATH="$BOOTSTRAP_ROOT/code/src"
cd "$BOOTSTRAP_ROOT"
"$PYTHON" - <<'PY'
import json
import pathlib
import platform
import sys

import networkx
import numpy
import ogb
import pandas
import scipy
import sklearn
import torch
import torch_geometric

assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == "1.26.4"
assert torch.__version__ == "1.13.1+git7d2dd01.abi0.dtk2310"
assert torch_geometric.__version__ == "2.5.3"

payload = {
    "format": "molgap-kunshan-v4-environment-v1",
    "accepted_cpu_imports": True,
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pandas": pandas.__version__,
    "scikit_learn": sklearn.__version__,
    "networkx": networkx.__version__,
    "ogb": ogb.__version__,
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "torch_geometric": torch_geometric.__version__,
    "official_validation_role_read": False,
    "test_dev_role_read": False,
    "training_executed": False,
}
root = pathlib.Path.cwd()
target = root / "environment_cpu.json"
temporary = target.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(payload, sort_keys=True), flush=True)
PY

"$PYTHON" -m pip check

"$PYTHON" -m pip freeze > "$RUNTIME_ROOT/environment.freeze.txt.partial"
mv "$RUNTIME_ROOT/environment.freeze.txt.partial" "$RUNTIME_ROOT/environment.freeze.txt"
sha256sum "$INSTALLER" "$TORCH_WHEEL" "$RUNTIME_ROOT/environment.freeze.txt" \
  > "$RUNTIME_ROOT/environment.sha256.partial"
mv "$RUNTIME_ROOT/environment.sha256.partial" "$RUNTIME_ROOT/environment.sha256"

mv "$BOOTSTRAP_ROOT/environment_cpu.json" \
  "$RUNTIME_ROOT/environment_cpu.json.partial"
mv "$RUNTIME_ROOT/environment_cpu.json.partial" \
  "$RUNTIME_ROOT/environment_cpu.json"
echo "PYTHON_BIN=$PYTHON"
echo "KUNSHAN_V4_ENV_READY_CPU=1"
