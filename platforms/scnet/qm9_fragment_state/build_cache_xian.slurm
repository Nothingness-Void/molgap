#!/usr/bin/env bash
# Optional CPU builder for a future account rotation.  The current cache is
# built locally from the pinned QM9 files, then independently accepted here.
#SBATCH --job-name=qm9-frag-cache-build
#SBATCH --partition=xahdtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${CODE:?submit with CODE}"
: "${ROOT:?submit with ROOT}"
: "${SOURCE_ROOT:?submit with SOURCE_ROOT containing qm9 cache files}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"
: "${PYTHON:?submit with PYTHON}"

for path in "$CODE" "$ROOT" "$SOURCE_ROOT"; do
  case "$(realpath -m "$path")" in
    "$HOME"/*) ;;
    *) echo "path outside HOME: $path" >&2; exit 2 ;;
  esac
done
test -x "$PYTHON"

module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
EXTRA_SITE="$HOME/molgap-envs/xian-card2-torch110-pyg24/lib/python3.9/site-packages"
RUNTIME_ROOT="$HOME/molgap-results/xian-card2-compatibility-20260905"
test -d "$EXTRA_SITE"
export PYTHONPATH="$CODE/src:$EXTRA_SITE${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RUNTIME_ROOT/glog-0.3.5-install/lib:$RUNTIME_ROOT/gflags-2.1.2-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
mkdir -p "$ROOT/logs"

"$PYTHON" "$CODE/experiments/qm9_architecture/qm9_fragment_state.py" \
  build-cache --source-root "$SOURCE_ROOT" --output-root "$ROOT" \
  --source-commit "$SOURCE_COMMIT"
"$PYTHON" "$CODE/experiments/qm9_architecture/qm9_fragment_state.py" \
  accept-cache --cache-root "$ROOT" --source-commit "$SOURCE_COMMIT"
