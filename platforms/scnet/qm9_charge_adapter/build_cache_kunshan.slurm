#!/usr/bin/env bash
# Build and accept the immutable 30K/3K charge cache on CPU.
#SBATCH --job-name=qm9-charge-cache
#SBATCH --partition=kshctest02
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
: "${SOURCE_ROOT:?submit with SOURCE_ROOT containing pinned QM9 files}"
: "${CPU_DEPS:?submit with CPU_DEPS containing RDKit and OGB}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"

for path in "$CODE" "$ROOT" "$CACHE_ROOT" "$SOURCE_ROOT" "$CPU_DEPS"; do
  case "$(realpath -m "$path")" in
    /public/home/scnaqkfcy3/*) ;;
    *) echo "path outside account HOME: $path" >&2; exit 2 ;;
  esac
done

module purge
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$CPU_DEPS:$CODE/src"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
PYTHON=/public/home/scnaqkfcy3/molgap-envs/cpu-cache-py311/bin/python
mkdir -p "$ROOT/logs" "$CACHE_ROOT"

"$PYTHON" -c 'import molgap.constants,ogb,rdkit,torch,torch_geometric; from rdkit.Chem import rdPartialCharges'
"$PYTHON" "$CODE/experiments/qm9_architecture/qm9_charge_adapter.py" \
  build-cache --source-root "$SOURCE_ROOT" --output-root "$CACHE_ROOT" \
  --source-commit "$SOURCE_COMMIT"
"$PYTHON" "$CODE/experiments/qm9_architecture/qm9_charge_adapter.py" \
  accept-cache --cache-root "$CACHE_ROOT" --source-commit "$SOURCE_COMMIT"
