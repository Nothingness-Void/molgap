#!/usr/bin/env bash
#SBATCH --job-name=molgap-qm9-hierarchy-cache
#SBATCH --partition=kshctest02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
: "${CODE:?}"
: "${ROOT:?}"
: "${SOURCE_COMMIT:?}"
for path in "$CODE" "$ROOT"; do
  case "$(realpath "$path")" in /public/home/scnaqkfcy3/*) ;; *) exit 2 ;; esac
done
module purge
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$ROOT/cpu-deps:$CODE/src"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
PYTHON=/public/home/scnaqkfcy3/molgap-envs/cpu-cache-py311/bin/python
mkdir -p "$ROOT/logs" "$ROOT/cache"
"$PYTHON" -c 'import ogb,rdkit,torch,torch_geometric'
"$PYTHON" -m molgap.qm9_local_hierarchy build-cache \
  --output-root "$ROOT/cache" \
  --source-commit "$SOURCE_COMMIT"
"$PYTHON" "$CODE/experiments/qm9_local_hierarchy/accept_cache.py" \
  --root "$ROOT/cache" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$ROOT/cache/acceptance.json"
