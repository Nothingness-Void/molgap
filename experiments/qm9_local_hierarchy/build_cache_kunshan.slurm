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
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$CODE/src"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python
mkdir -p "$ROOT/logs" "$ROOT/cache"
"$PYTHON" -m molgap.qm9_local_hierarchy build-cache \
  --output-root "$ROOT/cache" \
  --source-commit "$SOURCE_COMMIT"
