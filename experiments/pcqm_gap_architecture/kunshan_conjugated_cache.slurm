#!/usr/bin/env bash
#SBATCH --job-name=molgap-conjugated-cache
#SBATCH --partition=kshctest02
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${ROOT:?}"
: "${CODE:?}"
: "${INPUT_CACHE:?}"
: "${SOURCE_COMMIT:?}"
for path in "$ROOT" "$CODE" "$INPUT_CACHE"; do
  case "$(realpath "$path")" in /public/home/scnaqkfcy3/*) ;; *) exit 2 ;; esac
done
cd "$ROOT"
module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python
"$PYTHON" "$CODE/experiments/pcqm_gap_architecture/build_kunshan_conjugated_cache.py" \
  --input-root "$INPUT_CACHE" --output-root "$ROOT/cache" \
  --source-commit "$SOURCE_COMMIT"
"$PYTHON" "$CODE/experiments/pcqm_gap_architecture/accept_kunshan_conjugated_cache.py" \
  "$ROOT/cache" --source-commit "$SOURCE_COMMIT" \
  --output "$ROOT/cache_acceptance.json"
