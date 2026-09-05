#!/usr/bin/env bash
#SBATCH --job-name=molgap-kunshan-vector-s42
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --time=12:00:00
set -euo pipefail
: "${ROOT:?}"
: "${CODE:?}"
: "${CACHE:?}"
: "${SOURCE_COMMIT:?}"
for path in "$ROOT" "$CODE" "$CACHE"; do
  case "$(realpath "$path")" in /public/home/scnaqkfcy3/*) ;; *) exit 2 ;; esac
done
cd "$ROOT"
module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$CODE/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python
"$PYTHON" "$CODE/tests/remote_pcqm_vector_state.py" --output "$ROOT/preflight.json"
"$PYTHON" -m molgap.pcqm_kunshan_screen \
  --cache-root "$CACHE" --output-root "$ROOT/output" \
  --preflight "$ROOT/preflight.json" --source-commit "$SOURCE_COMMIT"
