#!/usr/bin/env bash
# Train the frozen fresh-control/fragment-state pair after cache acceptance and preflight.
#SBATCH --job-name=qm9-frag-screen
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${CODE:?submit with CODE}"
: "${ROOT:?submit with ROOT}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache}"

for path in "$CODE" "$ROOT" "$CACHE_ROOT"; do
  case "$(realpath -m "$path")" in
    /public/home/scnaqkfcy3/*) ;;
    *) echo "path outside account HOME: $path" >&2; exit 2 ;;
  esac
done
test -s "$CACHE_ROOT/manifest.json"
test -s "$CACHE_ROOT/acceptance.json"
test -s "$ROOT/preflight/preflight.json"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export PYTHONPATH="$CODE/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python
mkdir -p "$ROOT/output" "$ROOT/logs"

CACHE_SHA="$("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["complete"]; print(p["aggregate_sha256"])' "$CACHE_ROOT/manifest.json")"
"$PYTHON" "$CODE/experiments/qm9_architecture/qm9_fragment_state.py" \
  train --cache-root "$CACHE_ROOT" --output-root "$ROOT/output" \
  --source-commit "$SOURCE_COMMIT" --cache-sha256 "$CACHE_SHA" \
  --preflight "$ROOT/preflight/preflight.json"
