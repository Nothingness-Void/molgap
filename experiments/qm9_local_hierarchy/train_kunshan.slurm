#!/usr/bin/env bash
#SBATCH --job-name=molgap-qm9-local-hierarchy
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python
MANIFEST="$ROOT/cache/manifest.json"
test -s "$ROOT/cache/acceptance.json"
test -s "$ROOT/preflight/preflight.json"
CACHE_SHA="$($PYTHON -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["complete"] and not p["test_role_read"] and not p["model_inference_executed"]; print(p["aggregate_sha256"])' "$MANIFEST")"
mkdir -p "$ROOT/output"
"$PYTHON" -m molgap.qm9_local_hierarchy train \
  --cache-root "$ROOT/cache" \
  --output-root "$ROOT/output" \
  --source-commit "$SOURCE_COMMIT" \
  --cache-sha256 "$CACHE_SHA" \
  --preflight "$ROOT/preflight/preflight.json"
