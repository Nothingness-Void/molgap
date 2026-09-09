#!/usr/bin/env bash
#SBATCH --job-name=pcqm-localgeo-pf
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --time=00:30:00
set -euo pipefail
: "${CODE:?submit with CODE}"
: "${GRAPH_ROOT:?submit with GRAPH_ROOT}"
: "${OUTPUT_ROOT:?submit with OUTPUT_ROOT}"
: "${SOURCE_COMMIT:?submit with SOURCE_COMMIT}"
: "${PYTHON:?submit with accepted DTK Python}"

for path in "$CODE" "$GRAPH_ROOT" "$OUTPUT_ROOT"; do
  case "$(realpath -m "$path")" in "$HOME"/*) ;; *) exit 2 ;; esac
done
test -s "$GRAPH_ROOT/manifest.json"

module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
EXTRA_SITE="$HOME/molgap-envs/xian-card2-torch110-pyg24/lib/python3.9/site-packages"
RUNTIME_ROOT="$HOME/molgap-results/xian-card2-compatibility-20260905"
test -d "$EXTRA_SITE"
export PYTHONPATH="$CODE/src:$EXTRA_SITE${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RUNTIME_ROOT/glog-0.3.5-install/lib:$RUNTIME_ROOT/gflags-2.1.2-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MOLGAP_EXECUTION_PLATFORM=scnet-xian-dtk2210
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$OUTPUT_ROOT"
"$PYTHON" "$CODE/platforms/scnet/run_pcqm_local_geometry_pretraining.py" preflight \
  --graph-root "$GRAPH_ROOT" --output-root "$OUTPUT_ROOT" \
  --source-commit "$SOURCE_COMMIT"
