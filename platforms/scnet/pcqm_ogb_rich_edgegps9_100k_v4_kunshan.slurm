#!/bin/bash
#SBATCH --job-name=ogb-edgegps9-100k-v4
#SBATCH --partition=kshdtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=72:00:00
#SBATCH --output=/public/home/scnaqkfcy3/molgap-results/pcqm-ogb-rich-edgegps9-100k-v4-logs/edgegps9-%j.out

set -euo pipefail
ACCOUNT_HOME=/public/home/scnaqkfcy3
ROOT="${MOLGAP_RUN_ROOT:?submit with MOLGAP_RUN_ROOT inside account home}"
ROOT="$(realpath "$ROOT")"
case "$ROOT" in
  "$ACCOUNT_HOME"/*) ;;
  *) echo "refusing run root outside account home: $ROOT" >&2; exit 2 ;;
esac
DATA=/public/home/scnaqkfcy3/molgap-results/pcqm-gptrans-t-100k-v4-c1e2df2/input/pcqm4mv2-ogb-fixed-100k-v1
PYTHON=/public/home/scnaqkfcy3/molgap-envs/graphstate-dtk2310-py310/bin/python
mkdir -p "$ROOT/output"
cd "$ROOT/code"
module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/code/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
SOURCE_ARCHIVE_SHA256="$(tr -d '\r\n' < "$ROOT/SOURCE_ARCHIVE_SHA256.txt")"
"$PYTHON" -u experiments/pcqm_ogb_rich_edgegps9_100k_v4/run.py \
  --dataset-root "$DATA" \
  --manifest "$DATA/manifest.json" \
  --source-archive "$ROOT/source.tar.gz" \
  --source-commit "$(cat "$ROOT/SOURCE_COMMIT.txt")" \
  --source-archive-sha256 "$SOURCE_ARCHIVE_SHA256" \
  --output "$ROOT/output" \
  --platform-id scnet-kunshan
"$PYTHON" -u experiments/pcqm_ogb_rich_edgegps9_100k_v4/accept.py \
  --output "$ROOT/output" | tee "$ROOT/output/acceptance.json"
