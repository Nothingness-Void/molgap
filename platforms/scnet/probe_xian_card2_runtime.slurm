#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-card2-probe
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
: "${ROOT:?}"
: "${PROBE:?}"
for path in "$ROOT" "$PROBE"; do
  case "$(realpath "$path")" in /work/home/acf9jvb3sm/*) ;; *) exit 2 ;; esac
done
cd "$ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python "$PROBE" --output "$ROOT/runtime.json"
