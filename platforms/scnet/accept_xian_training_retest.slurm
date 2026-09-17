#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-accept
#SBATCH --partition=xahdtest
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
: "${ROOT:?submit with ROOT below /work/home/acf9jvb3sm}"
: "${AUDIT_SOURCE:?submit with AUDIT_SOURCE below /work/home/acf9jvb3sm}"
: "${RUN1:?submit with RUN1 below /work/home/acf9jvb3sm}"
: "${RUN2:?submit with RUN2 below /work/home/acf9jvb3sm}"

for path in "$ROOT" "$AUDIT_SOURCE" "$RUN1" "$RUN2"; do
  case "$(realpath "$path")" in
    /work/home/acf9jvb3sm/*) ;;
    *) echo "path outside authorized boundary: $path" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROOT"
cd "$ROOT"
module purge
module load Pytorch/1.10.0-hpcx-gcc-7.3.1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$AUDIT_SOURCE/src:${PYTHONPATH:-}"
PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python
"$PYTHON" "$AUDIT_SOURCE/platforms/scnet/audit_xian_reductions.py" \
  training-accept --first-root "$RUN1" --second-root "$RUN2" \
  --output "$ROOT/acceptance.json"
sha256sum "$ROOT/acceptance.json" > "$ROOT/SHA256SUMS"
