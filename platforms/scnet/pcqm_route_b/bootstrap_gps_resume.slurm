#!/usr/bin/env bash
#SBATCH --job-name=rb-pcqm-bootstrap
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail

: "${ROOT:?submit with ROOT}"
: "${CODE:?submit with CODE}"
: "${RUNTIME_ARCHIVE:?submit with RUNTIME_ARCHIVE}"

home_root="$(realpath "$HOME")"
root_path="$(realpath "$ROOT")"
archive_path="$(realpath "$RUNTIME_ARCHIVE")"
case "$root_path" in
  "/work1/share/$USER"/*) ;;
  *) echo "ROOT escapes user share: $root_path" >&2; exit 1 ;;
esac
case "$archive_path" in
  "$root_path"/*) ;;
  *) echo "runtime archive escapes ROOT: $archive_path" >&2; exit 1 ;;
esac

mkdir -p "$CODE" "$home_root/molgap/.scnet/site" "$root_path/preflight"
code_path="$(realpath "$CODE")"
case "$code_path" in
  "$home_root"/*) ;;
  *) echo "CODE escapes user home: $code_path" >&2; exit 1 ;;
esac

tar -xzf "$archive_path" -C "$code_path"

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$code_path/src:$home_root/molgap/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
cd "$root_path"

python - <<'PY'
import json
from pathlib import Path

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

device = torch.device("cuda")
x = torch.tensor([[1.0], [3.0]], device=device)
batch = torch.zeros(2, dtype=torch.long, device=device)
pooled = global_mean_pool(x, batch)
if not torch.isfinite(pooled).all() or pooled.item() != 2.0:
    raise RuntimeError(f"invalid DCU PyG pooling result: {pooled}")

report = {
    "status": "accepted",
    "torch": torch.__version__,
    "torch_geometric": torch_geometric.__version__,
    "device": torch.cuda.get_device_name(0),
    "pooling": pooled.item(),
}
path = Path("preflight/runtime_environment.json")
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(report, sort_keys=True))
PY
