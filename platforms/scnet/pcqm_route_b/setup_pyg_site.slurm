#!/usr/bin/env bash
#SBATCH --job-name=rb-pcqm-pyg
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail

: "${SITE:?submit with SITE}"
: "${PYG_ARCHIVE:?submit with PYG_ARCHIVE}"
: "${EXPECTED_SHA:?submit with EXPECTED_SHA}"

home_root="$(realpath "$HOME")"
mkdir -p "$SITE"
site_path="$(realpath "$SITE")"
archive_path="$(realpath "$PYG_ARCHIVE")"
case "$site_path" in
  "$home_root"/*) ;;
  *) echo "SITE escapes user home: $site_path" >&2; exit 1 ;;
esac
case "$archive_path" in
  "$home_root"/*) ;;
  *) echo "PYG_ARCHIVE escapes user home: $archive_path" >&2; exit 1 ;;
esac

actual_sha="$(sha256sum "$archive_path" | awk '{print $1}')"
test "$actual_sha" = "$EXPECTED_SHA"
tar -tf "$archive_path" | awk \
  '/^(\/|\.\.\/)|\/\.\.\// {bad=1; print} END{exit bad}'

module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10

# This archive contains only the pure-Python PyG package and dist-info copied
# from the accepted prior SCNet account. Platform DTK Torch remains untouched.
tar -xzf "$archive_path" -C "$site_path"

PYTHONPATH="$site_path${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import json
from pathlib import Path

import torch
import torch_geometric

if torch_geometric.__version__ != "2.7.0":
    raise RuntimeError(torch_geometric.__version__)
report = {
    "status": "installed",
    "torch": torch.__version__,
    "torch_geometric": torch_geometric.__version__,
    "source": "accepted_prior_scnet_account",
}
path = Path.home() / "molgap" / ".scnet" / "pyg_install.json"
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(report, sort_keys=True))
PY
