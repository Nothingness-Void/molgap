#!/bin/bash
# Install the minimum x86 CPU environment needed to build ETKDG PyG graph caches.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/molgap-3d-env/bin/python}"
TORCH_CPU_WHEEL="https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp38-cp38-linux_x86_64.whl"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --no-cache-dir "$TORCH_CPU_WHEEL"
"$PYTHON_BIN" -m pip install --no-cache-dir \
  "torch-geometric==2.5.3" \
  "rdkit==2023.9.6" \
  "pandas==2.0.3" \
  "tqdm==4.66.5"

"$PYTHON_BIN" - <<'PY'
import torch
import torch_geometric
from rdkit import rdBase

print("torch", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("rdkit", rdBase.rdkitVersion)
PY
