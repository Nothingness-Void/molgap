#!/usr/bin/env bash
#SBATCH --job-name=rb-pcqm-gps-preflight
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail

: "${ENCODER_NAME:?submit with ENCODER_NAME=gps9 or gps11_160}"
: "${ROOT:?submit with ROOT}"
: "${CODE:?submit with CODE}"
cd "$ROOT"
test "$(realpath "$ROOT")" = "$ROOT"
test "$(realpath "$CODE")" = "$CODE"
module purge
module load compiler/dtk/25.04
module load sghpcdas/25.6
source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate pytorch-python3.10
export PYTHONPATH="$CODE/src:$HOME/molgap/.scnet/site${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python "$CODE/platforms/scnet/pcqm_route_b/run_tuned_gps.py" \
  --root "$ROOT" --encoder "$ENCODER_NAME" --preflight

ROOT="$ROOT" CODE="$CODE" ENCODER_NAME="$ENCODER_NAME" python - <<'PY'
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

root = Path(os.environ["ROOT"])
code = Path(os.environ["CODE"])
encoder = os.environ["ENCODER_NAME"]
sys.path.insert(0, str(code / "platforms" / "scnet" / "pcqm_route_b"))

from run_tuned_gps import WARM_STARTS, _configuration
from molgap.pcqm_route_b_training import make_model

config, overrides = _configuration(root, encoder)
checkpoint_path = root / "outputs_tuned" / encoder / "last.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint.get("format") != "molgap-pcqm-route-b-encoder-checkpoint-v1":
    raise RuntimeError("unexpected resume checkpoint format")
if checkpoint.get("config") != asdict(config):
    raise RuntimeError("Route B resume config changed")
if checkpoint.get("training_overrides") != asdict(overrides):
    raise RuntimeError("Route B resume tuning overrides changed")
required = {
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "best_state",
    "rng_state",
    "input_reports",
}
missing = sorted(required - checkpoint.keys())
if missing:
    raise RuntimeError(f"resume checkpoint misses keys: {missing}")

device = torch.device("cuda")
model, _, _, _ = make_model(
    config,
    root / "warmstarts" / WARM_STARTS[encoder],
    device,
    overrides,
)
model.load_state_dict(checkpoint["model"], strict=True)
model.load_state_dict(checkpoint["best_state"], strict=True)

digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
report = {
    "status": "accepted",
    "encoder": encoder,
    "checkpoint_sha256": digest,
    "checkpoint_bytes": checkpoint_path.stat().st_size,
    "resume_epoch": int(checkpoint["epoch"]) + 1,
    "max_epochs": int(config.max_epochs),
    "best_epoch": int(checkpoint["best_epoch"]),
    "best_dev_gap_mae_eV": float(checkpoint["best_dev_gap_mae_eV"]),
    "config_identity": True,
    "training_overrides_identity": True,
    "model_state_strict": True,
    "best_state_strict": True,
}
path = root / "preflight" / f"{encoder}_resume_contract.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(report, sort_keys=True))
PY
