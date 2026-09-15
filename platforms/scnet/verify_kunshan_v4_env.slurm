#!/usr/bin/env bash
# Verify the frozen environment on one real Kunshan DCU; this is not training.
#SBATCH --job-name=molgap-v4-env-check
#SBATCH --partition=kshdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

: "${BOOTSTRAP_ROOT:?submit with BOOTSTRAP_ROOT}"
: "${RUNTIME_ROOT:?submit with RUNTIME_ROOT}"
: "${PYTHON_BIN:?submit with PYTHON_BIN}"

BOOTSTRAP_ROOT="$(realpath "$BOOTSTRAP_ROOT")"
RUNTIME_ROOT="$(realpath "$RUNTIME_ROOT")"
PYTHON_BIN="$(realpath "$PYTHON_BIN")"
ACCOUNT_HOME="$(realpath "${ACCOUNT_HOME:-$HOME}")"
for path in "$BOOTSTRAP_ROOT" "$RUNTIME_ROOT" "$PYTHON_BIN"; do
  case "$path" in
    "$ACCOUNT_HOME"/*) ;;
    *) echo "refusing path outside account home: $path" >&2; exit 2 ;;
  esac
done

test -f "$RUNTIME_ROOT/environment_cpu.json"
test -x "$PYTHON_BIN"

module purge
module load compiler/dtk/23.10
module load mpi/openmpi/gcc-9.3.0/4.1.5
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BOOTSTRAP_ROOT/code/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$RUNTIME_ROOT"

"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import random
import sys

import numpy as np
import torch
import torch_geometric

assert sys.version_info[:2] == (3, 10)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.version.hip is not None


def run_once():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 1),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(314159)
    x = torch.randn((128, 32), generator=generator).cuda()
    y = torch.randn((128, 1), generator=generator).cuda()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.l1_loss(model(x), y)
    loss.backward()
    optimizer.step()
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return float(loss.detach().cpu()), digest.hexdigest()


first = run_once()
second = run_once()
if first != second:
    raise RuntimeError(f"deterministic optimizer-step mismatch: {first} != {second}")

payload = {
    "format": "molgap-kunshan-v4-dcu-runtime-v1",
    "accepted": True,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "torch_geometric": torch_geometric.__version__,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "physical_batch_per_device": 128,
    "precision": "fp32",
    "tf32": False,
    "deterministic_algorithms": True,
    "optimizer_step_loss": first[0],
    "optimizer_step_state_sha256": first[1],
    "official_validation_role_read": False,
    "test_dev_role_read": False,
    "training_executed": False,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
target = pathlib.Path("environment_dcu.json")
temporary = target.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(payload, sort_keys=True), flush=True)
PY

sha256sum "$RUNTIME_ROOT/environment_dcu.json" \
  > "$RUNTIME_ROOT/environment_dcu.sha256.partial"
mv "$RUNTIME_ROOT/environment_dcu.sha256.partial" \
  "$RUNTIME_ROOT/environment_dcu.sha256"
echo "KUNSHAN_V4_ENV_ACCEPTED=1"
