"""Build the Drive-backed conservative 2D/3D Fusion notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WHEEL_NAME = "molgap-0.1.0-py3-none-any.whl"
INPUT_STAGING = (
    ROOT.parents[2]
    / "platforms"
    / "_records"
    / "colab"
    / "staging"
    / "conservative_2d3d_fusion_r1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


wheel = ROOT / "wheels" / WHEEL_NAME
if not wheel.is_file():
    raise FileNotFoundError(f"Build {WHEEL_NAME} before generating the notebook")
wheel_sha256 = _sha256(wheel)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MolGap conservative 2D + 3D Fusion\n",
                "Drive-backed six-head screen with exact 2D fallback and fixed "
                "OOD/P8-hard acceptance.\n",
            ],
        },
        _cell(
            f"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib
import subprocess
import sys

ROOT = Path('/content/drive/MyDrive/MolGap')
NOTEBOOK_DIR = ROOT / 'notebooks' / 'molgap_conservative_2d3d_fusion_r1'
INPUT_DIR = ROOT / 'results' / 'molgap_conservative_2d3d_fusion_r1' / 'inputs'
CHECKPOINT_DIR = ROOT / 'checkpoints' / 'molgap_conservative_2d3d_fusion_r1'
RESULTS_DIR = ROOT / 'results' / 'molgap_conservative_2d3d_fusion_r1' / 'run'
WHEEL = NOTEBOOK_DIR / '{WHEEL_NAME}'
EXPECTED_WHEEL_SHA256 = '{wheel_sha256}'

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

if not WHEEL.is_file():
    raise FileNotFoundError(WHEEL)
if sha256_file(WHEEL) != EXPECTED_WHEEL_SHA256:
    raise RuntimeError('The uploaded wheel differs from the notebook contract')

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet', '--force-reinstall', '--no-deps', str(WHEEL)],
    check=True,
)

import torch
if not torch.cuda.is_available():
    raise RuntimeError('Select an A100 GPU runtime before continuing')
GPU_NAME = torch.cuda.get_device_name(0)
if 'A100' not in GPU_NAME:
    raise RuntimeError(f'A100 required by this run contract; active GPU is {{GPU_NAME}}')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print('GPU:', GPU_NAME)
print('checkpoints:', CHECKPOINT_DIR)
print('results:', RESULTS_DIR)
"""
        ),
        _cell(
            """
import json

TRAINING_PAYLOAD = INPUT_DIR / 'training_payload.pt'
TRAINING_MANIFEST = INPUT_DIR / 'training_manifest.json'
EXTERNAL_PAYLOAD = INPUT_DIR / 'external_payload.pt'
EXTERNAL_MANIFEST = INPUT_DIR / 'external_manifest.json'

def accept_payload(payload_path, manifest_path):
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get('payload', {}).get('sha256')
    observed = sha256_file(payload_path)
    if manifest.get('status') != 'accepted' or observed != expected:
        raise RuntimeError(f'Payload acceptance failed: {payload_path.name}')
    return manifest

training_manifest = accept_payload(TRAINING_PAYLOAD, TRAINING_MANIFEST)
external_manifest = accept_payload(EXTERNAL_PAYLOAD, EXTERNAL_MANIFEST)
if training_manifest['context_dim'] != external_manifest['context_dim']:
    raise RuntimeError('Training and external context dimensions differ')
if external_manifest['scope_rows'] != {
    'all': 1973,
    'ood1000': 998,
    'p8_targeted_hard': 975,
}:
    raise RuntimeError('Fixed external evaluation identity differs')
print('training rows:', training_manifest['rows'])
print('external rows:', external_manifest['scope_rows'])
print('context dim:', training_manifest['context_dim'])
print('sealed 20K: not mounted and not used')
"""
        ),
        _cell(
            """
from molgap.conservative_fusion_runner import run_conservative_fusion

result = run_conservative_fusion(
    training_payload_path=TRAINING_PAYLOAD,
    external_payload_path=EXTERNAL_PAYLOAD,
    checkpoint_dir=CHECKPOINT_DIR,
    results_dir=RESULTS_DIR,
    device='cuda',
)
print('decision:', result['status'])
print('elapsed hours:', result['elapsed_s'] / 3600)
print('promotion gate:', json.dumps(result['promotion_gate'], indent=2))
print('completion:', RESULTS_DIR / 'completion_manifest.json')
"""
        ),
        _cell(
            """
summary = json.loads((RESULTS_DIR / 'metrics.json').read_text())
print('decision:', summary['status'])
for base, gate in summary['promotion_gate'].items():
    print(
        base,
        'all delta=', f"{gate['all_average_delta_eV']:+.6f} eV",
        'P8-hard delta=', f"{gate['p8_hard_average_delta_eV']:+.6f} eV",
        'PASS' if gate['passed'] else 'REJECT',
    )
print('No production registry was changed.')
"""
        ),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "molgap_conservative_2d3d_fusion.ipynb"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebook_path = ROOT / "molgap_conservative_2d3d_fusion.ipynb"
notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
manifest = {
    "format": "molgap-conservative-fusion-colab-package-v1",
    "wheel": {"name": wheel.name, "sha256": wheel_sha256},
    "notebook": {
        "name": notebook_path.name,
        "sha256": _sha256(notebook_path),
    },
    "drive_paths": {
        "notebook_dir": "MolGap/notebooks/molgap_conservative_2d3d_fusion_r1",
        "input_dir": "MolGap/results/molgap_conservative_2d3d_fusion_r1/inputs",
        "checkpoint_dir": "MolGap/checkpoints/molgap_conservative_2d3d_fusion_r1",
        "result_dir": "MolGap/results/molgap_conservative_2d3d_fusion_r1/run",
    },
    "inputs": {
        name: {
            "bytes": (INPUT_STAGING / name).stat().st_size,
            "sha256": _sha256(INPUT_STAGING / name),
        }
        for name in (
            "training_payload.pt",
            "training_manifest.json",
            "external_payload.pt",
            "external_manifest.json",
        )
    },
}
(ROOT / "package_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
