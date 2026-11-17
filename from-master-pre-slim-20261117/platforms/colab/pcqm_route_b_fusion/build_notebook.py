"""Build the resumable Colab notebook for the tuned-GPS9 Fusion screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MolGap PCQM Route B Fusion\n",
                "Mount Drive once, validate the immutable payload, and resume the "
                "frozen two-identity, three-seed bounded Fusion screen.\n",
            ],
        },
        _cell(
            """
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import subprocess
import sys
import torch

if not torch.cuda.is_available():
    raise RuntimeError('GPU runtime required: Runtime > Change runtime type > GPU')
print('GPU:', torch.cuda.get_device_name(0))

ROOT = Path('/content/drive/MyDrive/MolGap')
INPUT_DIR = ROOT / 'results' / 'pcqm_route_b_fusion_payload_20260729'
WHEEL = INPUT_DIR / 'molgap-0.1.0-py3-none-any.whl'
RUNNER = INPUT_DIR / 'run_colab_fusion.py'
CHECKPOINT_DIR = ROOT / 'checkpoints' / 'pcqm_route_b_fusion_tuned_gps9_20260729'
RESULT_DIR = ROOT / 'results' / 'pcqm_route_b_fusion_tuned_gps9_20260729'

for path in (WHEEL, RUNNER, INPUT_DIR / 'manifest.json', INPUT_DIR / 'fusion_payload.pt'):
    if not path.exists():
        raise FileNotFoundError(path)
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet', 'torch-geometric==2.7.0'],
    check=True,
)
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet', '--force-reinstall', '--no-deps', str(WHEEL)],
    check=True,
)
print('Drive and runtime ready')
"""
        ),
        _cell(
            """
import subprocess
import sys

command = [
    sys.executable,
    '-u',
    str(RUNNER),
    '--input-dir',
    str(INPUT_DIR),
    '--checkpoint-dir',
    str(CHECKPOINT_DIR),
    '--result-dir',
    str(RESULT_DIR),
]
print(' '.join(command))
subprocess.run(command, check=True)
print((RESULT_DIR / 'colab_completion.json').read_text())
"""
        ),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "molgap_pcqm_route_b_fusion.ipynb"},
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

(ROOT / "molgap_pcqm_route_b_fusion.ipynb").write_text(
    json.dumps(notebook, indent=2) + "\n",
    encoding="utf-8",
)
