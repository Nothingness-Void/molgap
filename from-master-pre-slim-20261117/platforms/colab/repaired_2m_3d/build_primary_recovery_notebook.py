"""Generate the repaired-2M primary SchNet stable-recovery notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "molgap_repaired_2m_primary_stable_recovery.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main() -> None:
    cells = [
        markdown(
            """# MolGap repaired-2M primary SchNet stable recovery

This notebook resumes the accepted repaired-2M primary SchNet from the finite
epoch-7 FP16 best state. Before training, it hashes every graph shard, checks
the software environment, and reproduces the recovery model on validation and
test. Any non-finite batch or evaluation inconsistency aborts the run. Inputs
and all checkpoints live on Drive; Colab is never the only copy."""
        ),
        code(
            """# 0. Mount Drive first, then install the uploaded MolGap wheel.
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib, importlib.metadata, json, shutil, subprocess, sys, torch

DRIVE = Path('/content/drive/MyDrive')
PROJECT_ROOT = DRIVE / 'MolGap'
NOTEBOOK_ROOT = PROJECT_ROOT / 'notebooks'
wheel_hits = sorted(NOTEBOOK_ROOT.glob('molgap-*.whl'))
assert len(wheel_hits) == 1, (
    f'Expected exactly one MolGap wheel in {NOTEBOOK_ROOT}; found {wheel_hits}'
)
EXPECTED_WHEEL_SHA256 = (
    'f38a93f558dd5c13e879740e7af7eb5e'
    '6135b520a3b220b53ff5863e6ed7cce5'
)
wheel_sha256 = hashlib.sha256(wheel_hits[0].read_bytes()).hexdigest()
assert wheel_sha256 == EXPECTED_WHEEL_SHA256, (
    f'Wrong MolGap wheel: {wheel_sha256}'
)

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'])
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
    'torch-geometric==2.6.1', 'pandas', 'tqdm', 'psutil',
])
torch_version = torch.__version__.split('+')[0]
cuda_tag = torch.version.cuda.replace('.', '')
wheel_index = f'https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_tag}.html'
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
    'torch_cluster', '-f', wheel_index,
])
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
    str(wheel_hits[0]),
])
assert torch.cuda.is_available(), 'Select an A100 GPU runtime.'
assert torch.cuda.is_bf16_supported(), 'Stable recovery requires BF16 support.'
print('GPU:', torch.cuda.get_device_name(0))
print('MolGap wheel:', wheel_hits[0].name)"""
        ),
        code(
            """# 1. Resolve immutable Drive inputs and a separate durable output.
from molgap.repaired_2m_schnet import (
    preflight_repaired_2m_schnet,
    sha256_file,
    stable_recovery_config,
    train_repaired_2m_schnet,
    verify_accepted_graph_cache,
)
from molgap.repaired_2m_3d_colab import atomic_json

INPUT_ROOT = PROJECT_ROOT / 'results' / 'molgap_repaired_2m_primary_payload'
DRIVE_GRAPH_DIR = INPUT_ROOT / 'graph_shards'
LOCAL_GRAPH_DIR = Path('/content/molgap_repaired_2m_primary_graph_shards')
ACCEPTANCE = INPUT_ROOT / 'primary_acceptance.json'
RECOVERY_CHECKPOINT = INPUT_ROOT / 'primary_fp16_epoch7_checkpoint.pt'
OUTPUT_DIR = (
    PROJECT_ROOT / 'checkpoints' /
    'molgap_repaired_2m_primary_verified_recovery_v2'
)
PREFLIGHT = OUTPUT_DIR / 'preflight.json'
EXPECTED_RECOVERY_SHA256 = (
    '858f48dd8455e07c89d2631fe99dbc91'
    'd75ff326faaaa566e45b8fce0641320a'
)
EXPECTED_ACCEPTANCE_SHA256 = (
    '9b0080b2da897880a3b51111dbf914af'
    '16db98b0dacf90353fdb800104f2e659'
)

drive_graph_paths = sorted(DRIVE_GRAPH_DIR.glob('graphs_*.pt'))
assert len(drive_graph_paths) == 100, (
    f'Expected 100 graph shards; found {len(drive_graph_paths)}'
)
assert ACCEPTANCE.is_file(), ACCEPTANCE
assert RECOVERY_CHECKPOINT.is_file(), RECOVERY_CHECKPOINT
assert sha256_file(RECOVERY_CHECKPOINT) == EXPECTED_RECOVERY_SHA256
assert sha256_file(ACCEPTANCE) == EXPECTED_ACCEPTANCE_SHA256
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
assert not (OUTPUT_DIR / 'completion_manifest.json').exists(), (
    f'Verified run already completed: {OUTPUT_DIR}'
)

drive_audit = verify_accepted_graph_cache(DRIVE_GRAPH_DIR, ACCEPTANCE)
print('Drive cache accepted:', drive_audit)

# Stage immutable inputs once on local SSD; Drive remains the durable source.
LOCAL_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
for index, source in enumerate(drive_graph_paths, start=1):
    target = LOCAL_GRAPH_DIR / source.name
    expected_hash = next(
        row['sha256']
        for row in json.loads(ACCEPTANCE.read_text())['shards']
        if Path(row['path']).name == source.name
    )
    if not target.exists() or sha256_file(target) != expected_hash:
        temporary = target.with_suffix('.pt.partial')
        shutil.copy2(source, temporary)
        assert sha256_file(temporary) == expected_hash, source.name
        temporary.replace(target)
    if index % 10 == 0:
        print(f'local staging: {index}/100')
GRAPH_DIR = LOCAL_GRAPH_DIR
graph_paths = sorted(GRAPH_DIR.glob('graphs_*.pt'))
local_audit = verify_accepted_graph_cache(GRAPH_DIR, ACCEPTANCE)
assert local_audit['graph_ledger_sha256'] == drive_audit['graph_ledger_sha256']

CONFIG = stable_recovery_config('primary')
environment = {
    'python': sys.version,
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'torch_geometric': importlib.metadata.version('torch-geometric'),
    'torch_cluster': importlib.metadata.version('torch-cluster'),
    'gpu': torch.cuda.get_device_name(0),
    'molgap_wheel': wheel_hits[0].name,
    'molgap_wheel_sha256': wheel_sha256,
    'recovery_sha256': sha256_file(RECOVERY_CHECKPOINT),
    'acceptance_sha256': sha256_file(ACCEPTANCE),
    'graph_ledger_sha256': local_audit['graph_ledger_sha256'],
}
atomic_json(environment, OUTPUT_DIR / 'environment.json')
print('input size GiB:', sum(p.stat().st_size for p in graph_paths) / 2**30)
print('output:', OUTPUT_DIR)
print('config:', CONFIG)
print('environment:', environment)"""
        ),
        code(
            """# 2. Hash all shards and run forward/backward on first and last shards.
preflight = preflight_repaired_2m_schnet(
    primary_graph_dir=GRAPH_DIR,
    primary_acceptance=ACCEPTANCE,
    secondary_graph_dir=None,
    secondary_acceptance=None,
    variant='primary',
    output_path=PREFLIGHT,
    config=CONFIG,
)
print(preflight)"""
        ),
        code(
            """# 3. Recover training. Re-running this cell resumes OUTPUT_DIR/checkpoint.pt.
result = train_repaired_2m_schnet(
    primary_graph_dir=GRAPH_DIR,
    primary_acceptance=ACCEPTANCE,
    secondary_graph_dir=None,
    secondary_acceptance=None,
    output_dir=OUTPUT_DIR,
    config=CONFIG,
    recovery_checkpoint=RECOVERY_CHECKPOINT,
)
print('test average MAE:', result['test_average_mae_eV'])
print('test MAE:', result['test_mae_eV'])
print('best epoch:', result['best_epoch'])
print('best origin:', result['best_origin'])
print('recovery baseline:', result['recovery_baseline'])
print('evaluation consistency:', result['evaluation_consistency'])
print('model:', result['artifacts']['model'])
print('metrics:', OUTPUT_DIR / 'metrics.json')"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
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
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
