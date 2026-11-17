"""Generate the repaired-2M resumable Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "molgap_repaired_2m_3d.ipynb"


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
            """# MolGap repaired-2M: resumable ETKDG and lightweight SchNet

This notebook builds the reusable repaired-2M ETKDG cache. It reuses matching
coordinates from the accepted original-1M cache by CID and canonical SMILES,
then builds only missing molecules. Every 20K-row shard is written atomically
to Drive and can be resumed independently.

Full SchNet training is disabled by default. The bounded-residual gate has
passed, but training still starts only after the primary cache is accepted.
The fixed Route B primary model is `176/160/6`, cutoff 10 A, dropout 0.05."""
        ),
        code(
            """# 0. Mount Drive first, install dependencies, then install the local MolGap wheel.
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os, subprocess, sys, torch

DRIVE = Path('/content/drive/MyDrive')
PROJECT_ROOT = DRIVE / 'MolGap'
NOTEBOOK_ROOT = PROJECT_ROOT / 'notebooks'
wheel_hits = sorted(NOTEBOOK_ROOT.glob('molgap-*.whl'))
assert len(wheel_hits) == 1, f'Expected one MolGap wheel in {NOTEBOOK_ROOT}; found {wheel_hits}'

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'])
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
    'torch-geometric==2.6.1', 'rdkit==2025.3.5', 'pandas', 'tqdm', 'psutil',
])
if torch.cuda.is_available():
    torch_version = torch.__version__.split('+')[0]
    cuda_tag = torch.version.cuda.replace('.', '')
    wheel_index = f'https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_tag}.html'
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
        'torch_cluster', '-f', wheel_index,
    ])
    print('GPU:', torch.cuda.get_device_name(0), '| PyG wheel index:', wheel_index)
else:
    print('CPU runtime: graph construction is available; full training remains disabled.')
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
    str(wheel_hits[0]),
])
print('MolGap wheel:', wheel_hits[0].name)"""
        ),
        code(
            """# 1. Resolve immutable inputs and durable Drive output paths.
from molgap.repaired_2m_3d_colab import (
    ROUTE_B_SCHNET_CONFIG,
    TRAINING_APPROVAL_TOKEN,
    build_graph_shards,
    export_embeddings,
    train_light_schnet,
    validate_graph_shards,
)

RAW_ROOT = PROJECT_ROOT / 'raw_data'
RESULTS_ROOT = PROJECT_ROOT / 'results'
CHECKPOINTS_ROOT = PROJECT_ROOT / 'checkpoints'

def find_one(root, exact_name):
    hits = sorted(root.rglob(exact_name))
    assert len(hits) == 1, f'Expected one {exact_name} below {root}; found {hits}'
    return hits[0]

REPAIRED_CSV = find_one(RAW_ROOT, 'phase8_repaired_2m.csv')
ORIGINAL_CSV = find_one(RAW_ROOT, 'phase8_expansion_1m.csv')
ORIGINAL_3D = find_one(RESULTS_ROOT, 'pyg_3d_graphs_etkdg_expansion_1m.pt')

RUN_ROOT = RESULTS_ROOT / 'molgap_phase8_repaired_2m_3d'
GRAPH_DIR = RUN_ROOT / 'graph_shards'
TRAIN_RESULT_DIR = RUN_ROOT / 'training'
EMBEDDING_DIR = RUN_ROOT / 'embedding_shards'
CHECKPOINT_DIR = CHECKPOINTS_ROOT / 'molgap_phase8_repaired_2m_3d'
for path in (RUN_ROOT, GRAPH_DIR, TRAIN_RESULT_DIR, EMBEDDING_DIR, CHECKPOINT_DIR):
    path.mkdir(parents=True, exist_ok=True)

print('repaired CSV:', REPAIRED_CSV, f'{REPAIRED_CSV.stat().st_size / 1e6:.1f} MB')
print('original CSV:', ORIGINAL_CSV, f'{ORIGINAL_CSV.stat().st_size / 1e6:.1f} MB')
print('original 3D :', ORIGINAL_3D, f'{ORIGINAL_3D.stat().st_size / 1e9:.2f} GB')
print('graph output:', GRAPH_DIR)
print('model config:', ROUTE_B_SCHNET_CONFIG)"""
        ),
        code(
            """# 2. Build or resume 100 atomic graph shards.
# Existing completed shards are SHA-checked and skipped.
BUILD_GRAPHS = True
SHARD_SIZE = 20_000
BUILD_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))

if BUILD_GRAPHS:
    build_result = build_graph_shards(
        repaired_csv=REPAIRED_CSV,
        original_csv=ORIGINAL_CSV,
        original_graph_cache=ORIGINAL_3D,
        output_dir=GRAPH_DIR,
        shard_size=SHARD_SIZE,
        workers=BUILD_WORKERS,
        seed=42,
        verify_repaired_sha256=True,
    )
    print(build_result)
else:
    print('Graph build skipped by configuration.')"""
        ),
        code(
            """# 3. Strict acceptance: hashes, counts, uniqueness, finite values, and Gap identity.
# This cell is resumable but intentionally reads every completed graph shard.
if (GRAPH_DIR / 'build_completion.json').exists():
    validation = validate_graph_shards(GRAPH_DIR)
    print(validation)
else:
    print('Build is incomplete; validation is not available yet.')"""
        ),
        code(
            """# 4. Full training gate. Enable only after all 100 primary shards pass acceptance.
ENABLE_FULL_TRAINING = False
APPROVAL_TOKEN = ''

if ENABLE_FULL_TRAINING:
    assert APPROVAL_TOKEN == TRAINING_APPROVAL_TOKEN, (
        'Use the frozen approval token only after primary graph acceptance.'
    )
    assert torch.cuda.is_available(), 'Select a GPU runtime before full training.'
    print('Full repaired-2M lightweight SchNet training is authorized.')
else:
    print('Full training remains gated. Graph shards are complete, durable, and reusable.')"""
        ),
        code(
            """# 5. Train the full repaired-2M lightweight SchNet with epoch-level Drive checkpoints.
training_result = None
if ENABLE_FULL_TRAINING:
    training_result = train_light_schnet(
        graph_dir=GRAPH_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        result_dir=TRAIN_RESULT_DIR,
        approval_token=APPROVAL_TOKEN,
        epochs=20,
        batch_size=128,
        num_workers=2,
        seed=42,
        source_rows=2_000_000,
        model_config=ROUTE_B_SCHNET_CONFIG,
    )
    print(training_result['test_metrics'])
else:
    print('Skipped by the training gate.')"""
        ),
        code(
            """# 6. Export 100 independently resumable embedding parts for later Fusion.
best_checkpoint = CHECKPOINT_DIR / 'repaired_2m_light_schnet_best.pt'
if ENABLE_FULL_TRAINING and best_checkpoint.exists():
    embedding_result = export_embeddings(
        graph_dir=GRAPH_DIR,
        best_checkpoint=best_checkpoint,
        output_dir=EMBEDDING_DIR,
        batch_size=128,
        num_workers=2,
        model_config=ROUTE_B_SCHNET_CONFIG,
    )
    print(embedding_result)
else:
    print('Embedding export waits for an accepted full-training checkpoint.')"""
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
