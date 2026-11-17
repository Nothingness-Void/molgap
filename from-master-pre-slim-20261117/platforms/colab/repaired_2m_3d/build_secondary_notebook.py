"""Generate the repaired-2M independent second-conformer notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "molgap_repaired_2m_second_conformer.ipynb"


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
            """# MolGap repaired-2M: resumable independent second conformer

Run only after the 100 primary graph shards pass acceptance. This notebook
generates one independently seeded ETKDGv3+MMFF view for every accepted primary
identity. It never modifies the primary cache and writes 100 atomic Drive
shards with per-shard hashes and resume state."""
        ),
        code(
            """# 0. Mount Drive and install the prepared MolGap wheel.
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/MolGap')
NOTEBOOK_ROOT = PROJECT_ROOT / 'notebooks'
wheel_hits = sorted(NOTEBOOK_ROOT.glob('molgap-*.whl'))
assert len(wheel_hits) == 1, wheel_hits
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
                       'torch-geometric==2.6.1', 'rdkit==2025.3.5', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                       '--force-reinstall', str(wheel_hits[0])])
print('MolGap wheel:', wheel_hits[0].name)"""
        ),
        code(
            """# 1. Resolve the accepted primary cache and durable secondary output.
from molgap.repaired_2m_3d_colab import (
    build_secondary_graph_shards,
    validate_graph_shards,
)

RAW_ROOT = PROJECT_ROOT / 'raw_data'
RESULTS_ROOT = PROJECT_ROOT / 'results'

def find_one(root, exact_name):
    hits = sorted(root.rglob(exact_name))
    assert len(hits) == 1, f'Expected one {exact_name}; found {hits}'
    return hits[0]

REPAIRED_CSV = find_one(RAW_ROOT, 'phase8_repaired_2m.csv')
PRIMARY_GRAPH_DIR = RESULTS_ROOT / 'molgap_phase8_repaired_2m_3d' / 'graph_shards'
SECONDARY_GRAPH_DIR = (
    RESULTS_ROOT / 'molgap_phase8_repaired_2m_3d_secondary' / 'graph_shards'
)
assert len(list(PRIMARY_GRAPH_DIR.glob('graphs_*.pt'))) == 100, (
    'Primary cache must contain 100 accepted shards before secondary build.'
)
SECONDARY_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
print('primary:', PRIMARY_GRAPH_DIR)
print('secondary:', SECONDARY_GRAPH_DIR)"""
        ),
        code(
            """# 2. Build or resume the 100 secondary shards.
BUILD_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
result = build_secondary_graph_shards(
    repaired_csv=REPAIRED_CSV,
    primary_graph_dir=PRIMARY_GRAPH_DIR,
    output_dir=SECONDARY_GRAPH_DIR,
    workers=BUILD_WORKERS,
    seed=314159,
    verify_repaired_sha256=True,
)
print(result)"""
        ),
        code(
            """# 3. Strictly validate all completed secondary shards.
if (SECONDARY_GRAPH_DIR / 'build_completion.json').exists():
    print(validate_graph_shards(SECONDARY_GRAPH_DIR))
else:
    print('Secondary build is incomplete; rerun cell 2 to resume.')"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
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
