from __future__ import annotations

import ast
import json
from pathlib import Path

from molgap.pcqm_shadow import (
    FROZEN_CANDIDATE,
    SHADOW_RESERVE_ROWS,
    SHADOW_ROWS,
    SHADOW_SPLIT_SEED,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "pcqm_shadow.py"
ENTRY = ROOT / "experiments" / "pcqm_k1_shadow" / "cpu_cache" / "run_cache.py"
METADATA = ENTRY.with_name("kernel-metadata.json")


def test_shadow_selection_is_frozen():
    assert FROZEN_CANDIDATE == "neural_atom_k1"
    assert SHADOW_ROWS == 10_000
    assert SHADOW_RESERVE_ROWS == 256
    assert SHADOW_SPLIT_SEED == 2_026_091_042


def test_cache_builder_is_label_sealed():
    source = MODULE.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'usecols=["idx", "smiles"]' in source
    assert 'csv_columns_read": ["idx", "smiles"]' in source
    assert "homolumogap" not in source.lower()
    assert 'if "y" in graph' in source
    assert '"shadow_labels_read": False' in source


def test_parent_overlap_is_rejected():
    source = MODULE.read_text(encoding="utf-8")
    assert "if any(value in used for value in effective)" in source
    assert "Expected 110000 distinct parent rows" in source


def test_cpu_kernel_has_only_frozen_inputs():
    ast.parse(ENTRY.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "false"
    assert metadata["dataset_sources"] == [
        "piero0/pcqm4mv2",
        "kaseichou/molgap-pcqm-k1-shadow-source-v2",
        "kaseichou/molgap-pcqm-geometry-cache-s42-dataset",
    ]


def test_cpu_kernel_pins_and_preflights_rdkit():
    source = ENTRY.read_text(encoding="utf-8")
    assert 'RDKIT_VERSION = "2026.3.6"' in source
    assert 'f"rdkit=={RDKIT_VERSION}"' in source
    assert 'Chem.MolFromSmiles("CC")' in source
    assert 'smiles2graph("CC")' in source
    assert source.index("from rdkit import Chem") < source.index(
        "from molgap.pcqm_shadow import build_shadow_cache"
    )
