from __future__ import annotations

import ast
import json
from pathlib import Path

from molgap.pcqm_fourier_edge import (
    BATCH_SIZE,
    EXPECTED_PARAMETERS,
    GAP_EPOCHS,
    GEOMETRY_CACHE_SHA256,
    PARENT_GRAPH_CACHE_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "pcqm_fourier_edge.py"
ENTRY = (
    ROOT
    / "experiments"
    / "pcqm_fourier_edge_transfer"
    / "gpu_seed42"
    / "run_screen.py"
)
METADATA = ENTRY.with_name("kernel-metadata.json")


def test_frozen_batch_and_exposure():
    assert BATCH_SIZE == 128
    assert GAP_EPOCHS == 40
    assert EXPECTED_PARAMETERS == {
        "full_gps": 4_771_073,
        "neural_atom_k1": 3_658_817,
        "fourier_edge_k1": 3_658_241,
    }


def test_cache_ancestry_is_pinned():
    assert GEOMETRY_CACHE_SHA256 == (
        "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
    )
    assert PARENT_GRAPH_CACHE_SHA256 == (
        "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
    )


def test_source_is_static_and_geometry_is_removed():
    source = MODULE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "_strip_geometry(roles)" in source
    for field in (
        '"pos"',
        '"edge_distance"',
        '"wedge_angle_cos"',
        '"wedge_edge_ids"',
        '"geometry_valid"',
    ):
        assert field in source
    assert "official_validation_role_read" in source
    assert "test_dev_role_read" in source
    assert "shadow_audit_read" in source


def test_entrypoint_requests_t4x2_and_two_isolated_workers():
    source = ENTRY.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'torch.cuda.device_count() != 2' in source
    assert 'launch_worker("anchor"' in source
    assert 'launch_worker("edge"' in source
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-pcqm-fourier-edge-source",
        "kaseichou/molgap-pcqm-geometry-cache-s42-dataset",
    ]


def test_no_forbidden_method_is_admitted():
    source = MODULE.read_text(encoding="utf-8").lower()
    for token in (
        "teacher_model",
        "distillation_loss",
        "residual_target",
        "load_test_dev",
    ):
        assert token not in source
