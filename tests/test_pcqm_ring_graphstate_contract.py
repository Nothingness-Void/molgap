from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"
MODEL = ROOT / "src" / "molgap" / "pcqm_gap_architecture.py"
RUNTIME = ROOT / "src" / "molgap" / "pcqm_local_global_runner.py"
ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_ring_graphstate_seed42.py"
PROTOCOL = EXPERIMENT / "ring_graphstate_seed42_protocol.md"
RUNNER = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "ring_graphstate_seed42"
    / "run_screen.py"
)
METADATA = RUNNER.with_name("kernel-metadata.json")
CANDIDATE = (
    "ogb_distance_angle_ring_hierarchy_triangle_edge_state_graph_state9"
)


def class_segment(name: str) -> str:
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    return ast.get_source_segment(source, node)


def test_combined_model_adds_only_ring_hierarchy_to_graphstate() -> None:
    source = MODEL.read_text(encoding="utf-8")
    segment = class_segment(
        "OGBRingHierarchyGraphStateGeometrySparseTriangleEdgeStateWrapper"
    )
    assert "OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper" in segment
    assert 'global_mode="graph_state"' in segment
    assert "HIERARCHY_LAYERS = (1, 3, 5, 7)" in segment
    assert "self.ring_update = _SharedRingHierarchyUpdate(" in segment
    assert "self.graph_context.initialize" in segment
    assert segment.index("self.ring_update(") < segment.index("self.graph_context(")
    assert CANDIDATE in source


def test_runtime_freezes_paired_t4x2_ring_graphstate_contract() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"ring_graphstate"' in source
    assert "RING_GRAPHSTATE_CANDIDATES = (" in source
    assert "EXPECTED_RING_CACHE_SHA256" in source
    assert "3_665_809" in source
    assert "3_723_849" in source
    assert "PARAMETER_BUDGET = 4_000_000" in source
    assert "SEARCH_BUDGET_S = 14_400" in source
    assert 'environment["CUDA_VISIBLE_DEVICES"]' in source
    assert "ring_return_gradient_nonzero" in source
    assert "paired_against_baseline" in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source


def test_acceptance_is_no_model_and_recomputes_the_gate() -> None:
    source = ACCEPTANCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "import torch" not in source
    assert "candidate_minus_baseline_eV" in source
    assert "ring_return_gradient_nonzero" in source
    assert "model_inference_executed" in source
    assert "official_validation_role_read" in source
    assert "test_dev_role_read" in source


def test_protocol_changes_one_mechanism_and_stops_after_seed42() -> None:
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "fresh GraphState9 baseline" in source
    assert "one material architecture change" in source
    assert "GPU 0" in source and "GPU 1" in source
    assert "14,400-second" in source
    assert "strictly lower" in source
    assert "does not automatically trigger seeds 43/44" in source
    assert "official validation/test-dev" in source


def test_kaggle_wrapper_pins_source_cache_and_t4x2() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "ring_graphstate"' in source
    assert '"MOLGAP_LOCAL_GLOBAL_SEED"] = "42"' in source
    assert "b9f8445ac400315d82441db8292ea99e68b37dfa" in source
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "nothingnessvoid/molgap-pcqm-ring-graphstate-s42"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == [
        "nothingnessvoid/molgap-pcqm-ring-hierarchy-cache-s42"
    ]
