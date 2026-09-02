from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"
MODEL = ROOT / "src" / "molgap" / "pcqm_gap_architecture.py"
RUNNER = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "local_global_allocation_seed42"
    / "run_local_global_screen.py"
)
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_local_global_allocation.py"
PROTOCOL = EXPERIMENT / "local_global_allocation_seed42_protocol.md"


def class_segment(name: str) -> str:
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    return ast.get_source_segment(source, node)


def test_local_global_model_has_one_isolated_schedule_question() -> None:
    scheduled = class_segment("_ScheduledGPSBlock")
    wrapper = class_segment(
        "OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper"
    )
    graph_state = class_segment("_SharedGraphContext")
    assert "use_global_attention" in scheduled
    assert "to_dense_batch" in scheduled
    assert 'GLOBAL_MODES = {"sparse_attention", "graph_state"}' in wrapper
    assert "self.MODES" not in wrapper
    assert "GLOBAL_BLOCKS = (3, 6, 9)" in wrapper
    assert 'geometry_mode="distance_angle"' in wrapper
    geometry_parent = class_segment(
        "OGBGeometrySparseTriangleEdgeStateGPSWrapper"
    )
    assert 'MODES = {"distance", "angle", "distance_angle"}' in geometry_parent
    assert "self.graph_context" in wrapper
    assert "global_mean_pool" in graph_state
    assert "global_add_pool" in graph_state
    assert "global_max_pool" in graph_state
    assert "prediction" not in graph_state.lower()


def test_factory_exposes_only_the_two_new_challengers() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert "ogb_distance_angle_triangle_edge_state_sparse_gps369" in source
    assert "ogb_distance_angle_triangle_edge_state_graph_state9" in source
    assert '"sparse_attention"' in source
    assert '"graph_state"' in source


def test_remote_runner_freezes_fair_seed42_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "CANDIDATES = (" in source
    assert "EXPECTED_GLOBAL_BLOCKS" in source
    assert "tuple(range(1, 10))" in source
    assert "(3, 6, 9)" in source
    assert "SEED = 42" in source
    assert "BATCH_SIZE = 48" in source
    assert "LEARNING_RATE = 1.6e-4" in source
    assert "WEIGHT_DECAY = 1.0e-6" in source
    assert "MAX_EPOCHS = 40" in source
    assert "PATIENCE = 8" in source
    assert "PARAMETER_BUDGET = 5_200_000" in source
    assert "SEARCH_BUDGET_S = 39_600" in source
    assert "paired_against_fresh_full_gps" in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert '"molecular_research_server_accessed": False' in source


def test_remote_metadata_is_one_private_gpu_task() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == (
        "nothingnessvoid/molgap-pcqm-local-global-allocation-s42"
    )
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["code_file"] == "run_local_global_screen.py"
    assert metadata["kernel_sources"] == []
    assert metadata["dataset_sources"] == [
        "nothingnessvoid/molgap-pcqm-local-global-source",
        "nothingnessvoid/molgap-pcqm-geometry-cache-s42-dataset",
    ]


def test_acceptance_is_no_inference_and_recomputes_selection() -> None:
    source = ACCEPTANCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "import torch" not in source
    assert "fresh_comparator_mae" in source
    assert "candidate_minus_full_gps_eV" in source
    assert "shared_parameter_mismatches" in source
    assert "model_inference_executed" in source
    assert "official_validation_role_read" in source
    assert "test_dev_role_read" in source


def test_protocol_keeps_ring_ready_and_sealed_roles_closed() -> None:
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "full-GPS comparator is trained fresh" in source
    assert "at or below 5.2M" in source
    assert "39,600 seconds" in source
    assert "strictly lower" in source
    assert "ring-hierarchy cache remains accepted and deferred" in source
    assert "official validation/test-dev" in source
