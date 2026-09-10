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
RUNTIME = ROOT / "src" / "molgap" / "pcqm_local_global_runner.py"
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_local_global_allocation.py"
PROTOCOL = EXPERIMENT / "local_global_allocation_seed42_protocol.md"
CONFIRMATION_PROTOCOL = EXPERIMENT / "local_global_allocation_multiseed_protocol.md"
CONFIRMATION_RUNNER = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "local_global_allocation_seed43"
    / "run_confirmation.py"
)
CONFIRMATION_METADATA = CONFIRMATION_RUNNER.with_name("kernel-metadata.json")
SEED44_RUNNER = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "local_global_allocation_seed44"
    / "run_confirmation.py"
)
SEED44_METADATA = SEED44_RUNNER.with_name("kernel-metadata.json")


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
    wrapper = RUNNER.read_text(encoding="utf-8")
    ast.parse(wrapper)
    assert '"MOLGAP_LOCAL_GLOBAL_RUN_MODE", "seed42_screen"' in wrapper
    assert '"MOLGAP_LOCAL_GLOBAL_SEED", "42"' in wrapper
    assert 'run_module("molgap.pcqm_local_global_runner"' in wrapper
    source = RUNTIME.read_text(encoding="utf-8")
    ast.parse(source)
    assert "SCREEN_CANDIDATES = (" in source
    assert "CONFIRMATION_CANDIDATES" in source
    assert "EXPECTED_GLOBAL_BLOCKS" in source
    assert "tuple(range(1, 10))" in source
    assert "(3, 6, 9)" in source
    assert '"MOLGAP_LOCAL_GLOBAL_SEED", "42"' in source
    assert "BATCH_SIZE = 48" in source
    assert "LEARNING_RATE = 1.6e-4" in source
    assert "WEIGHT_DECAY = 1.0e-6" in source
    assert "MAX_EPOCHS = 40" in source
    assert "PATIENCE = 8" in source
    assert "PARAMETER_BUDGET = 5_200_000" in source
    assert "SEARCH_BUDGET_S = 39_600" in source
    assert "EXPECTED_GPU_COUNT = 2" in source
    assert 'EXPECTED_GPU_TOKEN = "T4"' in source
    assert "LOADER_WORKERS = 0" in source
    assert "subprocess.Popen" in source
    assert 'os.environ.get("MOLGAP_T4_WORKER")' in source
    assert 'environment["CUDA_VISIBLE_DEVICES"]' in source
    assert "multiprocessing.get_context" not in source
    assert '"dual_t4_candidate_parallel"' in source
    assert "ensure_pascal_compatible_torch" not in source
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
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
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


def test_seed43_confirmation_is_paired_and_t4x2() -> None:
    source = CONFIRMATION_RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "confirmation"' in source
    assert '"MOLGAP_LOCAL_GLOBAL_SEED"] = "43"' in source
    assert "MOLGAP_EXPECTED_MODEL_SOURCE_COMMIT" in source
    assert "9068ddb82e6bdf16b841570abbff023b90c07f07" in source
    assert "__PIN_AFTER_FEATURE_COMMIT__" not in source
    metadata = json.loads(CONFIRMATION_METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == (
        "nothingnessvoid/molgap-pcqm-graphstate-confirmation-s43"
    )
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] == "true"
    assert metadata["is_private"] == "true"
    assert metadata["kernel_sources"] == []


def test_seed44_confirmation_changes_only_the_seed_identity() -> None:
    source = SEED44_RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "confirmation"' in source
    assert '"MOLGAP_LOCAL_GLOBAL_SEED"] = "44"' in source
    assert "9068ddb82e6bdf16b841570abbff023b90c07f07" in source
    metadata = json.loads(SEED44_METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == (
        "nothingnessvoid/molgap-pcqm-graphstate-confirmation-s44"
    )
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    seed43_metadata = json.loads(CONFIRMATION_METADATA.read_text(encoding="utf-8"))
    assert metadata["dataset_sources"] == seed43_metadata["dataset_sources"]


def test_confirmation_protocol_stops_early_and_keeps_roles_sealed() -> None:
    source = CONFIRMATION_PROTOCOL.read_text(encoding="utf-8")
    assert "Seed 43 runs first" in source
    assert "seed 44 is submitted only if seed 43 strictly passes" in source
    assert "GPU 0 trains the fresh full-GPS" in source
    assert "GPU 1 trains the fresh GraphState" in source
    assert "Official validation and test-dev remain unread" in source
    assert "full-data training" in source


def test_protocol_keeps_ring_ready_and_sealed_roles_closed() -> None:
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "full-GPS comparator is trained fresh" in source
    assert "at or below 5.2M" in source
    assert "39,600 seconds" in source
    assert "strictly lower" in source
    assert "ring-hierarchy cache remains accepted and deferred" in source
    assert "official validation/test-dev" in source
    assert "fresh Python processes" in source
