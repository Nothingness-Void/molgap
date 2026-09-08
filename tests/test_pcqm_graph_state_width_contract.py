from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/molgap/pcqm_gap_architecture.py"
RUNNER = ROOT / "src/molgap/pcqm_local_global_runner.py"
ACCEPTANCE = ROOT / "experiments/pcqm_gap_architecture/accept_pcqm100k_graph_state_width.py"
PROTOCOL = ROOT / "experiments/pcqm_gap_architecture/graph_state_width_seed42_protocol.md"
SINGLE_RUNNER = ROOT / "src/molgap/pcqm_graph_state_width_single_runner.py"
KERNEL = ROOT / "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/graph_state_width_seed42"
BASELINE_KERNEL = ROOT / "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/graph_state_width_baseline_seed42"
CANDIDATE_KERNEL = ROOT / "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/graph_state_width_candidate_seed42"
BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9_w128"


def test_graph_state_width_sources_parse() -> None:
    for path in (MODEL, RUNNER, SINGLE_RUNNER, ACCEPTANCE):
        ast.parse(path.read_text(encoding="utf-8"))


def test_factory_changes_only_the_graph_state_width_contract() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert CANDIDATE in source
    assert 'graph_state_channels=(128 if candidate.endswith("_w128") else 64)' in source
    runner = RUNNER.read_text(encoding="utf-8")
    assert '"graph_state_width"' in runner
    assert "GRAPH_STATE_WIDTH_CANDIDATES" in runner
    assert "GRAPH_STATE_WIDTH_CANDIDATES[1]: 3_803_985" in runner
    assert "shared_parameter_shape_differences" in runner
    assert 'name.startswith("graph_context.")' in runner


def test_protocol_freezes_the_bounded_seed42_gate() -> None:
    source = PROTOCOL.read_text(encoding="utf-8")
    for token in (
        "100,000",
        "10,000",
        "seed 42",
        "GraphState64",
        "GraphState128",
        "0.001 eV",
        "at least 70%",
        "official validation",
        "test-dev",
        "full scale gates require separate authorization",
        "same GPU model",
    ):
        assert token in source


def test_single_gpu_fallback_is_one_candidate_and_keeps_durable_outputs() -> None:
    source = SINGLE_RUNNER.read_text(encoding="utf-8")
    for token in (
        "MOLGAP_SINGLE_CANDIDATE",
        "ensure_pascal_compatible_torch",
        '"torch==2.7.1"',
        '"nvidia-cusparselt-cu12==0.6.3"',
        '"https://download.pytorch.org/whl/cu126"',
        '"sm_60"',
        "os.execv",
        "torch.cuda.device_count() != 1",
        "initialization_preflight.json",
        "single_result.json",
        "single_failure.json",
        "base.gpu_preflight",
        "base.train_one",
        '"full_data_authorized": False',
    ):
        assert token in source


def test_acceptance_recomputes_paired_metrics_and_keeps_roles_sealed() -> None:
    source = ACCEPTANCE.read_text(encoding="utf-8")
    for token in (
        BASELINE,
        CANDIDATE,
        "paired_delta_bootstrap_ci95_eV",
        "material_gain_at_least_0_001_eV",
        "throughput_gate_at_least_0_70x",
        '"official_validation_role_read": False',
        '"test_dev_role_read": False',
    ):
        assert token in source


def test_kaggle_package_pins_private_dual_t4_inputs() -> None:
    import json

    wrapper = (KERNEL / "run_screen.py").read_text(encoding="utf-8")
    ast.parse(wrapper)
    assert 'EXPECTED_SOURCE_COMMIT = "2f354e07f21818d7fdf6610b5a876483ae014fb8"' in wrapper
    assert '"MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "graph_state_width"' in wrapper
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "nothingnessvoid/molgap-pcqm-graphstate-width-s42"
    assert metadata["is_private"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "nothingnessvoid/molgap-pcqm-graphstate-width-source-20260908",
        "nothingnessvoid/molgap-pcqm-geometry-cache-s42-dataset",
    ]


def test_single_gpu_packages_pin_the_same_source_and_isolate_gpu_zero() -> None:
    import json

    expected_ids = {
        BASELINE_KERNEL: "nothingnessvoid/molgap-pcqm-graphstate-w64-s42",
        CANDIDATE_KERNEL: "nothingnessvoid/molgap-pcqm-graphstate-w128-s42",
    }
    for root, expected_id in expected_ids.items():
        wrapper = (root / "run_single.py").read_text(encoding="utf-8")
        ast.parse(wrapper)
        assert 'EXPECTED_SOURCE_COMMIT = "c413fc13b8b15e659e90853c429119189577c379"' in wrapper
        assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in wrapper
        assert 'os.environ["MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "graph_state_width"' in wrapper
        metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"] == expected_id
        assert metadata["is_private"] == "true"
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert metadata["dataset_sources"] == [
            "nothingnessvoid/molgap-pcqm-graphstate-width-source-v2-20260908",
            "nothingnessvoid/molgap-pcqm-geometry-cache-s42-dataset",
        ]
