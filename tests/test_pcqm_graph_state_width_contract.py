from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/molgap/pcqm_gap_architecture.py"
RUNNER = ROOT / "src/molgap/pcqm_local_global_runner.py"
ACCEPTANCE = ROOT / "experiments/pcqm_gap_architecture/accept_pcqm100k_graph_state_width.py"
PROTOCOL = ROOT / "experiments/pcqm_gap_architecture/graph_state_width_seed42_protocol.md"
BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9_w128"


def test_graph_state_width_sources_parse() -> None:
    for path in (MODEL, RUNNER, ACCEPTANCE):
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
