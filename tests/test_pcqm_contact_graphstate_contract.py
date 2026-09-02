from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "molgap" / "pcqm_gap_architecture.py"
RUNNER = ROOT / "src" / "molgap" / "pcqm_local_global_runner.py"
PROTOCOL = ROOT / "experiments" / "pcqm_gap_architecture" / "contact_state_gpu_seed42_protocol.md"
CANDIDATE = "ogb_distance_angle_contact_state_triangle_edge_state_graph_state9"


def test_contact_graphstate_sources_parse() -> None:
    ast.parse(MODEL.read_text(encoding="utf-8"))
    ast.parse(RUNNER.read_text(encoding="utf-8"))


def test_contact_graphstate_adds_only_the_frozen_narrow_state() -> None:
    source = MODEL.read_text(encoding="utf-8")
    for token in (
        "class OGBContactStateGraphStateGeometrySparseTriangleEdgeStateWrapper",
        "CONTACT_LAYERS = (1, 3, 5, 7)",
        "contact_channels: int = 32",
        "contact_basis_channels: int = 16",
        "contact_exchange_rank: int = 16",
        "_FixedGaussianBasis(\n            0.25, 5.0",
        CANDIDATE,
    ):
        assert token in source


def test_contact_parameter_arithmetic_matches_frozen_ceiling() -> None:
    baseline = 3_665_809
    initializer = 2 * 400 + (400 * 32 + 32) + 2 * 32
    endpoint_projection = 192 * 32 + 32
    update_norm = 2 * 96
    update_gate = 96 * 32 + 32
    update_value = (96 * 32 + 32) + (32 * 32 + 32)
    output_norm = 2 * 32
    low_rank_return = (2 * 32) + (32 * 16 + 16) + 2 * (16 * 192 + 192)
    assert baseline + sum(
        (
            initializer,
            endpoint_projection,
            update_norm,
            update_gate,
            update_value,
            output_norm,
            low_rank_return,
        )
    ) == 3_700_321
    runner = RUNNER.read_text(encoding="utf-8")
    assert "CONTACT_GRAPHSTATE_CANDIDATES[1]: 3_700_321" in runner


def test_contact_runner_freezes_cache_pairing_and_sealed_roles() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    for token in (
        '"contact_graphstate"',
        CANDIDATE,
        "49725b92c2c0d33e17633abf8ffa7148ebc8bc9721d3e5b3635f1309891bc826",
        "dual_t4_candidate_parallel",
        "contact_return_gradient_nonzero",
        "official_validation_role_read",
        "test_dev_role_read",
    ):
        assert token in source
    assert "one isolated candidate per GPU" in protocol
    assert "seeds" in protocol
    assert "43/44 require" in protocol
