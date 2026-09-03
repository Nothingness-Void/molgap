from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "molgap" / "pcqm_gap_architecture.py"
RUNNER = ROOT / "src" / "molgap" / "pcqm_local_global_runner.py"
PROTOCOL = (
    ROOT
    / "experiments"
    / "pcqm_gap_architecture"
    / "body_order_moment_seed42_protocol.md"
)
WRAPPER = (
    ROOT
    / "experiments"
    / "pcqm_gap_architecture"
    / "kaggle_pcqm_gap100k"
    / "body_order_graphstate_seed42"
    / "run_screen.py"
)
ACCEPTANCE = (
    ROOT
    / "experiments"
    / "pcqm_gap_architecture"
    / "accept_pcqm100k_body_order_graphstate_seed42.py"
)
BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_body_order_triangle_edge_state_graph_state9"


def class_segment(name: str) -> str:
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    return ast.get_source_segment(source, node)


def test_body_order_sources_parse() -> None:
    for path in (MODEL, RUNNER, WRAPPER, ACCEPTANCE):
        ast.parse(path.read_text(encoding="utf-8"))


def test_body_order_is_one_stateless_invariant_injection() -> None:
    source = class_segment(
        "OGBBodyOrderMomentGraphStateGeometrySparseTriangleEdgeStateWrapper"
    )
    for token in (
        "BODY_ORDER_BASIS_CHANNELS = 16",
        "_FixedGaussianBasis(\n            0.75, 2.25",
        "source, destination = edge_index.unbind(dim=0)",
        "scalar_density",
        "vector_moment",
        "rank2_moment",
        "vector_moment.square().sum(dim=-1)",
        "rank2_moment.square().sum(dim=(-1, -2))",
        "nn.LayerNorm(48)",
        "nn.Linear(48, 64)",
        "nn.SiLU()",
        "nn.Linear(64, 192, bias=False)",
        "nn.init.zeros_(self.body_order_injection[-1].weight)",
        "return h + self.body_order_injection(invariants) * node_mask",
    ):
        assert token in source
    for forbidden in (
        "contact_edge_index",
        "contact_update",
        "ring_update",
        "auxiliary_payload",
        "attention",
    ):
        assert forbidden not in source.lower()


def test_body_order_parameter_arithmetic_matches_protocol() -> None:
    baseline = 3_665_809
    injection = (2 * 48) + (48 * 64 + 64) + (64 * 192)
    assert baseline + injection == 3_681_329
    runner = RUNNER.read_text(encoding="utf-8")
    assert f'BODY_ORDER_GRAPHSTATE_CANDIDATES[1]: 3_681_329' in runner


def test_body_order_runner_freezes_geometry_and_gpu_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for token in (
        '"body_order_graphstate"',
        CANDIDATE,
        '"dual_t4_candidate_parallel"',
        "batch.pos",
        "body_order_injection_zero",
        "initial_prediction_equal_to_baseline",
        "initial_prediction_max_abs_difference",
        "rtol=1.0e-6",
        "atol=1.0e-7",
        "body_order_return_gradient_nonzero",
        "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22",
        "BATCH_SIZE = 48",
        "LEARNING_RATE = 1.6e-4",
        "WEIGHT_DECAY = 1.0e-6",
        "MAX_EPOCHS = 40",
        "PATIENCE = 8",
        "SEARCH_BUDGET_S = 14_400",
        '"official_validation_role_read": False',
        '"test_dev_role_read": False',
    ):
        assert token in source
    assert "find_contact_cache" in source
    assert "body_order_graphstate" in source
    model = MODEL.read_text(encoding="utf-8")
    assert CANDIDATE in model


def test_body_order_protocol_seals_roles_and_variants() -> None:
    source = PROTOCOL.read_text(encoding="utf-8")
    for token in (
        "one cheap, rotationally invariant local-environment summary",
        "one pre-message node injection",
        "No new CPU graph cache is required",
        "Official PCQM validation/test-dev",
        "extra seeds",
        "radial-channel, order, width, seed, optimizer,",
        "or schedule variants",
    ):
        assert token in source


def test_body_order_remote_package_pins_the_same_contract() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    acceptance = ACCEPTANCE.read_text(encoding="utf-8")
    for token in (
        '"body_order_graphstate"',
        "9785022300a54ee4e8f66c378cf6a9c19d2b20e5",
        "pcqm_gap100k_body_order_graphstate_seed42",
    ):
        assert token in wrapper
    for token in (
        CANDIDATE,
        "3_681_329",
        "body_order_moment_present",
        "body_order_injection_zero",
        "initial_prediction_equal_to_baseline",
        "initial_prediction_max_abs_difference",
        "body_order_return_gradient_nonzero",
        '"official_validation_role_read": False',
        '"test_dev_role_read": False',
    ):
        assert token in acceptance
