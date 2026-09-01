from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPS = ROOT / "src" / "molgap" / "gps.py"
QM9 = ROOT / "src" / "molgap" / "qm9_screen.py"
RUNNER = (
    ROOT
    / "experiments"
    / "top20_architecture_qm9"
    / "kaggle_edge_state_jk_readout_r5"
    / "gpu_validation"
    / "run_validation.py"
)
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = (
    ROOT
    / "experiments"
    / "top20_architecture_qm9"
    / "accept_edge_state_jk_readout_r5.py"
)


def assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment {name} in {path}")


def dict_entry_literal(path: Path, assignment: str, key: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == assignment
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, ast.Dict)
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if ast.literal_eval(key_node) == key:
                return ast.literal_eval(value_node)
    raise AssertionError(f"Missing {assignment}[{key!r}] in {path}")


def test_r5_model_and_config_are_sparse_identity_initialized() -> None:
    source = GPS.read_text(encoding="utf-8")
    assert "class EdgeJKReadoutStructuralGPSWrapper" in source
    assert "capture_layers=self.readout_layers" in source
    assert "return baseline + self.readout_delta(delta_input)" in source
    assert "nn.init.zeros_(self.readout_delta[-1].weight)" in source
    assert "nn.init.zeros_(self.readout_delta[-1].bias)" in source

    config = dict_entry_literal(
        QM9, "ENCODER_CONFIGS", "edge_state_structural_jk_readout"
    )
    assert config["kind"] == "structural_topology"
    assert config["hidden_channels"] == 192
    assert config["num_layers"] == 9
    assert config["edge_state_channels"] == 64
    assert config["readout_layers"] == (3, 6, 9)
    assert config["readout_channels"] == 32
    assert config["amp"] is False


def test_r5_parameter_contract_fits_the_frozen_budget() -> None:
    base_parameters = 4_739_651
    hidden = 192
    edge = 64
    layers = 3
    bottleneck = 32
    added = (
        layers * 2 * hidden
        + 2 * edge
        + (layers * hidden + edge) * bottleneck
        + bottleneck
        + bottleneck * hidden
        + hidden
    )
    assert base_parameters + added == 4_767_779
    assert base_parameters + added <= 4_800_000
    assert assignment_literal(RUNNER, "EXPECTED_PARAMETER_COUNT") == 4_767_779


def test_r5_runner_is_validation_only_and_anchors_accepted_r3() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "evaluate_test=False" in source
    assert '"test_role_read": False' in source
    assert "molgap-pure2d-r3-tensor-acceptance-v2" in source
    assert "EXPECTED_R3_MODEL_SHA256" in source
    assert 'EXPECTED_R3_WINNER = "edge_state_structural_gps"' in source
    assert "EXPECTED_PARAMETER_COUNT = 4_767_779" in source

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == (
        "kaseichou/molgap-edgestate-jk-readout-r5-qm9-validation"
    )
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "true"
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pairgps-r2-qm9-rwse-prep",
        "kaseichou/molgap-pure2d-r3-qm9-validation",
        "kaseichou/molgap-pure2d-r3-tensor-acceptance",
    ]


def test_r5_acceptance_is_inference_free_and_hashes_artifacts() -> None:
    tree = ast.parse(ACCEPTANCE.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "torch" not in imports
    source = ACCEPTANCE.read_text(encoding="utf-8")
    assert '"model_inference_executed": False' in source
    assert '"test_role_read": False' in source
    assert "sha256(artifact)" in source
