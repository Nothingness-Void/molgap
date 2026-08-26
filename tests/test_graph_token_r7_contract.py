from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPS = ROOT / "src" / "molgap" / "gps.py"
QM9 = ROOT / "src" / "molgap" / "qm9_screen.py"
EXPERIMENT = ROOT / "experiments" / "top20_architecture_qm9"
RUNNER = EXPERIMENT / "kaggle_graph_token_r7" / "gpu_validation" / "run_validation.py"
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_graph_token_r7.py"


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


def test_r7_model_has_recurrent_graph_to_node_flow() -> None:
    source = GPS.read_text(encoding="utf-8")
    assert "class GraphTokenStructuralGPSWrapper" in source
    assert "graph_state = self._initialize_graph_state(h, batch)" in source
    assert "h = self._condition_nodes_from_graph(h, batch, graph_state)" in source
    assert "graph_state = self._update_graph_state(h, batch, graph_state)" in source
    assert "self.graph_token.expand(num_graphs, -1)" in source
    assert "return h + broadcast[batch]" in source
    assert "nn.init.zeros_(self.token_to_node.weight)" in source
    config = dict_entry_literal(QM9, "ENCODER_CONFIGS", "graph_token_structural_gps")
    assert config["kind"] == "structural_topology"
    assert config["hidden_channels"] == 192
    assert config["num_layers"] == 9
    assert config["token_channels"] == 16
    assert config["batch_size"] == 48
    assert config["amp"] is False


def test_r7_parameter_contract_fits_budget() -> None:
    base_parameters = 4_739_651
    hidden = 192
    token_channels = 16
    added = (
        hidden
        + 2 * (2 * hidden)
        + (2 * hidden) * token_channels
        + token_channels
        + token_channels * hidden
        + hidden
        + 2 * hidden
        + hidden * hidden
        + hidden
    )
    assert added == 47_824
    assert base_parameters + added == 4_787_475
    assert base_parameters + added <= 4_800_000
    assert assignment_literal(RUNNER, "EXPECTED_PARAMETER_COUNT") == 4_787_475


def test_r7_remote_contract_is_validation_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "evaluate_test=False" in source
    assert '"test_role_read": False' in source
    assert "molgap-pure2d-r3-tensor-acceptance-v2" in source
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-graph-token-r7-qm9-validation"
    assert metadata["enable_gpu"] == "true"
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pairgps-r2-qm9-rwse-prep",
        "kaseichou/molgap-pure2d-r3-qm9-validation",
        "kaseichou/molgap-pure2d-r3-tensor-acceptance",
    ]


def test_r7_acceptance_imports_no_model_runtime() -> None:
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
