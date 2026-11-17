from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPS = ROOT / "src" / "molgap" / "gps.py"
QM9 = ROOT / "src" / "molgap" / "qm9_screen.py"
EXPERIMENT = ROOT / "experiments" / "top20_architecture_qm9"
RUNNER = (
    EXPERIMENT
    / "kaggle_edge_conditioned_r6"
    / "gpu_validation"
    / "run_validation.py"
)
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_edge_conditioned_r6.py"


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


def test_r6_model_fuses_edges_before_each_gps_block() -> None:
    source = GPS.read_text(encoding="utf-8")
    assert "class EdgeConditionedStructuralGPSWrapper" in source
    assert "h = self._condition_nodes_from_edges(h, edge_index, edge_state)" in source
    assert "edge_context.index_add_(0, target, edge_state)" in source
    assert "self.edge_to_node_film" in source
    assert "nn.init.zeros_(self.edge_to_node_film.weight)" in source
    assert "return h + conditioned" in source
    config = dict_entry_literal(QM9, "ENCODER_CONFIGS", "edge_conditioned_structural_gps")
    assert config["kind"] == "structural_topology"
    assert config["hidden_channels"] == 192
    assert config["num_layers"] == 9
    assert config["edge_state_channels"] == 64
    assert config["batch_size"] == 48
    assert config["amp"] is False


def test_r6_parameter_contract_fits_budget() -> None:
    base_parameters = 4_739_651
    hidden = 192
    edge = 64
    added = 2 * edge + 2 * hidden + edge * (2 * hidden) + 2 * hidden
    assert added == 25_472
    assert base_parameters + added == 4_765_123
    assert base_parameters + added <= 4_800_000
    assert assignment_literal(RUNNER, "EXPECTED_PARAMETER_COUNT") == 4_765_123


def test_r6_remote_contract_is_validation_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "evaluate_test=False" in source
    assert '"test_role_read": False' in source
    assert "molgap-pure2d-r3-tensor-acceptance-v2" in source
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-edge-conditioned-r6-qm9-validation"
    assert metadata["enable_gpu"] == "true"
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pairgps-r2-qm9-rwse-prep",
        "kaseichou/molgap-pure2d-r3-qm9-validation",
        "kaseichou/molgap-pure2d-r3-tensor-acceptance",
    ]


def test_r6_acceptance_imports_no_model_runtime() -> None:
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
