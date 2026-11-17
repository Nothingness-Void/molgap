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
    / "kaggle_directed_edge_state_r10"
    / "gpu_validation"
    / "run_validation.py"
)
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_directed_edge_state_r10.py"


def assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment {name} in {path}")


def test_r10_uses_non_backtracking_directed_bond_flow() -> None:
    source = GPS.read_text(encoding="utf-8")
    assert "class DirectedEdgeStateStructuralGPSWrapper" in source
    assert "incoming_sum.index_add_(0, target, edge_state)" in source
    assert "non_backtracking = incoming_sum[source] - edge_state[reverse_edge]" in source
    assert "nn.init.zeros_(self.incoming.weight)" in source
    assert "torch.searchsorted(sorted_keys, reverse_keys)" in source
    assert '"directed_edge_state_structural_gps"' in QM9.read_text(encoding="utf-8")


def test_r10_parameter_contract_fits_budget() -> None:
    added = 9 * 64 * 64
    assert 4_739_651 + added == 4_776_515
    assert assignment_literal(RUNNER, "EXPECTED_PARAMETER_COUNT") == 4_776_515
    assert assignment_literal(RUNNER, "EXPECTED_SOURCE_COMMIT") == (
        "06bf8f439783cced552760b873e1702a0098c802"
    )


def test_r10_remote_contract_is_validation_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "evaluate_test=False" in source
    assert '"test_role_read": False' in source
    assert '"reverse_edge_coverage": True' in source
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "true"
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pairgps-r2-qm9-rwse-prep",
        "kaseichou/molgap-pure2d-r3-qm9-validation",
        "kaseichou/molgap-pure2d-r3-tensor-acceptance",
    ]


def test_r10_acceptance_imports_no_model_runtime() -> None:
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
