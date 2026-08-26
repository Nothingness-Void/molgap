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
    / "kaggle_sparse_path_attention_r9"
    / "gpu_validation"
    / "run_validation.py"
)
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_sparse_path_attention_r9.py"


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


def test_r9_changes_attention_without_replacing_bond_edges() -> None:
    source = GPS.read_text(encoding="utf-8")
    assert "class SparsePathAttentionStructuralGPSWrapper" in source
    assert "h = conv(h, edge_index, batch, edge_attr=edge_state)" in source
    assert "h = self.path_attention(" in source
    assert "nn.init.zeros_(self.output.weight)" in source
    assert "self.distance_bias(distance - 1)" in source
    cache_source = QM9.read_text(encoding="utf-8")
    assert "graph.multihop_edge_index = graph.edge_index.clone()" in cache_source
    assert "graph.edge_index = bond_edge_index" in cache_source
    assert "graph.edge_attr = bond_edge_attr" in cache_source


def test_r9_parameter_and_training_contract() -> None:
    config = dict_entry_literal(
        QM9, "ENCODER_CONFIGS", "sparse_path_attention_structural_gps"
    )
    assert config["kind"] == "structural_multihop"
    assert config["path_attention_rank"] == 16
    assert config["path_max_distance"] == 4
    assert config["batch_size"] == 48
    assert config["amp"] is False
    added = 4 * 192 * 16 + 2 * 192 + 4
    assert 4_739_651 + added == 4_752_327
    assert assignment_literal(RUNNER, "EXPECTED_PARAMETER_COUNT") == 4_752_327


def test_r9_remote_contract_is_validation_only() -> None:
    assert assignment_literal(RUNNER, "EXPECTED_SOURCE_COMMIT") == (
        "0ffdef4ee046e6ffdba256d6c1b05758c61f2416"
    )
    assert assignment_literal(RUNNER, "EXPECTED_MULTIHOP_SHA256") == (
        "0ea8a0e27790b5bbdb038365d681b5f48974da959a1a8890e0ca1ef24a339dd3"
    )
    source = RUNNER.read_text(encoding="utf-8")
    assert "evaluate_test=False" in source
    assert '"test_role_read": False' in source
    assert '"local_edge_feature_dim": int(batch.edge_attr.shape[1])' in source
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "true"
    assert metadata["kernel_sources"][-1] == (
        "kaseichou/molgap-multihop-edgestate-r8-qm9-prep"
    )


def test_r9_acceptance_imports_no_model_runtime() -> None:
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
