from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QM9 = ROOT / "src" / "molgap" / "qm9_screen.py"
EXPERIMENT = ROOT / "experiments" / "top20_architecture_qm9"
PREP = EXPERIMENT / "kaggle_multihop_edge_state_r8" / "cpu_prep" / "run_prep.py"
METADATA = PREP.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_multihop_edge_state_r8_prep.py"
GPU_RUNNER = (
    EXPERIMENT
    / "kaggle_multihop_edge_state_r8"
    / "gpu_validation"
    / "run_validation.py"
)
GPU_METADATA = GPU_RUNNER.with_name("kernel-metadata.json")
GPU_ACCEPTANCE = EXPERIMENT / "accept_multihop_edge_state_r8.py"


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


def test_r8_candidate_changes_only_sparse_local_edge_contract() -> None:
    config = dict_entry_literal(
        QM9, "ENCODER_CONFIGS", "multihop_edge_state_structural_gps"
    )
    assert config == {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "model_edge_dim": 8,
        "multihop_max_distance": 4,
        "batch_size": 48,
        "amp": False,
    }
    source = QM9.read_text(encoding="utf-8")
    assert '"multihop_edge_state_structural_gps": EdgeStateStructuralGPSWrapper' in source
    assert "edge_dim = int(config.pop(\"model_edge_dim\", edge_dim))" in source


def test_r8_parameter_contract_fits_budget() -> None:
    accepted_r3_parameters = 4_739_651
    added_input_weights = (8 - 4) * 64
    assert accepted_r3_parameters + added_input_weights == 4_739_907
    assert accepted_r3_parameters + added_input_weights <= 4_800_000


def test_r8_cache_builder_is_train_validation_only_and_resumable() -> None:
    source = QM9.read_text(encoding="utf-8")
    start = source.index("def build_qm9_multihop_screen_cache")
    end = source.index("def attach_accepted_qm9_multihop", start)
    builder = source[start:end]
    assert "np.concatenate((split.train, split.validation))" in builder
    assert "split.test" not in builder
    assert '"test_role_read": False' in builder
    assert "_atomic_torch_save(part_path, payloads)" in builder
    assert "_atomic_json(\n            paths[\"progress\"]" in builder


def test_r8_cpu_prep_uses_no_gpu_and_no_test_role() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-multihop-edgestate-r8-qm9-prep"
    assert metadata["enable_gpu"] == "false"
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pairgps-r2-qm9-rwse-prep"
    ]
    source = PREP.read_text(encoding="utf-8")
    assert "source_cache_dir=source_cache" in source
    assert '"gpu_used": False' in source
    assert '"test_role_read": False' in source


def test_r8_local_acceptance_imports_no_model_runtime() -> None:
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


def assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment {name} in {path}")


def test_r8_gpu_contract_is_frozen_and_validation_only() -> None:
    assert assignment_literal(GPU_RUNNER, "EXPECTED_PARAMETER_COUNT") == 4_739_907
    assert assignment_literal(GPU_RUNNER, "EXPECTED_SOURCE_COMMIT") == (
        "56abea806ff88778bcbb847d569a266da074eee1"
    )
    assert assignment_literal(GPU_RUNNER, "EXPECTED_MULTIHOP_SHA256") == (
        "0ea8a0e27790b5bbdb038365d681b5f48974da959a1a8890e0ca1ef24a339dd3"
    )
    source = GPU_RUNNER.read_text(encoding="utf-8")
    assert "evaluate_test=False" in source
    assert "multihop_cache_dir=multihop_cache" in source
    assert '"test_role_read": False' in source
    metadata = json.loads(GPU_METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == (
        "kaseichou/molgap-multihop-edgestate-r8-qm9-validation"
    )
    assert metadata["enable_gpu"] == "true"
    assert metadata["kernel_sources"][-1] == (
        "kaseichou/molgap-multihop-edgestate-r8-qm9-prep"
    )


def test_r8_gpu_acceptance_imports_no_model_runtime() -> None:
    tree = ast.parse(GPU_ACCEPTANCE.read_text(encoding="utf-8"))
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
    source = GPU_ACCEPTANCE.read_text(encoding="utf-8")
    assert '"model_inference_executed": False' in source
    assert '"test_role_read": False' in source
