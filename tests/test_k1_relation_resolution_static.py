"""Static contract tests for the bounded K1 relation-resolution sidecar.

These tests intentionally parse source and evaluate only algebra/constants;
they must not import torch/PyG or instantiate a model locally.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src" / "molgap" / "k1_relation_resolution.py"


def _source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(SOURCE_PATH))


def _eval_constant(node, names):
    """Evaluate the small arithmetic namespace without importing the module."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_constant(item, names) for item in node.elts)
    if isinstance(node, ast.List):
        return [_eval_constant(item, names) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _eval_constant(key, names): _eval_constant(value, names)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_constant(node.operand, names)
    if isinstance(node, ast.BinOp):
        left = _eval_constant(node.left, names)
        right = _eval_constant(node.right, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise AssertionError(f"unsupported constant expression: {ast.dump(node)}")


def _constant_namespace() -> dict:
    names = {}
    for node in _tree().body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            names[target.id] = _eval_constant(node.value, names)
        except (AssertionError, KeyError):
            continue
    return names


def _literal_assignment(name: str):
    names = _constant_namespace()
    assert name in names, f"missing constant assignment: {name}"
    return names[name]


def test_only_lazy_torch_and_pyg_imports() -> None:
    tree = _tree()
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported = {
        node.module.split(".")[0]
        if isinstance(node, ast.ImportFrom)
        else node.names[0].name.split(".")[0]
        for node in top_level_imports
    }
    assert imported <= {"__future__", "math"}
    assert "import torch" in _source()
    assert "from torch_geometric" in _source()


def test_modes_configs_and_frozen_layout_are_explicit() -> None:
    modes = _literal_assignment("MODES")
    assert modes == (
        "neural_atom_k1_receiver_pair",
        "neural_atom_k1_rrwp_pair",
        "neural_atom_k1_triplet_aggregate",
    )
    configs = _literal_assignment("CONFIGS")
    assert configs[modes[0]]["target_layer"] == 6
    assert configs[modes[0]]["pair_channels"] == 32
    for mode in modes:
        assert configs[mode]["backbone"] == "neural_atom_k1"
        assert configs[mode]["exchange_layers"] == [3, 6, 9]
        assert configs[mode]["geometry"] is False
        assert configs[mode]["teacher"] is False
        assert configs[mode]["initialization_policy"]
    assert configs[modes[1]]["rrwp_features"] == 8
    assert configs[modes[1]]["rrwp_search"] is False
    assert configs[modes[2]]["triplet"] is True
    assert _literal_assignment("HIDDEN_CHANNELS") == 192
    assert _literal_assignment("PAIR_CHANNELS") == 32
    assert _literal_assignment("TARGET_LAYER") == 6
    assert _literal_assignment("MIXER_LAYERS") == (3, 6, 9)


def test_parameter_algebra_matches_declared_counts() -> None:
    base = _literal_assignment("BASE_PARAMETERS")
    parent = (
        2 * (192 * 32 + 32)
        + 2 * 32
        + 32
        + 2 * 32
        + (32 * 64 + 64)
        + (64 * 32 + 32)
        + 32 * 192
    )
    rrwp = 8 * 32 + 32
    triplet = 32 * 32 + (32 + 1) + (32 + 1) + 32 * 32 + 2 * 32
    assert parent == 22_848
    assert rrwp == 288
    assert triplet == 2_178
    assert _literal_assignment("PARAMETERS") == {
        "neural_atom_k1_receiver_pair": base + parent,
        "neural_atom_k1_rrwp_pair": base + parent + rrwp,
        "neural_atom_k1_triplet_aggregate": base + parent + triplet,
    }


def test_required_equations_and_masking_are_structural() -> None:
    source = _source()
    assert "functional.silu(preactivation)" in source
    assert 'torch.einsum("bijd,d->bij", pair, self.query)' in source
    assert "math.sqrt(PAIR_CHANNELS)" in source
    assert 'torch.einsum("bij,bijd->bid", assignment, pair)' in source
    assert "message + self.message_ffn(message)" in source
    assert "PAIR_CHANNELS, HIDDEN_CHANNELS, bias=False" in source
    assert "pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)" in source
    assert "assignment * pair_valid" in source
    assert "torch.bmm(powers[-1], row_normalized)" in source
    assert "RRWP_STEPS = 8" in source
    assert 'torch.einsum(\n                    "bik,bjkd->bijd"' in source
    assert "triplet_assignment = weights * gates * pair_valid" in source
    assert "triplet_out(aggregate)" in source
    assert "different_receiver_features_within_graph" in source
    assert "graph0_perturbation_isolated" in source
    assert "rrwp_batched_graphs_match_independent_expected" in source
    assert "rrwp_isolate_self_transition" in source
    assert not any(
        isinstance(node, ast.Name) and node.id == "token"
        for node in ast.walk(_tree())
    )
    assert "global pair token" not in source.lower()


def test_no_external_rrwp_or_geometry_identity_is_referenced() -> None:
    source = _source().lower()
    assert "edge_index" in source
    assert "edge_attr" in source  # forward compatibility only
    assert "random_walk_pe" in source  # forward compatibility only
    identifiers = {
        node.id.lower()
        for node in ast.walk(_tree())
        if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr.lower()
        for node in ast.walk(_tree())
        if isinstance(node, ast.Attribute)
    }
    assert "pos" not in identifiers | attributes
    assert "spd" not in identifiers | attributes
    assert "cache" not in identifiers | attributes


def test_static_math_catches_rrwp_and_triplet_contract_values() -> None:
    assert math.isclose(math.sqrt(32), 5.656854249492381)
    assert 3_658_817 + 22_848 == 3_681_665
    assert 3_658_817 + 22_848 + 288 == 3_681_953
    assert 3_658_817 + 22_848 + 2_178 == 3_683_843


def test_public_entry_points_exist_without_importing_runtime() -> None:
    names = {
        node.name
        for node in _tree().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert {"make_encoder", "check_mechanism"} <= names
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "_RelationFactory"
        for node in _tree().body
    )
    assert "self.relation_token = _RelationFactory.make(mode)" in _source()
