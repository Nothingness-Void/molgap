"""Static/tensor-only addon checks; no local model construction or inference."""
import ast
from pathlib import Path
import pytest
import torch

from molgap.gptrans_local_relation import MODES, PARAMETERS, INSERTIONS, valid_pairs, bond_pair_update, configuration
from molgap.pcqm_gptrans_v4 import capacity_module, _ema_decay
from molgap.gptrans_author_screen import validate_arm_allocation
from molgap.experiment_spec import ADDONS


def test_registration_counts_and_independent_allocation():
    assert PARAMETERS[MODES[0]] == 5_888_225
    assert PARAMETERS[MODES[1]] == 5_896_161
    for mode in MODES:
        assert ADDONS[(mode, "1")].single_addon
        assert capacity_module(mode).__name__ == "molgap.gptrans_local_relation"
        assert _ema_decay(mode) == .999
    assert validate_arm_allocation({"arms": {mode: {} for mode in MODES}}) == MODES
    with pytest.raises(ValueError):
        configuration("degree_pair_transition_ema999")


def test_virtual_scope_and_padding_mask_tensor_only():
    mask = valid_pairs(torch.tensor([[[[False, False, True]]]]))
    assert mask[0, 0, 0] and mask[0, 0, 1] and mask[0, 1, 0]
    assert not mask[0, 2].any() and not mask[0, :, 2].any()
    assert configuration(MODES[0])["changed_layers"] == list(INSERTIONS)


def test_sparse_bond_update_duplicate_sum_and_virtual_exclusion():
    pair = torch.zeros(1, 32, 4, 4)
    edges = (torch.tensor([0, 0, 0]), torch.tensor([1, 1, 2]), torch.tensor([2, 2, 1]))
    actual = bond_pair_update(pair, edges, torch.ones(3, 32))
    assert torch.equal(actual[0, :, 1, 2], torch.full((32,), 2.))
    assert torch.equal(actual[0, :, 2, 1], torch.ones(32))
    assert not actual[:, :, 0, :].any() and not actual[:, :, :, 0].any()
    assert not pair.any()


def test_same_block_consumption_and_tensor_only_initial_builder():
    source = Path("src/molgap/gptrans_local_relation.py").read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    freeze = ast.unparse(functions["freeze_initial"])
    assert "construct(" not in freeze and "OGBGPTransTiny" not in freeze
    assert "PARENT_TENSOR_SHA256" in freeze
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BondPairReturnBlock")
    body = ast.unparse(block)
    assert "super().forward(node, updated, padding, edges)" in body
    assert "self.pair_return(hidden)" in body


def test_connectivity_checked_after_existing_optimizer_warmup():
    source = Path("src/molgap/pcqm_gptrans_v4.py").read_text()
    tree = ast.parse(source)
    preflight = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_preflight")
    body = ast.unparse(preflight)
    assert body.index("PREFLIGHT_MEASURED_STEPS") < body.index("verify_connected_preflight")
    assert "'relation_connectivity': relation_connectivity" in body
