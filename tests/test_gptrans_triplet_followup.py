"""Synthetic addon algebra/checkpoint probes; no molecular inference/training."""
import copy
from pathlib import Path
import ast
import pytest
import torch
from molgap.gptrans_triplet_followup import MODES, PARAMETERS, PrivateBranchDrop, product_direction
from molgap.gptrans_triplet_communication import aggregate_direction
from molgap.pcqm_gptrans_v4 import capacity_module, _ema_decay
from molgap.gptrans_author_screen import validate_arm_allocation
from molgap.experiment_spec import ADDONS


def test_registered_inventory():
    assert PARAMETERS == {MODES[0]: 5880961, MODES[1]: 5885185}
    assert validate_arm_allocation({"arms": {m: {} for m in MODES}}) == MODES
    for mode in MODES:
        assert ADDONS[(mode, "1")].single_addon
        assert capacity_module(mode).__name__ == "molgap.gptrans_triplet_followup"
        assert _ema_decay(mode) == .999


def test_private_dropout_resume_eval_and_parent_rng_unchanged():
    block = PrivateBranchDrop(3)
    x = torch.ones(256, 2, 2, 2, requires_grad=True)
    global_state = torch.get_rng_state().clone()
    first = block(x)
    checkpoint = copy.deepcopy(block.state_dict())
    second = block(x)
    resumed = PrivateBranchDrop(3)
    resumed.load_state_dict(checkpoint)
    torch.testing.assert_close(second, resumed(x))
    assert torch.equal(global_state, torch.get_rng_state())
    assert not torch.equal(first, second)
    assert .7 < first.mean() < 1.3
    first.sum().backward()
    assert torch.isfinite(x.grad).all()
    block.eval()
    before = copy.deepcopy(block.state_dict())
    assert block(x) is x and torch.equal(block.rng_state, before["rng_state"])
    assert block.to(dtype=torch.float64).rng_state.device.type == "cpu"


@pytest.mark.parametrize("outward", [False, True])
def test_product_reduces_to_aggregate_and_permutation_padding(outward):
    g = torch.Generator().manual_seed(7)
    value = torch.randn(1, 3, 3, 2, 8, generator=g, requires_grad=True)
    bias, gate = (torch.randn(1, 3, 3, 2, generator=g) for _ in range(2))
    valid = torch.ones(1, 3, 3, dtype=torch.bool)
    actual = product_direction(value, torch.ones_like(value), bias, gate, valid, outward=outward)
    torch.testing.assert_close(actual, aggregate_direction(value, bias, gate, valid, outward=outward))
    operand = torch.randn(value.shape, generator=g, requires_grad=True)
    expected = product_direction(value, operand, bias, gate, valid, outward=outward)
    p = torch.tensor([0, 2, 1])
    permute = lambda x: x[:, p][:, :, p]
    torch.testing.assert_close(product_direction(permute(value), permute(operand), permute(bias), permute(gate), permute(valid), outward=outward), permute(expected))
    expected.square().sum().backward()
    assert torch.isfinite(operand.grad).all() and operand.grad.abs().sum() > 0
    zeros = torch.zeros_like(valid)
    assert not product_direction(value, operand, bias, gate, zeros, outward=outward).any()


def test_initial_freeze_never_executes_model():
    tree = ast.parse(Path("src/molgap/gptrans_triplet_followup.py").read_text())
    freeze = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "freeze_initial")
    assert "construct(" not in ast.unparse(freeze)
    assert "OGBGPTransTiny" not in ast.unparse(freeze)
