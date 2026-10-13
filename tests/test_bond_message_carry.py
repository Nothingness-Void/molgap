"""Synthetic algebra only; no molecular model forward, training or inference."""
import pytest
import torch

from molgap.bond_message_carry import BondMessageCarry


def test_zero_carry_parent_identity_inventory_and_rng():
    rng = torch.get_rng_state().clone()
    carry = BondMessageCarry()
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in carry.parameters()) == 11
    previous = None
    for depth in range(1, 13):
        messages = torch.arange(192, dtype=torch.float32).reshape(3, 64) + depth
        returned = carry(messages, previous, depth=depth)
        assert torch.equal(returned, messages)
        previous = returned


def test_bounded_signed_carry_and_connected_gradients():
    carry = BondMessageCarry(layers=3, channels=2)
    messages = torch.ones(4, 2, requires_grad=True)
    previous = torch.full((4, 2), 2., requires_grad=True)
    with torch.no_grad():
        carry.logits[0] = .4
        carry.logits[1] = -20
    result = carry(messages, previous, depth=2)
    torch.testing.assert_close(result, messages + torch.tensor(.4).tanh() * previous)
    result.sum().backward()
    assert messages.grad is not None and previous.grad is not None
    assert carry.logits.grad[0] > 0 and torch.isfinite(carry.logits.grad).all()
    negative = carry(messages, previous, depth=3)
    assert torch.all((negative - messages).abs() <= previous.abs())


def test_edge_permutation_no_mutation_and_no_cross_batch_state():
    carry = BondMessageCarry(layers=2, channels=2)
    with torch.no_grad():
        carry.logits.fill_(.7)
    current = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    previous = current + 10
    before = previous.clone()
    permutation = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(carry(current[permutation], previous[permutation], depth=2),
                               carry(current, previous, depth=2)[permutation])
    assert torch.equal(previous, before)
    assert carry(current, None, depth=1) is current
    assert set(carry.state_dict()) == {"logits"}
    with pytest.raises(ValueError, match="Every batch"):
        carry(current, previous, depth=1)


@pytest.mark.parametrize("depth", [0, 13, True])
def test_invalid_depth_rejected(depth):
    with pytest.raises(ValueError, match="Depth"):
        BondMessageCarry()(torch.ones(2, 64), None, depth=depth)


def test_shape_dtype_and_empty_edges():
    carry = BondMessageCarry(layers=2, channels=2)
    x = torch.ones(2, 2)
    for previous in (None, torch.ones(3, 2), x.double()):
        with pytest.raises(ValueError, match="Predecessor"):
            carry(x, previous, depth=2)
    empty = torch.empty(0, 2)
    assert carry(empty, None, depth=1).shape == (0, 2)
    assert carry(empty, empty, depth=2).shape == (0, 2)
