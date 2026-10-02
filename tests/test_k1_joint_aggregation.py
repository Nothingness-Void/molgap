"""Synthetic qualification only; no PCQM rows or protected roles."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from molgap.k1_joint_aggregation import (
    BoundedJointAggregation, attach_k1_joint_aggregation,
    backbone_state_dict, create_k1_joint_encoder,
)
from molgap.qm9_neural_atom import make_encoder


def _active(mode="ssma"):
    torch.manual_seed(17)
    addon = BoundedJointAggregation(mode=mode, chunk_nodes=2)
    torch.nn.init.normal_(addon.return_projection.weight, std=0.02)
    return addon


def test_bounded_degrees_and_permutation_invariance():
    addon = _active()
    index = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])
    messages = torch.randn(15, 192, requires_grad=True)
    update = addon(messages, index, 7)
    perm = torch.randperm(15)
    torch.testing.assert_close(update, addon(messages[perm], index[perm], 7), atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(update[[0, 5, 6]]) == 0
    assert torch.count_nonzero(update[1:5]) > 0
    update.square().sum().backward()
    assert torch.isfinite(messages.grad).all()
    assert torch.count_nonzero(messages.grad[index == 5]) == 0
    assert torch.count_nonzero(messages.grad[index <= 4]) > 0


@pytest.mark.parametrize("mode", ["ssma", "independent_sum"])
@pytest.mark.parametrize("scale", [0.0, 1e-10, 1.0, 1e12])
def test_finite_forward_backward(mode, scale):
    addon = _active(mode)
    messages = (torch.randn(8, 192) * scale).requires_grad_()
    update = addon(messages, torch.tensor([0, 1, 1, 2, 2, 2, 2, 3]), 4)
    assert torch.isfinite(update).all()
    update.square().mean().backward()
    assert torch.isfinite(messages.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in addon.parameters() if p.grad is not None)


def test_empty_and_high_degree_are_zero_without_dropping_original_sum():
    addon = _active()
    assert torch.count_nonzero(addon(torch.empty(0, 192), torch.empty(0, dtype=torch.long), 3)) == 0
    messages = torch.randn(5, 192)
    index = torch.zeros(5, dtype=torch.long)
    original_sum = messages.sum(dim=0, keepdim=True)
    torch.testing.assert_close(original_sum + addon(messages, index, 1), original_sum, rtol=0, atol=0)


def test_fixed_affine_signal_matches_source_operator_and_storage_bound():
    addon = _active()
    messages = torch.randn(2, 4, 192)
    latent = addon.project(addon.message_norm(messages))
    dense = torch.nn.functional.linear(latent, addon.affine_weight, addon.affine_bias)
    torch.testing.assert_close(addon._signals(messages).flatten(start_dim=2), dense, rtol=0, atol=0)
    assert addon.stored_numel() == addon.trainable_numel() + 1265 * 65
    assert addon.stored_numel() < 130_000


def test_independent_capacity_control_shares_identical_state():
    joint, independent = _active(), _active("independent_sum")
    for key, value in joint.state_dict().items():
        torch.testing.assert_close(value, independent.state_dict()[key], rtol=0, atol=0)
    messages = torch.randn(4, 192)
    index = torch.zeros(4, dtype=torch.long)
    assert not torch.allclose(joint(messages, index, 1), independent(messages, index, 1))


def test_attach_preserves_rng_backbone_and_exact_initial_forward():
    torch.manual_seed(19)
    model = make_encoder("neural_atom_k1").eval()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.zeros(6, 9, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    edge_attr = torch.zeros(6, 3, dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    rwse = torch.zeros(6, 16)
    with torch.no_grad():
        expected = model(x, edge_index, edge_attr, batch, rwse)
    rng = torch.random.get_rng_state().clone()
    addon = attach_k1_joint_aggregation(model)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert addon.training is False
    assert set(backbone_state_dict(model)) == set(state)
    for key, value in backbone_state_dict(model).items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with torch.no_grad():
        actual = model(x, edge_index, edge_attr, batch, rwse)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="already attached"):
        attach_k1_joint_aggregation(model)


def test_hook_uses_actual_gated_messages_and_retains_high_degree_sum():
    model = make_encoder("neural_atom_k1").eval()
    conv = model.local_blocks[5].conv
    x = torch.randn(7, 192)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 0, 2]])
    edge_attr = torch.randn(7, conv.edge_dim)
    with torch.no_grad():
        expected = conv(x, edge_index, edge_attr)
    addon = attach_k1_joint_aggregation(model)
    torch.nn.init.normal_(addon.return_projection.weight, std=0.02)
    captured = []
    handle = conv.register_message_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
    with torch.no_grad():
        actual = conv(x, edge_index, edge_attr)
        delta = addon(captured[0], edge_index[1], 7)
    handle.remove()
    torch.testing.assert_close(actual, expected + delta, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)


def test_create_loads_frozen_backbone_separately():
    original = make_encoder("neural_atom_k1")
    model = create_k1_joint_encoder(backbone_state=original.state_dict())
    for key, value in backbone_state_dict(model).items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
