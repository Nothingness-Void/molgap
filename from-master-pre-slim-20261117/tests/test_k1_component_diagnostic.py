"""Synthetic restoration and message/root separation, without scientific roles."""
import pytest
import torch
from torch import nn
from torch_geometric.nn import ResGatedGraphConv

from molgap.k1_component_diagnostic import component_intervention, hook_inventory


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_emb = nn.Linear(2, 2)
        self.edge_emb = nn.Linear(1, 2)
        self.rwse_encoder = nn.Linear(2, 2)
        self.local_blocks = nn.ModuleList([nn.Module() for _ in range(9)])
        for block in self.local_blocks:
            block.conv = ResGatedGraphConv(2, 2, edge_dim=2)
            block.mlp = nn.Linear(2, 2)
        self.edge_updates = nn.ModuleList([EdgeUpdate() for _ in range(9)])


class EdgeUpdate(nn.Module):
    def forward(self, hidden, edge_index, state):
        return state + 1


def test_atom_half_restores_state_and_hooks_after_failure():
    model = Tiny().eval()
    value = torch.tensor([[1., 2.]])
    expected = model.node_emb(value)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    hooks = hook_inventory(model)
    with pytest.raises(RuntimeError, match="interrupted"):
        with component_intervention(model, "atom"):
            torch.testing.assert_close(model.node_emb(value), expected / 2)
            raise RuntimeError("interrupted")
    assert hook_inventory(model) == hooks
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    torch.testing.assert_close(model.node_emb(value), expected)


def test_neighbor_half_preserves_root_projection_and_bias():
    model = Tiny().eval()
    conv = model.local_blocks[0].conv
    hidden = torch.tensor([[1., 2.], [3., 4.]])
    edges = torch.tensor([[0, 1], [1, 0]])
    state = torch.tensor([[.2, .3], [.4, .5]])
    original = conv(hidden, edges, state)
    root = conv.lin_skip(hidden) + conv.bias
    with component_intervention(model, "message", (0,)):
        changed = conv(hidden, edges, state)
    torch.testing.assert_close(changed, root + (original - root) / 2)
    torch.testing.assert_close(conv(hidden, edges, state), original)


def test_memory_reset_runs_update_with_initial_bond_each_time():
    model = Tiny().eval()
    hidden = torch.ones((2, 2))
    edges = torch.tensor([[0], [1]])
    bond = torch.ones((1, 1))
    hooks = hook_inventory(model)
    with component_intervention(model, "edge_memory"):
        initial = model.edge_emb(bond)
        first = model.edge_updates[0](hidden, edges, initial)
        second = model.edge_updates[1](hidden, edges, first)
        third = model.edge_updates[2](hidden, edges, second)
        torch.testing.assert_close(first, initial + 1)
        torch.testing.assert_close(second, initial + 1)
        torch.testing.assert_close(third, initial + 1)
    assert hook_inventory(model) == hooks


def test_train_mode_and_unknown_case_rejected_without_leaking_hooks():
    model = Tiny()
    with pytest.raises(ValueError, match="eval"):
        with component_intervention(model, "atom"):
            pass
    model.eval()
    hooks = hook_inventory(model)
    with pytest.raises(ValueError, match="Unknown"):
        with component_intervention(model, "unknown"):
            pass
    assert hook_inventory(model) == hooks
