import copy

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from molgap.k1_execution_layout import cpu_layout_context, layout_from_cpu_batch
from molgap.qm9_neural_atom import _NeuralAtomMixerFactory


def batch_fixture():
    return Batch.from_data_list([Data(x=torch.zeros(n, 9)) for n in (2, 5, 3)])


def test_dense_layout_has_identical_packing_unpacking_and_input_gradient():
    batch = batch_fixture()
    layout = layout_from_cpu_batch(batch, "cpu")
    h = torch.randn(10, 7, requires_grad=True)
    baseline, mask = to_dense_batch(h, batch.batch)
    dense, valid = layout.pack(h, batch.batch)
    assert torch.equal(baseline, dense)
    assert torch.equal(mask, valid)
    returned = torch.randn_like(dense, requires_grad=True)
    assert torch.equal(returned[mask], layout.unpack(returned))
    a = torch.autograd.grad(returned[mask].square().sum(), returned, retain_graph=True)[0]
    b = torch.autograd.grad(layout.unpack(returned).square().sum(), returned)[0]
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="differ"):
        layout.pack(h, batch.batch.clone())


def test_layout_rejects_empty_or_unordered_membership():
    batch = batch_fixture()
    batch.batch[0] = 2
    with pytest.raises(ValueError, match="membership"):
        layout_from_cpu_batch(batch, "cpu")
    batch = batch_fixture()
    batch.ptr[1] = 0
    with pytest.raises(ValueError, match="boundaries"):
        layout_from_cpu_batch(batch, "cpu")


def test_mixer_forward_gradient_and_rng_exact_and_context_cleanup():
    torch.manual_seed(42)
    mixer = _NeuralAtomMixerFactory.make(192, 64, 4, 4, 1, 0.05).train()
    torch.nn.init.normal_(mixer.return_projection.weight, std=0.01)
    candidate = copy.deepcopy(mixer)
    wrapper = torch.nn.Module()
    wrapper.pooling = "mean"
    wrapper.neural_atom_mixers = torch.nn.ModuleDict({"3": candidate})
    batch = batch_fixture()
    hidden = torch.randn(10, 192, requires_grad=True)
    other = hidden.detach().clone().requires_grad_(True)
    rng = torch.get_rng_state()
    a = mixer(hidden, batch.batch)
    a.square().mean().backward()
    baseline_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    with cpu_layout_context(wrapper, batch, "cpu"):
        b = candidate(other, batch.batch)
        b.square().mean().backward()
    assert torch.equal(a, b)
    assert torch.equal(hidden.grad, other.grad)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    for p, q in zip(mixer.parameters(), candidate.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)
    assert not hasattr(candidate, "_execution_dense_layout")
    assert not hasattr(wrapper, "_execution_graph_count")
    with pytest.raises(RuntimeError):
        with cpu_layout_context(wrapper, batch, "cpu"):
            raise RuntimeError("test cleanup")
    assert not hasattr(candidate, "_execution_dense_layout")
    assert set(mixer.state_dict()) == set(candidate.state_dict())
