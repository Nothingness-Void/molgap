"""CPU structural transfer checks; no accelerator or molecular role data."""
import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap import k1_screen_training as owner


def synthetic_batch():
    return Batch.from_data_list([Data(x=torch.zeros(n, 9, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, 3, dtype=torch.long), random_walk_pe=torch.zeros(n, 16),
        y=torch.tensor([float(n)]), source_idx=torch.tensor([index]),
        chemical_fingerprint=torch.ones(1, 4)) for index, n in enumerate((1, 3))])


@pytest.mark.parametrize("training,auxiliary", [(True, False), (True, True), (False, False)])
def test_only_declared_model_and_loss_fields_are_transferred(monkeypatch, training, auxiliary):
    batch = synthetic_batch()
    calls = []
    original = Data.to

    def observed(data, device, *keys, **kwargs):
        calls.append((device, keys, kwargs))
        # The test observes a CUDA request without allocating any CUDA tensors.
        return original(data, "cpu", *keys, **kwargs)

    monkeypatch.setattr(Data, "to", observed)
    result = owner._model_batch_to_device(batch, "cuda", training=training, auxiliary=auxiliary)
    expected = {"x", "edge_index", "edge_attr", "batch", "random_walk_pe"}
    if training: expected.add("y")
    if auxiliary: expected.add("chemical_fingerprint")
    assert result is batch and len(calls) == 1
    assert set(calls[0][1]) == expected
    assert calls[0][2] == {"non_blocking": True}
    assert batch.source_idx.device.type == batch.ptr.device.type == "cpu"


def test_development_preserves_target_and_row_order_without_uploading_them(monkeypatch):
    first = synthetic_batch()
    tail = Batch.from_data_list([Data(x=torch.zeros(2, 9, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, 3, dtype=torch.long), random_walk_pe=torch.zeros(2, 16),
        y=torch.tensor([2.]), source_idx=torch.tensor([2]))])
    calls = []
    original = Data.to

    def observed(data, device, *keys, **kwargs):
        calls.append(keys)
        return original(data, "cpu", *keys, **kwargs)

    monkeypatch.setattr(Data, "to", observed)
    monkeypatch.setattr(owner, "_forward", lambda model, data: data.y.view(-1) + 0.5)
    mae, target, prediction, idx = owner._evaluate(torch.nn.Linear(1, 1), [first, tail], 0., 1.)
    assert mae == 0.5
    assert torch.equal(target, torch.tensor([1., 3., 2.]))
    assert torch.equal(prediction, target + 0.5)
    assert torch.equal(idx, torch.arange(3))
    assert all("y" not in keys and "source_idx" not in keys and "ptr" not in keys for keys in calls)


def test_auxiliary_development_request_fails_before_transfer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No transfer allowed")

    monkeypatch.setattr(Data, "to", forbidden)
    with pytest.raises(ValueError, match="training only"):
        owner._model_batch_to_device(synthetic_batch(), "cuda", training=False, auxiliary=True)


@pytest.mark.parametrize("training", [True, False])
def test_real_pyg_transfer_keeps_metadata_cpu_with_meta_device(training):
    batch = synthetic_batch()
    source, boundaries, target = batch.source_idx, batch.ptr, batch.y
    owner._model_batch_to_device(batch, "meta", training=training)
    assert batch.x.device.type == batch.batch.device.type == "meta"
    assert batch.source_idx is source and batch.ptr is boundaries
    assert source.device.type == boundaries.device.type == "cpu"
    if training:
        assert batch.y.device.type == "meta"
    else:
        assert batch.y is target and target.device.type == "cpu"
