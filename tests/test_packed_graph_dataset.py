"""Tiny synthetic scalar packed shards; no molecular inputs or models."""
import pickle

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap.packed_graph_dataset import PackedGraphDataset
from molgap.v4_runtime import torch_load_compat


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "_lazy_init",
                        lambda *a, **k: pytest.fail("Packed CPU loading must not initialize CUDA"))
    yield
    assert torch.cuda.is_initialized() == initialized


def save_shard(path, rows=4):
    graphs = [Data(x=torch.tensor([[i]], dtype=torch.long),
                   source_idx=torch.tensor(i)) for i in range(rows)]
    torch.save(InMemoryDataset.collate(graphs), path)
    return path


def test_legacy_load_and_pickle_roundtrip(tmp_path):
    path = save_shard(tmp_path / "scalar.pt")

    class Legacy(InMemoryDataset):
        def __init__(self):
            super().__init__(root=None)
            self.data, self.slices = torch_load_compat(path, map_location="cpu", weights_only=False)

    legacy = Legacy()
    shared = PackedGraphDataset(path)
    assert shared._data_list is None
    restored = pickle.loads(pickle.dumps(shared))
    for i in range(len(legacy)):
        assert torch.equal(legacy[i].x, restored[i].x)
        assert torch.equal(legacy[i].source_idx, restored[i].source_idx)
    assert shared._data_list is None


def test_exact_loading_arguments(tmp_path, monkeypatch):
    import molgap.packed_graph_dataset as owner
    path = save_shard(tmp_path / "scalar.pt")
    calls = []
    original = owner.torch_load_compat

    def load(source, **kwargs):
        calls.append((source, kwargs))
        return original(source, **kwargs)

    monkeypatch.setattr(owner, "torch_load_compat", load)
    PackedGraphDataset(path)
    assert calls == [(path, {"map_location": "cpu", "weights_only": False})]


def test_screen_factory_preserves_forbidden_field_strip(tmp_path):
    from molgap.k1_screen_training import _PackedGraphDatasetFactory, FORBIDDEN_MODEL_FIELDS
    graph = Data(x=torch.tensor([[1]]), source_idx=torch.tensor(0))
    for field in FORBIDDEN_MODEL_FIELDS:
        graph[field] = torch.tensor([1])
    path = tmp_path / "strip.pt"
    torch.save(InMemoryDataset.collate([graph, graph.clone()]), path)
    untouched = PackedGraphDataset(path)
    stripped = _PackedGraphDatasetFactory.load(path)
    assert type(stripped) is PackedGraphDataset
    assert all(field in untouched._data for field in FORBIDDEN_MODEL_FIELDS)
    assert all(field not in stripped._data and field not in stripped.slices
               for field in FORBIDDEN_MODEL_FIELDS)


@pytest.mark.parametrize("payload", [(1,), (1, 2, 3)])
def test_not_a_new_payload_format(tmp_path, payload):
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        PackedGraphDataset(path)
