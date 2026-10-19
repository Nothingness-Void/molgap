"""Namespace compatibility using tiny synthetic objects; no accepted data access."""
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


MODULE_PAIRS = (
    ("molgap.packed_graph_dataset", "molgap.data.packed_graph"),
    ("molgap.pcqm_fixed_records", "molgap.data.fixed_records"),
)


def _isolated(code, directory):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=directory, env=env,
        check=True, capture_output=True, text=True,
    )


def test_package_and_fixed_records_exports_stay_lazy(tmp_path):
    _isolated("""
import sys
import molgap.data as data
assert 'PackedGraphDataset' in dir(data)
assert not any(name.startswith('molgap.data.') for name in sys.modules)
from molgap.data import graph_records
assert graph_records.__module__ == 'molgap.pcqm_fixed_records'
assert 'molgap.data.packed_graph' not in sys.modules
assert 'molgap.data.indexed_inputs' not in sys.modules
assert 'torch' not in sys.modules
assert 'torch_geometric' not in sys.modules
assert 'molgap.experiment_cli' not in sys.modules
assert 'molgap.family_training_runner' not in sys.modules
""", tmp_path)


@pytest.mark.parametrize("new_first", [False, True])
def test_module_aliases_in_both_import_orders(tmp_path, new_first):
    _isolated(f"""
import importlib
pairs = {MODULE_PAIRS!r}
for old_name, new_name in pairs:
    first, second = (new_name, old_name) if {new_first!r} else (old_name, new_name)
    a = importlib.import_module(first)
    b = importlib.import_module(second)
    assert a is b
    assert a.__dict__ is b.__dict__
assert 'molgap.experiment_cli' not in __import__('sys').modules
""", tmp_path)


@pytest.mark.parametrize("old_name,new_name", MODULE_PAIRS)
def test_legacy_private_exports_share_the_implementation_globals(old_name, new_name):
    old = importlib.import_module(old_name)
    new = importlib.import_module(new_name)
    assert old is new
    if new_name.endswith("packed_graph"):
        names = ("Path", "InMemoryDataset", "torch_load_compat", "PackedGraphDataset")
    else:
        names = ("FIXED_KAGGLE_MANIFESTS", "graph_records")
    for name in names:
        assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize("old_name,name", (
    ("molgap.packed_graph_dataset", "PackedGraphDataset"),
    ("molgap.pcqm_fixed_records", "graph_records"),
))
def test_historical_pickle_globals_resolve_to_the_same_objects(old_name, name):
    obj = getattr(importlib.import_module(old_name), name)
    assert obj.__module__ == old_name
    legacy = f"c{old_name}\n{name}\n.".encode("ascii")
    assert pickle.loads(legacy) is obj
    serialized = pickle.dumps(obj, protocol=2)
    assert old_name.encode("ascii") in serialized
    assert pickle.loads(serialized) is obj


@pytest.mark.parametrize("mmap", [False, True])
def test_packed_payload_arguments_cache_and_instance_pickle(monkeypatch, mmap):
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    from molgap import packed_graph_dataset as legacy
    from molgap.data import packed_graph

    graphs = [Data(x=torch.tensor([[i]]), y=torch.tensor([float(i)])) for i in (1, 2)]
    payload = InMemoryDataset.collate(graphs)
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return payload

    monkeypatch.setattr(legacy, "torch_load_compat", load)
    dataset = packed_graph.PackedGraphDataset(Path("synthetic-not-opened.pt"), mmap=mmap)
    assert calls == [(Path("synthetic-not-opened.pt"), {
        "map_location": "cpu", "weights_only": False, **({"mmap": True} if mmap else {}),
    })]
    assert len(dataset) == 2
    assert torch.equal(dataset.get(0).x, graphs[0].x)
    cached = dataset._data_list[0]
    assert dataset.get(0).x.data_ptr() == cached.x.data_ptr()
    assert dataset._data_list[0] is cached
    restored = pickle.loads(pickle.dumps(dataset))
    assert type(restored) is legacy.PackedGraphDataset
    assert torch.equal(restored.get(1).x, graphs[1].x)




def test_packed_topology_projection_preserves_the_desktop_geometry_removal(monkeypatch):
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    from molgap.data import packed_graph

    graph = Data(x=torch.zeros((2, 9), dtype=torch.long), pos=torch.ones((2, 3)),
                 edge_distance=torch.ones(1), geometry_valid=torch.ones(1))
    payload = InMemoryDataset.collate([graph])
    monkeypatch.setattr(packed_graph, "torch_load_compat", lambda *args, **kwargs: payload)
    dataset = packed_graph.PackedGraphDataset(Path("synthetic-not-opened.pt"))
    assert dataset.topology_only() is dataset
    for name in packed_graph.TOPOLOGY_EXCLUDED_FIELDS:
        assert name not in dataset._data
    assert "x" in dataset._data
    assert dataset._data_list is None
