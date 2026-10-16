"""Namespace compatibility using tiny synthetic objects; no accepted data access."""
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest


MODULE_PAIRS = (
    ("molgap.packed_graph_dataset", "molgap.data.packed_graph"),
    ("molgap.pcqm_fixed_records", "molgap.data.fixed_records"),
    ("molgap.pcqm_indexed_inputs", "molgap.data.indexed_inputs"),
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
    elif new_name.endswith("fixed_records"):
        names = ("FIXED_KAGGLE_MANIFESTS", "graph_records")
    else:
        names = ("_pin", "_local", "_safe_local", "authorize_role", "GRAPH_FIELDS",
                 "IndexedInputPermit", "IndexedRoleDataset", "mirror_prerequisites")
    for name in names:
        assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize("old_name,name", (
    ("molgap.packed_graph_dataset", "PackedGraphDataset"),
    ("molgap.pcqm_fixed_records", "graph_records"),
    ("molgap.pcqm_indexed_inputs", "IndexedInputPermit"),
    ("molgap.pcqm_indexed_inputs", "IndexedRoleDataset"),
    ("molgap.pcqm_indexed_inputs", "mirror_prerequisites"),
    ("molgap.pcqm_indexed_inputs", "_local"),
))
def test_historical_pickle_globals_resolve_to_the_same_objects(old_name, name):
    obj = getattr(importlib.import_module(old_name), name)
    assert obj.__module__ == old_name
    legacy = f"c{old_name}\n{name}\n.".encode("ascii")
    assert pickle.loads(legacy) is obj
    serialized = pickle.dumps(obj, protocol=2)
    assert old_name.encode("ascii") in serialized
    assert pickle.loads(serialized) is obj


def test_permit_instance_round_trip_and_cpu_only_guard():
    from molgap.data import IndexedInputPermit
    from molgap.pcqm_indexed_inputs import IndexedInputPermit as LegacyPermit

    permit = IndexedInputPermit("common_dev", *(["a" * 64] * 5))
    restored = pickle.loads(pickle.dumps(permit))
    assert type(restored) is LegacyPermit
    assert restored == permit
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        IndexedInputPermit("common_dev", *(["a" * 64] * 5), scope="REAL_DATA")
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        IndexedInputPermit("independent_audit", *(["a" * 64] * 5))


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


def test_indexed_legacy_pin_patch_cache_invalidation_and_clone_parity(tmp_path, monkeypatch):
    import torch
    from torch_geometric.data import Data
    from molgap import packed_graph_dataset as packed_legacy
    from molgap import pcqm_indexed_inputs as legacy
    from molgap.data import indexed_inputs

    path = tmp_path / "payloads" / "common_dev" / "synthetic.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic fixture")
    item = {"file": "payloads/common_dev/synthetic.pt", "sha256": "a" * 64}
    graphs = [Data(x=torch.tensor([[i]]), source_idx=torch.tensor([i])) for i in (11, 7)]
    pins, opens = [], []

    def pin(root, record):
        pins.append((root, dict(record)))
        return path

    def open_packed(opened_path, *, mmap):
        opens.append((opened_path, mmap))
        return SimpleNamespace(get=lambda index: graphs[index])

    monkeypatch.setattr(legacy, "_pin", pin)
    monkeypatch.setattr(packed_legacy, "PackedGraphDataset", open_packed)
    assert indexed_inputs.IndexedRoleDataset._load.__globals__["_pin"] is pin
    dataset = object.__new__(indexed_inputs.IndexedRoleDataset)
    dataset.root = tmp_path
    dataset.manifest = {"role": "common_dev", "graph_chunks": [item]}
    dataset.source_indices = (7, 11)
    dataset._maps, dataset._ends = [(1, 0)], [2]
    dataset._loaded_index = dataset._loaded = dataset._loaded_file_identity = None
    assert len(dataset) == 2
    assert [int(dataset[i].source_idx.item()) for i in range(2)] == [7, 11]
    loaded = dataset._loaded
    dataset[0].x.zero_()
    assert int(dataset[0].x.item()) == 7
    assert dataset._loaded is loaded
    assert len(pins) == len(opens) == 1
    path.write_bytes(b"changed synthetic fixture length")
    assert int(dataset[0].source_idx.item()) == 7
    assert dataset._loaded is not loaded
    assert len(pins) == len(opens) == 2
    item["sha256"] = "b" * 64
    dataset[1]
    assert len(pins) == len(opens) == 3
