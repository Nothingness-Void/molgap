"""Bounded file-lifetime tests for the indexed loader, without graph payloads."""
from collections import OrderedDict
import pickle
from types import SimpleNamespace

import pytest

from molgap.data import indexed_inputs as owner
from molgap.data import packed_graph
from molgap.research_memory.trace import file_digest


@pytest.fixture
def cache(tmp_path, monkeypatch):
    directory = tmp_path / "payloads" / "full_minus_holdout"
    directory.mkdir(parents=True)
    chunks = []
    for index in range(3):
        path = directory / (str(index) + ".pt")
        path.write_bytes(("synthetic-" + str(index)).encode())
        chunks.append({"file": path.relative_to(tmp_path).as_posix(), "sha256": file_digest(path)})
    data = object.__new__(owner.IndexedRoleDataset)
    data.root, data.permit = tmp_path, SimpleNamespace()
    data.manifest = {"role": "train_100k", "graph_chunks": chunks}
    data._loaded = data._loaded_index = data._loaded_file_identity = None
    data._verified_files, data._open_shards = {}, OrderedDict()
    hashes, opens = [], []
    pin = owner._pin
    def tracked_pin(root, item):
        hashes.append(item["file"])
        return pin(root, item)
    def opened(path, *, mmap):
        assert mmap
        opens.append(str(path))
        return SimpleNamespace(path=str(path))
    monkeypatch.setattr(owner, "_pin", tracked_pin)
    monkeypatch.setattr(packed_graph, "PackedGraphDataset", opened)
    return data, hashes, opens


def test_eviction_does_not_repeat_whole_shard_hash(cache):
    data, hashes, opens = cache
    for index in [0, 1, 2, 0, 1, 2, 0]:
        data._load(index)
        assert len(data._open_shards) <= 2
    assert len(hashes) == 3
    assert len(opens) == 7
    data._load(0)
    assert len(opens) == 7


@pytest.mark.parametrize("resident", [True, False])
def test_changed_shard_is_rehashed_and_rejected(cache, resident):
    data, hashes, _ = cache
    data._load(0)
    if not resident:
        data._load(1)
        data._load(2)
    (data.root / data.manifest["graph_chunks"][0]["file"]).write_bytes(b"changed synthetic payload")
    with pytest.raises(ValueError):
        data._load(0)
    assert hashes.count(data.manifest["graph_chunks"][0]["file"]) == 2


def test_spawn_serialization_clears_verified_and_open_handles(cache):
    data, hashes, _ = cache
    data._load(0)
    restored = pickle.loads(pickle.dumps(data))
    assert not restored._verified_files and not restored._open_shards
    assert restored._loaded is None
    restored._load(0)
    assert len(hashes) == 2
    assert len(data._verified_files) == 1


def test_change_while_opening_is_not_cached(cache, monkeypatch):
    data, _, _ = cache
    def racing_open(path, **kwargs):
        path.write_bytes(b"changed while opening")
        return SimpleNamespace()
    monkeypatch.setattr(packed_graph, "PackedGraphDataset", racing_open)
    with pytest.raises(ValueError, match="changed while opening"):
        data._load(0)
    assert not data._verified_files and not data._open_shards
