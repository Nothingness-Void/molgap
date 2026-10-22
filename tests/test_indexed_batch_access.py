"""Execution-only indexed reads over tiny generated packed graphs; no models."""
from pathlib import Path
import os
import pickle
import random
import struct

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from molgap.common.wire import canonical_fingerprint
from molgap.cross_scale_checkpoint import state_digest
from molgap.cross_scale_inputs import index_digest, target_digest
from molgap.data import indexed_inputs as owner
from molgap.data.packed_graph import PackedGraphDataset
from molgap.research_memory.trace import file_digest, json_bytes


def build_fixture(root, role="train_100k"):
    prefix = "payloads/common_dev" if role == "common_dev" else "payloads/full_minus_holdout"
    (root / prefix).mkdir(parents=True)
    (root / "roles").mkdir()
    (root / "members").mkdir()
    chunks, members, expected = [], [], {}
    for chunk, size in enumerate((4, 4, 4, 1)):
        graphs = []
        for offset in range(size):
            source = 10 * chunk + offset
            nodes = 2 + offset
            graphs.append(Data(
                x=torch.full((nodes, 9), source, dtype=torch.long),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_attr=torch.full((2, 3), offset, dtype=torch.long),
                random_walk_pe=torch.full((nodes, 16), source / 4),
                y=torch.tensor([source / 4]), source_idx=torch.tensor([source]),
                num_nodes=nodes))
        relative = f"{prefix}/{chunk}.pt"
        torch.save(InMemoryDataset.collate(graphs), root / relative)
        offsets = [0, 2] if size > 1 else [0]
        selected = [graphs[i] for i in offsets]
        ids = [int(g.source_idx.item()) for g in selected]
        data, slices = InMemoryDataset.collate(selected)
        chunks.append({
            "file": relative, "sha256": file_digest(root / relative),
            "offsets": offsets if size > 1 else None, "rows": len(selected),
            "row_sha256": index_digest(ids),
            "target_sha256": target_digest([float(g.y.item()) for g in selected]),
            "topology_sha256": state_digest({
                "data": {k: v for k, v in data.to_dict().items() if k not in {"source_idx", "y"}},
                "slices": {k: v for k, v in (slices or {}).items() if k not in {"source_idx", "y"}}})})
        members.extend(ids)
        expected.update(zip(ids, selected))
    member = root / f"members/{role}.i64le"
    member.write_bytes(struct.pack("<" + "q" * len(members), *members))
    logical = canonical_fingerprint({"members": index_digest(members), "chunks": [
        {k: c[k] for k in ("row_sha256", "target_sha256", "topology_sha256")} for c in chunks]})
    version = "a" * 64
    manifest = {
        "format": owner.FORMAT, "status": "complete", "protected": False,
        "role": role, "version_identity": version, "rows": len(members),
        "feature_schema": "ogb-atom9-bond3-rwse16-v1", "pure_2d": True,
        "ordering": "ascending-source-index", "source_index_file": member.relative_to(root).as_posix(),
        "source_index_file_sha256": file_digest(member), "source_index_sha256": index_digest(members),
        "logical_content_sha256": logical, "graph_chunks": chunks}
    manifest_path = root / f"roles/{role}.json"
    manifest_path.write_bytes(json_bytes(manifest))
    manifest_sha = file_digest(manifest_path)
    receipt = {
        "scope": "CPU_SYNTHETIC_ONLY", "status": "SYNTHETIC_DATA_ROLES_VERIFIED",
        "rule_identity": version, "roles": {role: {
            "path": f"roles/{role}.json", "sha256": manifest_sha, "logical_content_sha256": logical}}}
    (root / "acceptance.json").write_bytes(json_bytes(receipt))
    authority = {"scope": "CPU_SYNTHETIC_ONLY", "role": role, "manifest_sha256": manifest_sha,
                 "operations": ["prediction_input", "labels_read"], "protected": False}
    history = {"role": role, "manifest_sha256": manifest_sha,
               "usage": "selection-used" if role == "common_dev" else "training"}
    permit = owner.IndexedInputPermit(role, file_digest(root / "acceptance.json"), manifest_sha,
        version, canonical_fingerprint(authority), canonical_fingerprint(history))
    return owner.IndexedRoleDataset(root, permit, authority=authority, role_history=history), expected


@pytest.fixture
def dataset(tmp_path):
    return build_fixture(tmp_path)[0]


@pytest.mark.parametrize("role", ["train_100k", "common_dev"])
def test_noncontiguous_batch_matches_pyg_and_clones_every_occurrence(tmp_path, role, monkeypatch):
    data, expected = build_fixture(tmp_path, role)
    requested = [6, 1, 4, 0, 1, 3, 5, 2, 6]
    # A graph memo table must never be allocated, even transiently in get().
    def forbidden_get(*args):
        raise AssertionError("Indexed reads must bypass PyG graph memoization")
    monkeypatch.setattr(PackedGraphDataset, "get", forbidden_get)
    graphs = data.__getitems__(requested)
    for graph, index in zip(graphs, requested):
        reference = expected[data.source_indices[index]]
        assert set(graph.keys()) == set(reference.keys()) | {"role_name"}
        assert graph.role_name == "CPU_SYNTHETIC_ONLY"
        for key, value in reference.to_dict().items():
            if torch.is_tensor(value):
                assert torch.equal(graph[key], value)
            else:
                assert graph[key] == value
    original = graphs[4].x.clone()
    graphs[1].x.zero_()
    graphs[1].y.fill_(-1)
    graphs[0].edge_attr.fill_(-7)
    assert torch.equal(graphs[4].x, original)
    assert torch.equal(data[1].x, original)
    assert torch.equal(data[1].y, expected[data.source_indices[1]].y)
    assert torch.equal(graphs[8].edge_attr, expected[data.source_indices[6]].edge_attr)
    assert all(shard._data_list is None for _, shard in data._open_shards.values())


def test_batch_loads_each_shard_once_without_unbounded_handles_or_rehash(dataset, monkeypatch):
    data, touched = dataset, []
    load = data._load
    def tracked_load(chunk):
        touched.append(chunk)
        shard = load(chunk)
        assert len(data._open_shards) <= 2
        assert len(data._verified_files) <= len(data.manifest["graph_chunks"])
        assert all(value._data_list is None for _, value in data._open_shards.values())
        return shard
    def forbidden_pin(*args):
        raise AssertionError("Unchanged verified shards must not be rehashed after eviction")
    monkeypatch.setattr(data, "_load", tracked_load)
    monkeypatch.setattr(owner, "_pin", forbidden_pin)
    requested = [0, 2, 4, 6, 1, 3, 5, 6, 0]
    for _ in range(2):
        touched.clear()
        assert [int(g.source_idx.item()) for g in data.__getitems__(requested)] == [
            data.source_indices[i] for i in requested]
        assert touched == [0, 1, 2, 3]


def test_warm_batch_checks_stat_once_per_touched_shard(dataset, monkeypatch):
    data, counts = dataset, {}
    paths = {data.root / item["file"] for item in data.manifest["graph_chunks"]}
    stat = Path.stat
    def tracked_stat(path, *args, **kwargs):
        if path in paths:
            counts[path] = counts.get(path, 0) + 1
        return stat(path, *args, **kwargs)
    # Isolate identity stats from the shared path-confinement helper's lstat calls.
    monkeypatch.setattr(owner, "_local", lambda root, relative: root / relative)
    monkeypatch.setattr(Path, "stat", tracked_stat)
    data.__getitems__([4, 6, 5, 6, 4])
    assert counts == {data.root / data.manifest["graph_chunks"][i]["file"]: 1 for i in (2, 3)}


@pytest.mark.parametrize("bad", [-1, 7, True, 1.0, "1", None])
def test_batch_validates_entire_request_before_io(dataset, monkeypatch, bad):
    def forbidden_load(*args):
        raise AssertionError("Invalid batches must fail before opening any shard")
    monkeypatch.setattr(dataset, "_load", forbidden_load)
    with pytest.raises(IndexError, match="row out of range"):
        dataset.__getitems__([0, bad])
    with pytest.raises(IndexError, match="row out of range"):
        dataset[bad]
    assert dataset.__getitems__([]) == []


@pytest.mark.parametrize("index", [0, 6])
@pytest.mark.parametrize("mutation", ["append", "same_size"])
def test_batch_rejects_changed_evicted_and_resident_files(dataset, index, mutation):
    chunk, _ = dataset._row_location(index)
    path = dataset.root / dataset.manifest["graph_chunks"][chunk]["file"]
    # Never truncate a live mmap; both mutations must reject before decoding.
    if mutation == "append":
        with path.open("ab") as stream:
            stream.write(b"changed")
    else:
        info = path.stat()
        with path.open("r+b") as stream:
            stream.write(b"bad!")
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="SHA mismatch"):
        dataset.__getitems__([index, index])


@pytest.mark.parametrize("bad_path", [
    "payloads/independent_audit/0.pt", "payloads/full_minus_holdout/../../outside.pt"])
def test_batch_retains_physical_role_and_path_checks(dataset, bad_path):
    dataset.manifest["graph_chunks"][0]["file"] = bad_path
    with pytest.raises(ValueError):
        dataset.__getitems__([0])


def test_batch_checks_every_decoded_source(dataset, monkeypatch):
    clone = dataset._clone_graph
    calls = []
    def changed_source(shard, offset):
        graph = clone(shard, offset)
        calls.append(offset)
        if len(calls) == 2:
            graph.source_idx.fill_(-1)
        return graph
    monkeypatch.setattr(dataset, "_clone_graph", changed_source)
    with pytest.raises(ValueError, match="row identity changed"):
        dataset.__getitems__([6, 6])


def test_accepted_chunk_budget_still_applies_to_warm_batch(dataset):
    original = dataset.permit
    chunk = dataset.manifest["graph_chunks"][3]
    dataset.permit = owner.AcceptedIndexedInputPermit(
        original.role, original.acceptance_sha256, original.manifest_sha256,
        original.version_identity, original.authority_sha256, original.role_history_sha256,
        max_chunk_bytes=(dataset.root / chunk["file"]).stat().st_size - 1)
    with pytest.raises(ValueError, match="safety cap"):
        dataset.__getitems__([6])


@pytest.mark.parametrize("tamper", ["authority", "history"])
def test_accepted_semantic_permit_checks_precede_io(dataset, monkeypatch, tamper):
    original = dataset.permit
    authority = {"scope": "ACCEPTED_INDEXED_INPUT_V1", "role": original.role,
        "acceptance_sha256": original.acceptance_sha256, "manifest_sha256": original.manifest_sha256,
        "version_identity": original.version_identity, "operations": ["prediction_input", "labels_read"],
        "protected": False, "max_chunk_bytes": 4096}
    history = {"role": original.role, "manifest_sha256": original.manifest_sha256, "usage": "training"}
    if tamper == "authority":
        authority["protected"] = True
    else:
        history["usage"] = "unseen"
    # Even self-consistent hashes cannot authorize changed semantic fields.
    permit = owner.AcceptedIndexedInputPermit(original.role, original.acceptance_sha256,
        original.manifest_sha256, original.version_identity, canonical_fingerprint(authority),
        canonical_fingerprint(history), max_chunk_bytes=4096)
    def forbidden_path(*args):
        raise AssertionError("Permit must be checked before any file access")
    monkeypatch.setattr(owner, "_local", forbidden_path)
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        owner.IndexedRoleDataset(dataset.root, permit, authority=authority, role_history=history)


def test_serialized_worker_reverifies_batch_and_keeps_global_rng(dataset, monkeypatch):
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._open_shards and not restored._verified_files
    pins, pin = [], owner._pin
    def tracked_pin(root, item):
        pins.append(item["file"])
        return pin(root, item)
    monkeypatch.setattr(owner, "_pin", tracked_pin)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    restored.__getitems__([0, 2, 0, 3])
    assert len(pins) == 2
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and np.array_equal(numpy_rng[1], after_numpy[1])
    assert numpy_rng[2:] == after_numpy[2:]
    assert torch.equal(torch_rng, torch.get_rng_state())


@pytest.mark.parametrize("workers", [0, 1])
def test_pyg_loader_keeps_requested_batch_order_with_private_worker_rng(dataset, workers):
    requested = [[6, 0, 4, 0], [3, 1, 5, 2]]
    generator = torch.Generator().manual_seed(42)
    global_rng = torch.get_rng_state()
    options = {"multiprocessing_context": "spawn"} if workers else {}
    loader = DataLoader(dataset, batch_sampler=requested, num_workers=workers,
                        generator=generator, **options)
    assert [batch.source_idx.tolist() for batch in loader] == [
        [dataset.source_indices[i] for i in indices] for indices in requested]
    assert torch.equal(global_rng, torch.get_rng_state())
