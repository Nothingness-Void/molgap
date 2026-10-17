"""Generated accepted-ABI leaf checks; no actual PCQM or role access is used."""

from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import pickle

import pytest
import torch
from torch_geometric.data import Data

from molgap.common.wire import canonical_fingerprint
from molgap.cross_scale_checkpoint import state_digest
from molgap.cross_scale_inputs import index_digest, target_digest
from molgap.data.indexed_inputs import AcceptedIndexedInputPermit, IndexedRoleDataset
from molgap.family_training_data import (
    DatasetSelection,
    TrainingData,
    load_indexed_training_data,
)
from molgap.pcqm_unified_roles import FORMAT
from molgap.research_memory.trace import file_digest, json_bytes


MAX_CHUNK_BYTES = 32 * 1024**2
TRAIN_ROWS = 100_000
DEV_ROWS = 50_000
AUDIT_ONLY_NAMES = {
    "private_identity.sqlite",
    "preparation_role_events.json",
    "rule.json",
    "progress.json",
}


def _fixture_version_identity():
    return canonical_fingerprint(
        {
            "format": FORMAT,
            "fixture_schema": "generated-compact-pygdata-slices-v1",
            "source_ids": "train=2*i; common_dev=200001+2*i",
            "target_dtype": "torch.float32",
        }
    )


def _write_member_file(path, source_indices):
    # The fixture uses the native little-endian int64 member representation.
    path.write_bytes(
        source_indices.detach().cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()
    )


def _write_compact_pyg_chunk(path, source_indices, targets):
    """Serialize one compact PyG payload without constructing per-row Data objects."""
    rows = int(source_indices.numel())
    data = Data(
        x=torch.zeros((rows, 9), dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 3), dtype=torch.long),
        random_walk_pe=torch.zeros((rows, 16), dtype=torch.float32),
        source_idx=source_indices.detach().cpu().contiguous(),
        y=targets.detach().cpu().contiguous(),
    )
    one_value_slices = torch.arange(rows + 1, dtype=torch.long)
    empty_edge_slices = torch.zeros(rows + 1, dtype=torch.long)
    slices = {
        "x": one_value_slices,
        "edge_index": empty_edge_slices,
        "edge_attr": empty_edge_slices,
        "random_walk_pe": one_value_slices,
        "source_idx": one_value_slices,
        "y": one_value_slices,
    }
    # This is the native compact graph ABI: (Data, slices), not a list of
    # Python graph objects and not an embedding/feature cache.
    torch.save((data, slices), path)
    topology = {key: value for key, value in data.to_dict().items() if key not in {"source_idx", "y"}}
    topology_sha256 = state_digest(
        {
            "data": topology,
            "slices": {key: value for key, value in slices.items() if key not in {"source_idx", "y"}},
        }
    )
    return {
        "rows": rows,
        "row_sha256": index_digest(source_indices.tolist()),
        "target_sha256": target_digest(targets.tolist()),
        "topology_sha256": topology_sha256,
        "source_idx_min": int(source_indices[0]),
        "source_idx_max": int(source_indices[-1]),
        "offsets": None,
    }


def _role_arrays(role):
    if role == "train_100k":
        rows = TRAIN_ROWS
        source_indices = 2 * torch.arange(rows, dtype=torch.int64)
        targets = 0.25 + (torch.arange(rows, dtype=torch.float32) % 97) / 10.0
        chunk_sizes = (40_000, 40_000, 20_000)
    elif role == "common_dev":
        rows = DEV_ROWS
        source_indices = 200_001 + 2 * torch.arange(rows, dtype=torch.int64)
        targets = 20.0 + (torch.arange(rows, dtype=torch.float32) % 89) / 10.0
        chunk_sizes = (25_000, 25_000)
    else:
        raise AssertionError(f"Unexpected generated fixture role: {role}")
    return source_indices, targets, chunk_sizes


def _build_accepted_abi_fixture(base):
    """Create only generated compact payloads with native accepted-role metadata."""
    root = Path(base) / "accepted-indexed-fixture"
    (root / "members").mkdir(parents=True)
    (root / "roles").mkdir()
    (root / "payloads" / "full_minus_holdout").mkdir(parents=True)
    (root / "payloads" / "common_dev").mkdir(parents=True)

    version_identity = _fixture_version_identity()
    generator_sha256 = hashlib.sha256(b"test_pcqm_accepted_indexed_inputs.py:compact-v1").hexdigest()
    source_commit = hashlib.sha1(b"generated-accepted-indexed-fixture").hexdigest()
    source_manifest = {
        "format": "generated-pcqm-fixture-source-v1",
        "fixture_only": True,
        "roles": ["train_100k", "common_dev"],
        "source_ids": "noncontiguous-and-disjoint",
    }
    source_acceptance = {
        "status": "GENERATED_FIXTURE_ONLY",
        "official_validation_read": False,
        "test_roles_read": False,
        "audit_payload_created": False,
    }
    (root / "fixture_source_manifest.json").write_bytes(json_bytes(source_manifest))
    (root / "fixture_source_acceptance.json").write_bytes(json_bytes(source_acceptance))

    role_specs = {}
    for role in ("train_100k", "common_dev"):
        source_indices, targets, chunk_sizes = _role_arrays(role)
        member_path = root / "members" / f"{role}.i64le"
        _write_member_file(member_path, source_indices)
        prefix = "payloads/full_minus_holdout" if role.startswith("train_") else "payloads/common_dev"
        chunks = []
        cursor = 0
        for chunk_number, chunk_rows in enumerate(chunk_sizes):
            stop = cursor + chunk_rows
            relative = f"{prefix}/{role}-{chunk_number:02d}.pt"
            chunk_path = root / relative
            item = _write_compact_pyg_chunk(
                chunk_path,
                source_indices[cursor:stop],
                targets[cursor:stop],
            )
            assert chunk_path.stat().st_size <= MAX_CHUNK_BYTES
            item.update(
                {
                    "file": relative,
                    "sha256": file_digest(chunk_path),
                    "parent_sha256": source_commit,
                }
            )
            chunks.append(item)
            cursor = stop
        assert cursor == int(source_indices.numel())
        logical_content_sha256 = canonical_fingerprint(
            {
                "members": index_digest(source_indices.tolist()),
                "chunks": [
                    {key: item[key] for key in ("row_sha256", "target_sha256", "topology_sha256")}
                    for item in chunks
                ],
            }
        )
        manifest = {
            "format": FORMAT,
            "status": "complete",
            "fixture_only": True,
            "version_identity": version_identity,
            "role": role,
            "rows": int(source_indices.numel()),
            "source_index_sha256": index_digest(source_indices.tolist()),
            "ordering": "ascending-source-index",
            "source_index_file": f"members/{role}.i64le",
            "source_index_file_sha256": file_digest(member_path),
            "source_manifest_sha256": file_digest(root / "fixture_source_manifest.json"),
            "source_acceptance_sha256": file_digest(root / "fixture_source_acceptance.json"),
            "role_inventory_sha256": canonical_fingerprint({"role": role, "rows": int(source_indices.numel())}),
            "molecule_rule": "generated-one-atom-no-molecule-v1",
            "generator_sha256": generator_sha256,
            "source_commit": source_commit,
            "feature_schema": "ogb-atom9-bond3-rwse16-v1",
            "target": "pcqm4mv2-gap-eV-direct",
            "target_dtype": "torch.float32",
            "pure_2d": True,
            "graph_chunks": chunks,
            "protected": False,
            "access": "TRAIN_OR_COMMON_DEVELOPMENT",
            "label_statistics_published": False,
            "checks": {
                "generated_fixture_only": True,
                "all_rows_finite_fp32": True,
                "all_topology_compact_native_format": True,
            },
            "all_row_sha256": index_digest(source_indices.tolist()),
            "all_target_sha256": target_digest(targets.tolist()),
            "all_topology_sha256": state_digest({"chunks": [item["topology_sha256"] for item in chunks]}),
            "logical_content_sha256": logical_content_sha256,
        }
        manifest_path = root / "roles" / f"{role}.json"
        manifest_path.write_bytes(json_bytes(manifest))
        role_specs[role] = {
            "rows": int(source_indices.numel()),
            "source_indices": source_indices,
            "targets": targets,
            "manifest_path": manifest_path,
            "member_path": member_path,
            "manifest": manifest,
        }

    receipt = {
        "format": FORMAT + "-acceptance",
        "status": "DATA_ROLE_ACCEPTED",
        "scope": "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY",
        "fixture_only": True,
        "acceptance_claim": "GENERATED_FIXTURE_ONLY",
        "rule_identity": version_identity,
        "official_validation_read": False,
        "test_roles_read": False,
        "gpu_used": False,
        "graphs_rebuilt": False,
        "audit_label_statistics_exposed": False,
        "training_authority": "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING",
        "roles": {},
    }
    for role, spec in role_specs.items():
        manifest = spec["manifest"]
        receipt["roles"][role] = {
            "path": f"roles/{role}.json",
            "sha256": file_digest(spec["manifest_path"]),
            "rows": spec["rows"],
            "source_index_sha256": manifest["source_index_sha256"],
            "logical_content_sha256": manifest["logical_content_sha256"],
            "member_file": manifest["source_index_file"],
            "member_file_sha256": manifest["source_index_file_sha256"],
            "fixture_only": True,
        }
    receipt_path = root / "acceptance.json"
    receipt_path.write_bytes(json_bytes(receipt))
    return {
        "root": root,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "role_specs": role_specs,
    }


def _role_record(fixture, role):
    root = fixture["root"]
    receipt = json.loads(fixture["receipt_path"].read_bytes())
    manifest_path = root / receipt["roles"][role]["path"]
    manifest_sha256 = file_digest(manifest_path)
    authority = {
        "scope": "ACCEPTED_INDEXED_INPUT_V1",
        "role": role,
        "acceptance_sha256": file_digest(fixture["receipt_path"]),
        "manifest_sha256": manifest_sha256,
        "version_identity": receipt["rule_identity"],
        "operations": ["prediction_input", "labels_read"],
        "protected": False,
        "max_chunk_bytes": MAX_CHUNK_BYTES,
    }
    history = {
        "role": role,
        "manifest_sha256": manifest_sha256,
        "usage": "training" if role.startswith("train_") else "selection-used",
    }
    permit = AcceptedIndexedInputPermit(
        role,
        authority["acceptance_sha256"],
        manifest_sha256,
        receipt["rule_identity"],
        canonical_fingerprint(authority),
        canonical_fingerprint(history),
        MAX_CHUNK_BYTES,
    )
    return permit, authority, history


def _write_binding(fixture, parent):
    train = _role_record(fixture, "train_100k")
    development = _role_record(fixture, "common_dev")
    path = Path(parent) / "indexed-training-input.json"
    payload = {
        "format": "molgap-indexed-training-input-v1",
        "data_root": fixture["root"].relative_to(path.parent).as_posix(),
        "scope": "ACCEPTED_INDEXED_INPUT_V1",
        "train": {
            "permit": asdict(train[0]),
            "authority": train[1],
            "role_history": train[2],
        },
        "development": {
            "permit": asdict(development[0]),
            "authority": development[1],
            "role_history": development[2],
        },
    }
    path.write_bytes(json_bytes(payload))
    return path, payload, DatasetSelection(path.relative_to(parent).as_posix(), file_digest(path), True)


def _refresh_role_binding(fixture, role):
    """Re-pin a deliberately edited manifest for a negative test."""
    receipt_path = fixture["receipt_path"]
    receipt = json.loads(receipt_path.read_bytes())
    manifest_path = fixture["root"] / receipt["roles"][role]["path"]
    manifest_sha256 = file_digest(manifest_path)
    receipt["roles"][role]["sha256"] = manifest_sha256
    receipt["roles"][role]["rows"] = json.loads(manifest_path.read_bytes())["rows"]
    receipt_path.write_bytes(json_bytes(receipt))
    return _role_record(fixture, role)


def _assert_one_atom_graph(graph, source_index, role):
    assert graph.role_name == role
    assert graph.num_nodes == 1
    assert graph.x.shape == (1, 9)
    assert graph.x.dtype == torch.int64
    assert torch.equal(graph.x, torch.zeros((1, 9), dtype=torch.long))
    assert graph.random_walk_pe.shape == (1, 16)
    assert graph.random_walk_pe.dtype == torch.float32
    assert torch.equal(graph.random_walk_pe, torch.zeros((1, 16), dtype=torch.float32))
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 3)
    assert graph.edge_attr.dtype == torch.int64
    assert graph.source_idx.dtype == torch.int64
    assert int(graph.source_idx.item()) == source_index
    assert graph.y.dtype == torch.float32
    assert graph.y.numel() == 1 and bool(torch.isfinite(graph.y).all())


@pytest.mark.parametrize(
    "role,expected_rows,expected_first,expected_last",
    [
        ("train_100k", TRAIN_ROWS, 0, 199_998),
        ("common_dev", DEV_ROWS, 200_001, 299_999),
    ],
)
def test_generated_accepted_indexed_role_dataset_positive_real_abi(
    tmp_path, role, expected_rows, expected_first, expected_last
):
    fixture = _build_accepted_abi_fixture(tmp_path)
    permit, authority, history = _role_record(fixture, role)
    dataset = IndexedRoleDataset(fixture["root"], permit, authority=authority, role_history=history)

    assert len(dataset) == expected_rows
    expected_indices = tuple((expected_first + 2 * i) for i in range(expected_rows))
    assert dataset.source_indices == expected_indices
    assert dataset.source_indices[0] == expected_first
    assert dataset.source_indices[-1] == expected_last

    first = dataset[0]
    last = dataset[len(dataset) - 1]
    _assert_one_atom_graph(first, expected_first, role)
    _assert_one_atom_graph(last, expected_last, role)
    assert first.x.data_ptr() != dataset._loaded._data.x.data_ptr()
    first.x.fill_(7)
    _assert_one_atom_graph(dataset[0], expected_first, role)


def test_generated_accepted_fixture_pins_all_content_and_keeps_chunks_below_32_mib(tmp_path):
    fixture = _build_accepted_abi_fixture(tmp_path)
    receipt = json.loads(fixture["receipt_path"].read_bytes())
    assert receipt["status"] == "DATA_ROLE_ACCEPTED"
    assert receipt["scope"] == "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY"
    assert receipt["fixture_only"] is True
    assert receipt["acceptance_claim"] == "GENERATED_FIXTURE_ONLY"
    assert receipt["training_authority"] == "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING"

    for role, spec in fixture["role_specs"].items():
        manifest = json.loads(spec["manifest_path"].read_bytes())
        source_indices = spec["source_indices"]
        targets = spec["targets"]
        assert manifest["rows"] == int(source_indices.numel())
        assert manifest["source_index_sha256"] == index_digest(source_indices.tolist())
        assert manifest["all_row_sha256"] == index_digest(source_indices.tolist())
        assert manifest["all_target_sha256"] == target_digest(targets.tolist())
        assert manifest["all_topology_sha256"] == state_digest(
            {"chunks": [item["topology_sha256"] for item in manifest["graph_chunks"]]}
        )
        assert manifest["source_index_file_sha256"] == file_digest(spec["member_path"])
        binding = receipt["roles"][role]
        assert binding["sha256"] == file_digest(spec["manifest_path"])
        assert binding["member_file"] == manifest["source_index_file"]
        assert binding["member_file_sha256"] == manifest["source_index_file_sha256"]
        assert all(
            (fixture["root"] / item["file"]).stat().st_size <= MAX_CHUNK_BYTES
            for item in manifest["graph_chunks"]
        )


def test_accepted_dataset_memoizes_mmap_chunks_and_serializes_without_payload(tmp_path, monkeypatch):
    fixture = _build_accepted_abi_fixture(tmp_path)
    permit, authority, history = _role_record(fixture, "train_100k")

    import molgap.data.packed_graph as packed_graph

    original = packed_graph.PackedGraphDataset
    opened = []

    class CountingPackedGraphDataset(original):
        def __init__(self, path, *, mmap=False):
            opened.append((Path(path).name, mmap))
            super().__init__(path, mmap=mmap)

    monkeypatch.setattr(packed_graph, "PackedGraphDataset", CountingPackedGraphDataset)
    dataset = IndexedRoleDataset(fixture["root"], permit, authority=authority, role_history=history)
    assert len(opened) == 3
    assert all(mmap is True for _, mmap in opened)

    serialized = pickle.dumps(dataset)
    state = dataset.__getstate__()
    assert state["_loaded"] is None
    assert state["_loaded_index"] is None
    assert state["_loaded_file_identity"] is None
    payload_bytes = sum(
        (fixture["root"] / item["file"]).stat().st_size
        for item in fixture["role_specs"]["train_100k"]["manifest"]["graph_chunks"]
    )
    assert len(serialized) < payload_bytes

    before = len(opened)
    dataset[0]
    assert len(opened) == before + 1
    dataset[1]
    dataset[39_999]
    assert len(opened) == before + 1
    dataset[40_000]
    assert len(opened) == before + 2
    dataset[40_001]
    dataset[99_999]
    assert len(opened) == before + 3

    restored = pickle.loads(serialized)
    assert restored._loaded is None
    assert restored._loaded_index is None
    assert restored._loaded_file_identity is None
    _assert_one_atom_graph(restored[0], 0, "train_100k")


def test_accepted_loader_returns_exact_typed_training_data_train_only_stats_and_no_audit_io(
    tmp_path, monkeypatch
):
    fixture = _build_accepted_abi_fixture(tmp_path)
    binding_path, _, selection = _write_binding(fixture, tmp_path)
    for name in AUDIT_ONLY_NAMES:
        (fixture["root"] / name).write_bytes(b"must not be read by indexed training loader")

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name in AUDIT_ONLY_NAMES:
            raise AssertionError(f"audit/provenance IO reached: {path.name}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    data = load_indexed_training_data(tmp_path, selection)

    assert type(data) is TrainingData
    assert tuple(field.name for field in fields(TrainingData)) == (
        "train",
        "development",
        "manifest_sha256",
        "train_rows",
        "development_rows",
        "train_source_start",
        "development_source_start",
        "target_mean_eV",
        "target_std_eV",
        "synthetic",
        "train_source_indices",
        "development_source_indices",
        "indexed",
    )
    assert data.indexed is True
    assert data.synthetic is False
    assert data.train_rows == TRAIN_ROWS
    assert data.development_rows == DEV_ROWS
    assert data.train_source_start is None
    assert data.development_source_start is None
    assert data.train_source_indices == tuple(2 * i for i in range(TRAIN_ROWS))
    assert data.development_source_indices == tuple(200_001 + 2 * i for i in range(DEV_ROWS))
    assert set(data.train_source_indices).isdisjoint(data.development_source_indices)

    train_targets = torch.cat(list(data.train.selected_targets()))
    assert data.target_mean_eV == pytest.approx(float(train_targets.mean()))
    assert data.target_std_eV == pytest.approx(float(train_targets.std().clamp_min(1e-6)))
    development_targets = torch.cat(list(data.development.selected_targets()))
    combined_targets = torch.cat((train_targets, development_targets))
    assert data.target_mean_eV != pytest.approx(float(combined_targets.mean()))

    _assert_one_atom_graph(data.train[TRAIN_ROWS - 1], 199_998, "train_100k")
    _assert_one_atom_graph(data.development[DEV_ROWS - 1], 299_999, "common_dev")
    assert binding_path.exists()


def test_tampered_chunk_is_rejected_after_byte_pin_check(tmp_path):
    fixture = _build_accepted_abi_fixture(tmp_path)
    manifest = fixture["role_specs"]["train_100k"]["manifest"]
    payload = fixture["root"] / manifest["graph_chunks"][0]["file"]
    payload.write_bytes(payload.read_bytes() + b"tampered")
    permit, authority, history = _role_record(fixture, "train_100k")
    with pytest.raises(ValueError, match="SHA"):
        IndexedRoleDataset(fixture["root"], permit, authority=authority, role_history=history)


def test_tampered_offsets_are_rejected_from_repinned_manifest(tmp_path):
    fixture = _build_accepted_abi_fixture(tmp_path)
    manifest_path = fixture["role_specs"]["train_100k"]["manifest_path"]
    manifest = json.loads(manifest_path.read_bytes())
    manifest["graph_chunks"][0]["offsets"] = [0] * manifest["graph_chunks"][0]["rows"]
    manifest_path.write_bytes(json_bytes(manifest))
    permit, authority, history = _refresh_role_binding(fixture, "train_100k")
    with pytest.raises(ValueError, match="offsets"):
        IndexedRoleDataset(fixture["root"], permit, authority=authority, role_history=history)


def test_repinned_wrong_count_fails_before_payload_open(tmp_path, monkeypatch):
    fixture = _build_accepted_abi_fixture(tmp_path)
    manifest_path = fixture["role_specs"]["train_100k"]["manifest_path"]
    manifest = json.loads(manifest_path.read_bytes())
    manifest["rows"] = TRAIN_ROWS - 1
    manifest_path.write_bytes(json_bytes(manifest))
    permit, authority, history = _refresh_role_binding(fixture, "train_100k")

    import molgap.data.packed_graph as packed_graph

    opened = []

    def forbidden(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("wrong-count fixture opened a payload")

    monkeypatch.setattr(packed_graph, "PackedGraphDataset", forbidden)
    with pytest.raises(ValueError, match="frozen scale"):
        IndexedRoleDataset(fixture["root"], permit, authority=authority, role_history=history)
    assert opened == []


def test_protected_role_and_authority_fail_before_payload_open(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="accepted indexed train/Common-Dev"):
        AcceptedIndexedInputPermit(
            "independent_audit",
            *(["a" * 64] * 5),
            MAX_CHUNK_BYTES,
        )

    fixture = _build_accepted_abi_fixture(tmp_path)
    permit, authority, history = _role_record(fixture, "common_dev")
    protected_authority = {**authority, "protected": True}

    import molgap.data.packed_graph as packed_graph

    opened = []

    def forbidden(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("protected role reached payload")

    monkeypatch.setattr(packed_graph, "PackedGraphDataset", forbidden)
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        IndexedRoleDataset(
            fixture["root"], permit, authority=protected_authority, role_history=history
        )
    assert opened == []


@pytest.mark.parametrize("tamper", ["authority", "history", "version"])
def test_accepted_binding_authority_history_and_version_fail_before_payload_open(
    tmp_path, monkeypatch, tamper
):
    fixture = _build_accepted_abi_fixture(tmp_path)
    binding_path, payload, selection = _write_binding(fixture, tmp_path)
    changed = deepcopy(payload)
    if tamper == "authority":
        changed["train"]["authority"]["protected"] = True
    elif tamper == "history":
        changed["train"]["role_history"]["usage"] = "unbound"
    else:
        changed["train"]["permit"]["version_identity"] = "f" * 64
    binding_path.write_bytes(json_bytes(changed))
    selection = DatasetSelection(
        binding_path.relative_to(tmp_path).as_posix(), file_digest(binding_path), True
    )

    import molgap.data.packed_graph as packed_graph

    opened = []

    def forbidden(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("tampered binding reached payload")

    monkeypatch.setattr(packed_graph, "PackedGraphDataset", forbidden)
    with pytest.raises(ValueError):
        load_indexed_training_data(tmp_path, selection)
    assert opened == []
