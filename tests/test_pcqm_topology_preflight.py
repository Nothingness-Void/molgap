"""Synthetic loader mechanics; only checked-in hashes identify accepted data."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap import pcqm_topology as topology
from molgap.screen_policy import canonical_fingerprint


@pytest.mark.parametrize("name,expected", [
    ("ogb-train-full", "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"),
    ("ogb-train-100k", "b087fb07b12b7610b0934283384e2f6eb2110422a3fdc2210c93be1deccf6879"),
    ("ogb-train-500k-scnet-v1", "384553c8268447b8f7b83f5104073309e7f12755ab3fc29796dbf78f1914320b"),
    ("ogb-train-1m", "e339e82f903d1381c65fd061c23e7403b72f16a79114a84b9822349555ae4017"),
])
def test_accepted_manifest_fingerprints(name, expected):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "platforms/_records/ims/pcqm_fixed_datasets_v1" /
                           f"{name}.manifest.json").read_text(encoding="utf-8"))
    assert canonical_fingerprint(manifest) == expected


def _synthetic_shard(path, start):
    graphs = [Data(
        x=torch.zeros((2, 9), dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 3), dtype=torch.long),
        random_walk_pe=torch.zeros((2, 16), dtype=torch.float32),
        y=torch.tensor([6.0], dtype=torch.float32),
        row_index=torch.tensor([start + offset], dtype=torch.long),
    ) for offset in range(128)]
    torch.save(InMemoryDataset.collate(graphs), path)
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@pytest.fixture
def synthetic_inputs(tmp_path, monkeypatch):
    paths = []
    records = []
    for role, start in (("train", 0), ("development", 128)):
        name = f"{role}.pt"
        path = tmp_path / name
        receipt = _synthetic_shard(path, start)
        paths.append({"path": name, "role": role, **receipt})
        records.append({"file": name, "role": role, "rows": 128,
                        "source_idx_min": start, "source_idx_max": start + 127, **receipt})
    manifest = {
        "format": "molgap-pcqm4mv2-fixed-subset-v1", "status": "complete",
        "identity": {"name": "ogb-train-100k"},
        "source": {"official_archive_sha256": topology.OFFICIAL_ARCHIVE_SHA256,
                   "official_row_manifest_sha256": topology.OFFICIAL_ROW_MANIFEST_SHA256,
                   "external_data_used": False},
        "assets": {"topology": records},
    }
    (tmp_path / "fixed.json").write_text(json.dumps(manifest), encoding="utf-8")
    accepted = copy.deepcopy(topology.PREFLIGHT_DATASETS)
    accepted[("edge_state_gps", "1")]["ogb-train-100k"] = canonical_fingerprint(manifest)
    monkeypatch.setattr(topology, "PREFLIGHT_DATASETS", accepted)
    entry = {"fixed_manifest": {"path": "fixed.json"}, "files": paths}
    arm = {"family": {"name": "edge_state_gps", "version": "1"}}
    return tmp_path, entry, arm


def test_selected_shards_build_cpu_batches_without_full_role_claim(synthetic_inputs):
    root, entry, arm = synthetic_inputs
    observed = topology.inspect_selected_topology_shards(root, entry, arm)
    assert set(observed) == {"train", "development"}
    assert observed["train"]["graphs"] == 128
    assert observed["development"]["source_idx_min"] == 128
    assert observed["train"]["device"] == "cpu"
    assert observed["train"]["rows"] == 128


def test_k1_train_only_loader(synthetic_inputs, monkeypatch):
    root, entry, arm = synthetic_inputs
    manifest = json.loads((root / "fixed.json").read_text(encoding="utf-8"))
    manifest["identity"]["name"] = "ogb-train-full"
    manifest["assets"]["topology"] = manifest["assets"]["topology"][:1]
    (root / "fixed.json").write_text(json.dumps(manifest), encoding="utf-8")
    accepted = copy.deepcopy(topology.PREFLIGHT_DATASETS)
    accepted[("neural_atom_k1", "1")]["ogb-train-full"] = canonical_fingerprint(manifest)
    monkeypatch.setattr(topology, "PREFLIGHT_DATASETS", accepted)
    entry["files"] = entry["files"][:1]
    arm["family"]["name"] = "neural_atom_k1"
    observed = topology.inspect_selected_topology_shards(root, entry, arm)
    assert set(observed) == {"train"}
    assert observed["train"]["source_idx_max"] == 127


@pytest.mark.parametrize("change,match", [
    ("wrong_manifest", "accepted frozen"),
    ("geometry_asset", "topology inventory"),
    ("wrong_role", "topology inventory"),
    ("changed_bytes", "changed after staging"),
])
def test_selected_shard_rejects_identity_and_file_mismatch(synthetic_inputs, change, match):
    root, entry, arm = synthetic_inputs
    if change == "wrong_manifest":
        value = json.loads((root / "fixed.json").read_text(encoding="utf-8"))
        value["identity"]["name"] = "not-frozen"
        (root / "fixed.json").write_text(json.dumps(value), encoding="utf-8")
    elif change == "geometry_asset":
        entry["files"][0]["path"] = "geometry.pt"
    elif change == "wrong_role":
        entry["files"][0]["role"] = "development"
        entry["files"][1]["role"] = "train"
    else:
        with (root / "train.pt").open("ab") as handle:
            handle.write(b"tampered")
    with pytest.raises(ValueError, match=match):
        topology.inspect_selected_topology_shards(root, entry, arm)
