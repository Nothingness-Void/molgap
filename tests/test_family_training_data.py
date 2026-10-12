"""Synthetic pinned shards only; no repository molecular data is read."""
import json
import shutil
from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap.family_training_data import DatasetSelection, PassBatchSampler, load_training_data
from molgap.training_reproducibility import sha256_file


@pytest.fixture
def pinned_data(tmp_path):
    root = tmp_path / "platform-one"
    root.mkdir()
    records = []
    for role, start, rows in (("train", 0, 4), ("development", 4, 2)):
        graphs = [Data(x=torch.zeros(1, 9, dtype=torch.long),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 3, dtype=torch.long),
            random_walk_pe=torch.zeros(1, 16),
            y=torch.tensor([float(i)]), source_idx=torch.tensor([i]))
            for i in range(start, start + rows)]
        path = root / (role + ".pt")
        torch.save(InMemoryDataset.collate(graphs), path)
        records.append(dict(role=role, file=path.name, rows=rows, source_idx_min=start,
            source_idx_max=start + rows - 1, sha256=sha256_file(path)))
    manifest = {"format": "molgap-pcqm4mv2-fixed-subset-v1", "status": "complete",
        "source": {"external_data_used": False, "official_train_rows": 6},
        "graph_contract": {"feature_schema": "ogb", "node_feature_dim": 9,
                           "edge_feature_dim": 3, "rwse_dim": 16},
        "roles": {"train": {"source_idx_start": 0, "source_idx_stop": 4, "rows": 4},
                  "development": {"source_idx_start": 4, "source_idx_stop": 6, "rows": 2}},
        "assets": {"topology": records}}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, DatasetSelection(path.name, sha256_file(path), True), manifest


def test_mount_path_is_not_dataset_identity(pinned_data, tmp_path):
    root, selection, _ = pinned_data
    second = tmp_path / "platform-two"
    shutil.copytree(root, second)
    first, moved = load_training_data(root, selection), load_training_data(second, selection)
    assert first.manifest_sha256 == moved.manifest_sha256
    assert first.train_rows == moved.train_rows == 4
    assert first.development_rows == 2
    assert first.target_mean_eV == moved.target_mean_eV == 1.5
    assert first.target_std_eV == pytest.approx(torch.arange(4.).std().item())
    assert [int(row.source_idx) for row in moved.train] == [0, 1, 2, 3]


def test_full_train_selection_never_reads_unselected_development(pinned_data):
    root, selection, _ = pinned_data
    (root / "development.pt").unlink()
    data = load_training_data(root, replace(selection, development=False))
    assert data.development is None and data.development_rows == 0


@pytest.mark.parametrize("development", [False, True])
def test_existing_kaggle_mirror_uses_same_topology_loader(pinned_data, development):
    root, selection, manifest = pinned_data
    manifest["format"] = "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
    manifest["geometry_shards"] = manifest.pop("assets")["topology"]
    for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        manifest[key] = False
    for record in manifest["geometry_shards"]:
        path = root / record["file"]
        payload, slices = torch.load(path, weights_only=False)
        payload.pos = torch.ones(payload.x.shape[0], 3)
        slices["pos"] = slices["x"].clone()
        torch.save((payload, slices), path)
        record["sha256"] = sha256_file(path)
    path = root / selection.manifest
    path.write_text(json.dumps(manifest), encoding="utf-8")
    if not development:
        (root / "development.pt").unlink()
    data = load_training_data(root, replace(selection, manifest_sha256=sha256_file(path),
                                          development=development))
    assert data.train_rows == 4
    assert data.development_rows == (2 if development else 0)
    assert data.target_mean_eV == 1.5
    assert all("pos" not in graph for graph in data.train)
    if development:
        assert all("pos" not in graph for graph in data.development)


@pytest.mark.parametrize("flag", ["official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"])
@pytest.mark.parametrize("value", [True, None])
def test_kaggle_mirror_missing_or_protected_flags_fail_before_pickle(pinned_data, monkeypatch, flag, value):
    root, selection, manifest = pinned_data
    manifest["format"] = "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
    manifest["geometry_shards"] = manifest.pop("assets")["topology"]
    for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        manifest[key] = False
    manifest[flag] = value
    path = root / selection.manifest
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("molgap.packed_graph_dataset.PackedGraphDataset",
                        lambda *a, **k: pytest.fail("no protected/ambiguous pickle access"))
    with pytest.raises(ValueError, match="protected-role"):
        load_training_data(root, replace(selection, manifest_sha256=sha256_file(path)))


@pytest.mark.parametrize("change", ["overlap", "protected", "missing", "reorder", "bad_features"])
def test_bad_view_is_rejected_before_deserialization(pinned_data, monkeypatch, change):
    root, selection, manifest = pinned_data
    if change == "overlap":
        manifest["roles"]["development"] = {"source_idx_start": 2, "source_idx_stop": 4, "rows": 2}
    elif change == "protected":
        manifest["roles"]["development"]["source_idx_stop"] = 7
    elif change == "missing":
        manifest["assets"]["topology"] = []
    elif change == "reorder":
        manifest["assets"]["topology"][0]["source_idx_min"] = 1
    else:
        manifest["graph_contract"]["node_feature_dim"] = 18
    path = root / selection.manifest
    path.write_text(json.dumps(manifest), encoding="utf-8")
    selection = replace(selection, manifest_sha256=sha256_file(path))
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid view reached pickle decoding")
    monkeypatch.setattr("molgap.packed_graph_dataset.PackedGraphDataset", forbidden)
    with pytest.raises(ValueError):
        load_training_data(root, selection)


def test_changed_shard_is_rejected_before_pickle(pinned_data, monkeypatch):
    root, selection, _ = pinned_data
    (root / "train.pt").write_bytes(b"invalid")
    def forbidden(*args, **kwargs):
        raise AssertionError("Changed bytes reached pickle decoding")
    monkeypatch.setattr("molgap.packed_graph_dataset.PackedGraphDataset", forbidden)
    with pytest.raises(ValueError, match="shard SHA"):
        load_training_data(root, selection)


@pytest.mark.parametrize("path", ["../manifest.json", "/manifest.json", "C:/manifest.json", "a\\b.json"])
def test_manifest_traversal_is_rejected(path):
    with pytest.raises(ValueError):
        DatasetSelection(path, "1" * 64, False)


def test_pinned_manifest_changes_fail(pinned_data):
    root, selection, _ = pinned_data
    with (root / selection.manifest).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="manifest SHA"):
        load_training_data(root, selection)


@pytest.mark.parametrize("algorithm", ["python-epoch-shuffle", "torch-pass-randperm"])
def test_sampler_resume_and_tail_are_scale_independent(algorithm):
    for rows in (7, 31, 107):
        complete = list(PassBatchSampler(rows, 3, 42, algorithm=algorithm, pass_index=2))
        resumed = list(PassBatchSampler(rows, 3, 42, algorithm=algorithm, pass_index=2, start_batch=1))
        assert resumed == complete[1:]
        assert len(complete) == rows // 3
        flat = [i for batch in complete for i in batch]
        assert len(set(flat)) == len(flat)
        assert min(flat) >= 0 and max(flat) < rows


def test_sampler_matches_existing_k1_and_gptrans_order_without_global_rng(monkeypatch):
    from molgap import k1_screen_training, pcqm_gptrans_v4
    monkeypatch.setattr(k1_screen_training, "TRAIN_ROWS", 300)
    monkeypatch.setattr(pcqm_gptrans_v4, "TRAIN_ROWS", 300)
    monkeypatch.setattr(pcqm_gptrans_v4, "BATCHES_PER_EPOCH", 2)
    expected_k1 = k1_screen_training.epoch_order(3)[:256]
    actual_k1 = list(PassBatchSampler(300, 128, 42, algorithm="python-epoch-shuffle", pass_index=3))
    assert [i for batch in actual_k1 for i in batch] == expected_k1
    rng = torch.get_rng_state()
    actual_gp = list(PassBatchSampler(300, 128, 42, algorithm="torch-pass-randperm", pass_index=3))
    assert actual_gp == list(pcqm_gptrans_v4.DeterministicEpochBatchSampler(300, 3))
    assert torch.equal(rng, torch.get_rng_state())
