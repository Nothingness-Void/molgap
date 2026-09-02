from __future__ import annotations

import pytest
import builtins

pytest.importorskip("torch")

import torch
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.data import InMemoryDataset

from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder
from molgap.pcqm_geometry_warmstart import (
    CANDIDATE,
    GeometryWarmstartConfig,
    _load_source_checkpoint,
    build_geometry_shard,
    load_pretrained_backbone,
)
from molgap.pcqm_official_edge_state import OfficialEdgeStateConfig, _make_model
from molgap.pcqm_official_edge_state import atomic_json, atomic_torch, sha256_file
from molgap.pcqm_wedge import WedgeData, directed_nonbacktracking_wedges


def _graph() -> WedgeData:
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    wedges = directed_nonbacktracking_wedges(edge_index)
    return WedgeData(
        x=torch.zeros((3, 9), dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.zeros((4, 3), dtype=torch.long),
        random_walk_pe=torch.zeros((3, 16), dtype=torch.float32),
        wedge_edge_ids=wedges,
        edge_distance=torch.ones((4, 1), dtype=torch.float32),
        wedge_angle_cos=torch.zeros((len(wedges), 1), dtype=torch.float32),
        geometry_valid=torch.tensor([True]),
        y=torch.tensor([1.0]),
        source_idx=torch.tensor([0]),
    )


def test_warmstart_maps_every_source_tensor_and_preserves_initial_function():
    source = _make_model(OfficialEdgeStateConfig(feature_schema="ogb"), 9).eval()
    target = make_pcqm_gap_encoder(CANDIDATE).eval()
    report = load_pretrained_backbone(target, source.state_dict())
    assert report["mapped_tensor_count"] == len(source.state_dict())
    assert report["new_tensor_count"] > 0
    batch = Batch.from_data_list([_graph(), _graph()])
    with torch.no_grad():
        source_value = source(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.random_walk_pe,
        )
        target_value = target(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.random_walk_pe, batch.wedge_edge_ids, batch.edge_distance,
            batch.wedge_angle_cos, batch.geometry_valid,
        )
    torch.testing.assert_close(source_value, target_value, atol=2e-6, rtol=2e-6)


def test_warmstart_mapping_accepts_local_categorical_encoder_keys():
    source = _make_model(OfficialEdgeStateConfig(feature_schema="ogb"), 9)
    target = _make_model(OfficialEdgeStateConfig(feature_schema="ogb"), 9)
    report = load_pretrained_backbone(target, source.state_dict())
    assert report["mapped_tensor_count"] == len(source.state_dict())
    assert report["new_tensor_count"] == 0


def test_geometry_encoder_falls_back_when_ogb_package_is_absent(monkeypatch):
    original_import = builtins.__import__

    def without_ogb(name, *args, **kwargs):
        if name == "ogb" or name.startswith("ogb."):
            error = ModuleNotFoundError("No module named 'ogb'")
            error.name = "ogb"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ogb)
    source = _make_model(OfficialEdgeStateConfig(feature_schema="ogb"), 9).eval()
    target = make_pcqm_gap_encoder(CANDIDATE).eval()
    report = load_pretrained_backbone(target, source.state_dict())
    assert report["mapped_tensor_count"] == len(source.state_dict())
    assert "node_emb.embeddings.0.weight" in target.state_dict()


def test_warmstart_contract_is_bounded_and_uses_two_learning_rates():
    config = GeometryWarmstartConfig()
    assert config.max_epochs == 12
    assert config.max_projected_training_s == 12 * 3600
    assert config.new_learning_rate > config.shared_learning_rate
    assert config.minimum_memory_headroom_fraction >= 0.15


def test_continuation_checkpoint_uses_hashed_base_config(tmp_path):
    config = OfficialEdgeStateConfig(feature_schema="ogb")
    source = _make_model(config, 9)
    acceptance = {
        "node_feature_dim": 9,
        "target_mean_gap": 5.5,
        "target_std_gap": 1.2,
    }
    base_path = tmp_path / "base.pt"
    atomic_torch(base_path, {
        "config": config.__dict__,
        "model": source.state_dict(),
        "acceptance_sha256": "accepted-graphs",
    })
    continuation_path = tmp_path / "continuation.pt"
    atomic_torch(continuation_path, {
        "format": "molgap-pcqm4mv2-official-edge-state-continuation-best-v1",
        "model": source.state_dict(),
        "best_epoch": 30,
        "target_mean_gap": 5.5,
        "target_std_gap": 1.2,
        "acceptance_sha256": "accepted-graphs",
        "source_best_sha256": sha256_file(base_path),
    })
    checkpoint, loaded = _load_source_checkpoint(
        continuation_path, acceptance, base_path
    )
    assert checkpoint["best_epoch"] == 30
    assert set(loaded.state_dict()) == set(source.state_dict())

    changed_path = tmp_path / "changed.pt"
    changed_path.write_bytes(base_path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="does not reference"):
        _load_source_checkpoint(continuation_path, acceptance, changed_path)


def test_geometry_shard_is_atomic_resumable_and_batch_aligned(tmp_path):
    rows_dir = tmp_path / "rows"
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "geometry"
    rows_dir.mkdir()
    (base_dir / "train").mkdir(parents=True)
    row_path = rows_dir / "source_0000000_0000002.csv.gz"
    pd.DataFrame({
        "idx": [0, 1], "smiles": ["CCO", "CCO"],
        "gap": [5.0, 5.1], "split_code": [0, 0],
    }).to_csv(row_path, index=False, compression="gzip")
    source_record = {
        "path": row_path.name, "source_start": 0, "source_end": 2,
        "rows": 2, "counts": {"train": 2, "valid": 0},
        "bytes": row_path.stat().st_size, "sha256": sha256_file(row_path),
    }
    atomic_json(rows_dir / "manifest.json", {
        "status": "complete", "counts": {"train": 2, "valid": 0},
        "shards": [source_record],
    })
    graph = _graph()
    graph.source_idx = torch.tensor([0])
    graph.edge_attr = torch.zeros((4, 3), dtype=torch.long)
    second_graph = _graph()
    second_graph.source_idx = torch.tensor([1])
    second_graph.edge_attr = torch.zeros((4, 3), dtype=torch.long)
    data, slices = InMemoryDataset.collate([graph, second_graph])
    base_path = base_dir / "train" / "train_shard_0000.pt"
    atomic_torch(base_path, (data, slices))

    first = build_geometry_shard(
        rows_dir, base_dir, output_dir, shard_index=0, workers=1,
    )
    second = build_geometry_shard(
        rows_dir, base_dir, output_dir, shard_index=0, workers=1,
    )
    assert first == second
    assert first["status"] == "complete"
    assert first["counts"] == {"train": 2, "valid": 0}
    path = output_dir / first["outputs"][0]["path"]
    payload, slices = torch.load(path, weights_only=False)
    assert payload.edge_distance.shape == (8, 1)
    assert payload.wedge_angle_cos.shape == (4, 1)
    assert slices["source_idx"].tolist() == [0, 1, 2]
