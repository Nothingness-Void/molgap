from __future__ import annotations

import gzip
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from molgap.pcqm_official_edge_state import (
    OfficialEdgeStateConfig,
    PackedGraphDataset,
    _forward,
    _graph_from_row,
    _init_graph_worker,
    _make_model,
    _official_warmup_cosine_factor,
    accept_training_graphs,
    atomic_ogb_submission,
    atomic_npz,
    build_training_graph_shard,
    load_official_splits,
    prepare_training_rows,
    validate_official_splits,
    validate_submission_files,
)


def _mini_archive(path: Path) -> None:
    table = pd.DataFrame(
        {
            "idx": np.arange(8),
            "smiles": ["C", "CC", "CO", "N", "O", "C#N", "C=C", "F"],
            "homolumogap": [1.0, 2.0, 1.5, 3.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    split_buffer = io.BytesIO()
    torch.save(
        {
            "train": torch.tensor([0, 1, 2]),
            "valid": torch.tensor([3]),
            "test-dev": torch.tensor([4, 6]),
            "test-challenge": torch.tensor([5, 7]),
        },
        split_buffer,
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pcqm4m-v2/raw/data.csv.gz", gzip.compress(csv_bytes))
        archive.writestr("pcqm4m-v2/split_dict.pt", split_buffer.getvalue())


def test_official_split_and_graph_contract_is_exact_and_resumable(tmp_path: Path) -> None:
    archive = tmp_path / "pcqm4m-v2.zip"
    _mini_archive(archive)
    splits = load_official_splits(archive)
    assert validate_official_splits(splits)["counts"] == {
        "train": 3,
        "valid": 1,
        "test-dev": 2,
        "test-challenge": 2,
    }

    rows_dir = tmp_path / "rows"
    graph_dir = tmp_path / "graphs"
    manifest = prepare_training_rows(archive, rows_dir, source_shard_rows=3)
    assert manifest["counts"] == {"train": 3, "valid": 1}
    assert manifest["test_smiles_materialized"] is False
    assert prepare_training_rows(archive, rows_dir, source_shard_rows=3) == manifest

    for shard_index in range(len(manifest["shards"])):
        first = build_training_graph_shard(
            rows_dir, graph_dir, shard_index=shard_index, workers=1
        )
        second = build_training_graph_shard(
            rows_dir, graph_dir, shard_index=shard_index, workers=1
        )
        assert first == second

    acceptance_path = graph_dir / "acceptance.json"
    acceptance = accept_training_graphs(
        archive, rows_dir, graph_dir, acceptance_path
    )
    assert acceptance["status"] == "accepted"
    assert acceptance["counts"] == {"train": 3, "valid": 1}
    assert acceptance["test_graphs_built"] is False
    assert acceptance["external_data_used"] is False

    dataset = PackedGraphDataset(next((graph_dir / "train").glob("*.pt")))
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    config = OfficialEdgeStateConfig(
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        rwse_dim=16,
        edge_state_channels=8,
        max_epochs=1,
    )
    model = _make_model(config, acceptance["node_feature_dim"])
    prediction = _forward(model, batch)
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert model.edge_updates[0].source.weight.grad is not None


def test_submission_npz_uses_official_key_and_float32(tmp_path: Path) -> None:
    path = tmp_path / "submission.npz"
    atomic_npz(path, np.asarray([0.1, 0.2], dtype=np.float64))
    with np.load(path) as payload:
        assert payload.files == ["y_pred"]
        assert payload["y_pred"].shape == (2,)
        assert payload["y_pred"].dtype == np.float32


def test_official_evaluator_submission_and_acceptance(tmp_path: Path) -> None:
    atomic_ogb_submission(
        tmp_path,
        "test-dev",
        np.linspace(0.0, 1.0, 147_037, dtype=np.float64),
    )
    atomic_ogb_submission(
        tmp_path,
        "test-challenge",
        np.linspace(0.0, 1.0, 147_432, dtype=np.float64),
    )
    acceptance = validate_submission_files(tmp_path)
    assert acceptance["status"] == "accepted"
    assert acceptance["outputs"]["test-dev"]["rows"] == 147_037
    assert acceptance["outputs"]["test-challenge"]["rows"] == 147_432


def test_official_ogb_graph_contract_and_model_are_aligned(tmp_path: Path) -> None:
    archive = tmp_path / "pcqm4m-v2.zip"
    _mini_archive(archive)
    rows_dir = tmp_path / "rows"
    graph_dir = tmp_path / "ogb_graphs"
    manifest = prepare_training_rows(archive, rows_dir, source_shard_rows=3)
    for shard_index in range(len(manifest["shards"])):
        first = build_training_graph_shard(
            rows_dir,
            graph_dir,
            shard_index=shard_index,
            workers=1,
            feature_schema="ogb",
        )
        second = build_training_graph_shard(
            rows_dir,
            graph_dir,
            shard_index=shard_index,
            workers=1,
            feature_schema="ogb",
        )
        assert first == second

    acceptance = accept_training_graphs(
        archive,
        rows_dir,
        graph_dir,
        graph_dir / "acceptance.json",
        feature_schema="ogb",
    )
    assert acceptance["feature_schema"] == "ogb"
    assert acceptance["node_feature_dim"] == 9
    assert acceptance["edge_feature_dim"] == 3
    dataset = PackedGraphDataset(next((graph_dir / "train").glob("*.pt")))
    assert dataset._data.x.dtype == torch.long
    assert dataset._data.edge_attr.dtype == torch.long
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    config = OfficialEdgeStateConfig(
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        rwse_dim=16,
        edge_state_channels=8,
        max_epochs=1,
        feature_schema="ogb",
    )
    model = _make_model(config, acceptance["node_feature_dim"])
    prediction = _forward(model, batch)
    prediction.square().mean().backward()
    assert model.node_emb.embeddings[5].weight.grad is not None


def test_official_warmup_cosine_schedule_reaches_minimum() -> None:
    factors = [
        _official_warmup_cosine_factor(
            epoch,
            max_epochs=20,
            warmup_epochs=2,
            minimum_factor=0.0025,
        )
        for epoch in range(20)
    ]
    assert factors[:3] == [0.5, 1.0, 1.0]
    assert factors[-1] == 0.0025


def test_training_import_does_not_require_optional_ogb() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "ogb" or name.startswith("ogb."):
        raise ModuleNotFoundError("ogb deliberately blocked")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import molgap.pcqm_official_edge_state as module
assert callable(module.continue_official_edge_state)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_ims_continuation_payload_is_isolated_and_resumable() -> None:
    repo = Path(__file__).resolve().parents[1]
    continuation = (
        repo
        / "platforms"
        / "ims"
        / "pcqm_edge_state_full"
        / "continue_rich_full.pbs"
    ).read_text(encoding="utf-8")
    preflight = (
        repo
        / "platforms"
        / "ims"
        / "pcqm_edge_state_full"
        / "convergence_preflight_v2.pbs"
    ).read_text(encoding="utf-8")
    assert "--source-dir rich_full/training" in continuation
    assert "--output-dir rich_full/convergence_40" in continuation
    assert "--hard-job-budget-hours 13.5" in continuation
    assert "convergence_40/code/src" in continuation
    assert "continue-training \\" in continuation
    assert "+  --" not in continuation
    assert "job_status.json" in continuation
    assert "record_status failed" in continuation
    assert "convergence_40/code/src" in preflight
    assert "convergence_40/preflight_v2" in preflight
    assert "job_status.json" in preflight
    assert "record_status failed" in preflight


def test_hypervalent_official_smiles_uses_visible_unsanitized_fallback() -> None:
    _init_graph_worker((6, 14, 17), 4)
    graph, failure = _graph_from_row(
        (155174, "Cl[SiH]12C[SiH2]2C1", 7.613746, 0)
    )
    assert failure is None
    assert graph is not None
    assert int(graph.smiles_sanitized.item()) == 0
    assert torch.isfinite(graph.random_walk_pe).all()

    _init_graph_worker((6, 14, 17), 4, "ogb")
    rich, rich_failure = _graph_from_row(
        (155174, "Cl[SiH]12C[SiH2]2C1", 7.613746, 0)
    )
    assert rich_failure is None
    assert rich is not None
    assert rich.x.shape[1] == 9
    assert rich.edge_attr.shape[1] == 3
    assert rich.x.dtype == torch.long


def test_prepared_manifest_does_not_persist_test_smiles(tmp_path: Path) -> None:
    archive = tmp_path / "pcqm4m-v2.zip"
    _mini_archive(archive)
    rows_dir = tmp_path / "rows"
    prepare_training_rows(archive, rows_dir, source_shard_rows=8)
    text = "\n".join(
        gzip.open(path, "rt", encoding="utf-8").read()
        for path in rows_dir.glob("source_*.csv.gz")
    )
    assert "C#N" not in text
    assert "C=C" not in text
    manifest = json.loads((rows_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_rows"] == 4
