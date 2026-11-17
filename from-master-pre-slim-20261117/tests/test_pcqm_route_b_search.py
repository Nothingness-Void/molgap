from __future__ import annotations

import torch
from torch_geometric.data import Data, InMemoryDataset

import molgap.pcqm_route_b_search as search
from molgap.pcqm_route_b_search import (
    DEV_ROWS,
    GPS_TRIALS,
    SCHNET_CUTOFF_ANGSTROM,
    SCHNET_TRIALS,
    TRAIN_CORE_ROWS,
    TRAIN_EXTENSION_ROWS,
    _ranked_subset,
    _scheduler_lambda,
    build_nested_subsets,
    search_trials,
    verify_nested_subsets,
)
from molgap.pcqm_route_b_training import CONFIGS
from molgap.schnet import SchNetWrapper


def test_route_b_search_space_is_fixed_and_complete() -> None:
    assert len(GPS_TRIALS) == 12
    assert len(SCHNET_TRIALS) == 12
    assert search_trials("gps9") is GPS_TRIALS
    assert search_trials("gps11_160") is GPS_TRIALS
    assert search_trials("primary_schnet") is SCHNET_TRIALS
    assert search_trials("augmented_schnet") is SCHNET_TRIALS
    assert SCHNET_CUTOFF_ANGSTROM == 6.0


def test_nested_selection_is_deterministic() -> None:
    values = torch.arange(200_000).numpy()
    selected_50 = _ranked_subset(values, TRAIN_CORE_ROWS, 123)
    selected_100 = _ranked_subset(
        values, TRAIN_CORE_ROWS + TRAIN_EXTENSION_ROWS, 123
    )
    selected_dev = _ranked_subset(values, DEV_ROWS, 124)
    assert set(selected_50).issubset(set(selected_100))
    assert len(set(selected_50)) == TRAIN_CORE_ROWS
    assert len(set(selected_dev)) == DEV_ROWS


def test_warmup_cosine_schedule_is_positive() -> None:
    values = [
        _scheduler_lambda(epoch, epochs=20, warmup_ratio=0.1)
        for epoch in range(20)
    ]
    assert values[0] < values[1]
    assert all(value > 0 for value in values)
    assert values[-1] <= 0.011


def _write_packed(path, source_indices) -> None:
    graphs = [
        Data(
            x=torch.ones(2, 18),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.ones(2, 4),
            z=torch.tensor([6, 6]),
            pos=torch.zeros(2, 3),
            charges=torch.zeros(2),
            y=torch.tensor([float(index)]),
            source_idx=torch.tensor([index]),
        )
        for index in source_indices
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(InMemoryDataset.collate(graphs), path)


def test_nested_subset_builder_aligns_modalities(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(search, "TRAIN_CORE_ROWS", 4)
    monkeypatch.setattr(search, "TRAIN_EXTENSION_ROWS", 3)
    monkeypatch.setattr(search, "DEV_ROWS", 2)
    monkeypatch.setattr(search, "SHARD_ROWS", 3)
    source = tmp_path / "source"
    for modality in ("gps", "primary", "secondary"):
        _write_packed(
            source / modality / "train_shard_000.pt",
            range(10),
        )
        _write_packed(
            source / modality / "dev_shard_000.pt",
            range(10, 15),
        )
    output = tmp_path / "subsets"
    result = build_nested_subsets(source_root=source, output_root=output)
    accepted = verify_nested_subsets(output)
    assert result["status"] == "complete"
    assert accepted["counts"]["gps"] == {
        "core": 4,
        "extension": 3,
        "dev": 2,
    }
    assert accepted["source_idx_sha256"] == result["source_idx_sha256"]
    (output / "manifest.json").unlink()
    resumed = build_nested_subsets(source_root=source, output_root=output)
    assert resumed["source_idx_sha256"] == result["source_idx_sha256"]


def test_schnet_search_reinitializes_six_angstrom_basis(tmp_path) -> None:
    source_config = {
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "dropout": 0.05,
    }
    source = SchNetWrapper(
        **source_config,
        use_charges=True,
        n_targets=3,
    )
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "model_config": source_config,
            "target_mean": torch.zeros(3),
            "target_std": torch.ones(3),
        },
        checkpoint,
    )
    model, report, _, _ = search._search_model(
        CONFIGS["primary_schnet"],
        SCHNET_TRIALS[0],
        checkpoint,
        torch.device("cpu"),
    )
    assert model.schnet.cutoff == 6.0
    assert torch.isclose(
        model.schnet.distance_expansion.offset[-1],
        torch.tensor(6.0),
    )
    assert report["distance_basis_reinitialized"] is True
