from __future__ import annotations

import ast
from pathlib import Path

import pytest

from molgap.pcqm_distance_angle_scale import (
    ARMS,
    BASELINE,
    CANDIDATE,
    EXPECTED_PARAMETER_COUNTS,
    DistanceAngleScaleConfig,
    _forward,
    _load_graphs,
    accept_pair,
    make_model,
    parameter_count,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_paired_contract() -> None:
    config = DistanceAngleScaleConfig()
    config.validate()
    assert (config.hidden_channels, config.num_layers, config.num_heads) == (192, 9, 4)
    assert (config.edge_state_channels, config.wedge_channels) == (64, 16)
    assert config.geometry_basis_channels == 16
    assert config.rwse_dim == 16
    assert config.dropout == pytest.approx(0.1)
    assert config.batch_size == 128
    assert config.loader_workers == 4
    assert config.prefetch_factor == 4
    assert config.learning_rate == pytest.approx(1.6e-4)
    assert config.weight_decay == pytest.approx(1.0e-6)
    assert config.epochs == 60
    assert config.seed == 42
    assert ARMS == (BASELINE, CANDIDATE)


@pytest.mark.parametrize("arm", ARMS)
def test_inference_parameter_count(arm: str) -> None:
    assert parameter_count(arm, DistanceAngleScaleConfig()) == EXPECTED_PARAMETER_COUNTS[arm]


def test_both_forward_contracts() -> None:
    import torch
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_wedge import WedgeData

    graph = WedgeData(
        x=torch.zeros((3, 9), dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.zeros((4, 3), dtype=torch.long),
        random_walk_pe=torch.zeros((3, 16)),
        wedge_edge_ids=torch.tensor([[0, 2], [3, 1]], dtype=torch.long),
        edge_distance=torch.ones((4, 1)),
        wedge_angle_cos=torch.zeros((2, 1)),
        geometry_valid=torch.ones((1,), dtype=torch.bool),
        y=torch.tensor([1.0]),
    )
    batch = next(iter(DataLoader([graph, graph], batch_size=2)))
    for arm in ARMS:
        model = make_model(arm, DistanceAngleScaleConfig())
        prediction = _forward(model, batch, arm)
        assert prediction.shape == (2,)
        prediction.sum().backward()


def test_packed_graph_loader(tmp_path) -> None:
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    class Dataset(InMemoryDataset):
        def __init__(self):
            super().__init__(root=None)
            self.data, self.slices = self.collate(
                [Data(x=torch.ones((2, 1))), Data(x=torch.ones((3, 1)))]
            )

    source = Dataset()
    path = tmp_path / "packed.pt"
    torch.save((source.data, source.slices), path)
    loaded = _load_graphs(path)
    assert len(loaded) == 2
    assert loaded[0].num_nodes == 2
    assert loaded[1].num_nodes == 3


def test_thin_scnet_entrypoint_parses() -> None:
    path = ROOT / "platforms" / "scnet" / "run_pcqm_distance_angle_500k.py"
    ast.parse(path.read_text(encoding="utf-8"))


def test_pair_acceptance_recomputes_mae_and_gate(tmp_path) -> None:
    import json
    import torch

    config = DistanceAngleScaleConfig().__dict__
    for arm, mae in ((BASELINE, 0.12), (CANDIDATE, 0.118)):
        root = tmp_path / arm
        root.mkdir(parents=True)
        best = root / "direct_gap_best.pt"
        last = root / "direct_gap_last.pt"
        torch.save({"arm": arm, "kind": "best"}, best)
        torch.save({"arm": arm, "kind": "last"}, last)
        target = torch.zeros(50_000)
        prediction = torch.full((50_000,), mae)
        torch.save(
            {
                "prediction_eV": prediction,
                "target_eV": target,
                "source_idx": torch.arange(500_000, 550_000),
            },
            root / "direct_gap_development.pt",
        )
        completion = {
            "format": "molgap-pcqm-distance-angle-500k-run-v1",
            "complete": True,
            "role": arm,
            "architecture": arm,
            "parameter_count": EXPECTED_PARAMETER_COUNTS[arm],
            "source_commit": "abc123",
            "config": config,
            "cache_aggregate_sha256": "cache-sha",
            "train_rows": 500_000,
            "development_rows": 50_000,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "gap": {
                "epochs_completed": 60,
                "best_epoch": 10,
                "best_development_mae_eV": mae,
                "best_sha256": sha256_file(best),
                "last_sha256": sha256_file(last),
            },
        }
        (root / "completion_manifest.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    result = accept_pair(tmp_path)
    assert result["accepted"] is True
    assert result["nominated"] is True
    assert result["candidate_minus_baseline_mae_eV"] == pytest.approx(
        -0.002, abs=2e-8
    )


def test_protocol_keeps_official_roles_sealed() -> None:
    source = (
        ROOT / "experiments" / "pcqm_distance_angle_500k" / "protocol.md"
    ).read_text(encoding="utf-8")
    assert "Official validation, test-dev, and test-challenge are not read" in source
    assert "physical batch 128" in source
    assert "60 direct-Gap epochs" in source
