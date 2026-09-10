from __future__ import annotations

import ast
from pathlib import Path

from molgap.pcqm_edgestate_scale import (
    EXPECTED_PARAMETER_COUNT,
    EdgeStateScaleConfig,
    _load_graphs,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_equal_exposure_contract() -> None:
    config = EdgeStateScaleConfig()
    config.validate()
    assert config.hidden_channels == 304
    assert config.num_layers == 9
    assert config.num_heads == 4
    assert config.batch_size == 128
    assert config.scratch_epochs == config.pretrain_epochs + config.finetune_epochs
    assert (config.scratch_epochs, config.pretrain_epochs, config.finetune_epochs) == (
        60,
        20,
        40,
    )


def test_inference_parameter_count() -> None:
    assert parameter_count(EdgeStateScaleConfig()) == EXPECTED_PARAMETER_COUNT
    assert EXPECTED_PARAMETER_COUNT == 11_270_993


def test_thin_scnet_entrypoint_parses() -> None:
    path = ROOT / "platforms" / "scnet" / "run_pcqm_edgestate304_500k.py"
    ast.parse(path.read_text(encoding="utf-8"))


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


def test_protocol_keeps_official_roles_sealed() -> None:
    source = (
        ROOT / "experiments" / "pcqm_edgestate304_500k" / "protocol.md"
    ).read_text(encoding="utf-8")
    assert "Official validation, test-dev, and test-challenge are not read" in source
    assert "exactly 60 passes" in source
