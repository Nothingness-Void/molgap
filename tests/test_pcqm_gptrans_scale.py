from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from molgap.gptrans import OGBGPTransTiny
from molgap.pcqm_gptrans_scale import (
    GPTransScaleConfig,
    _largest_training_graphs,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def _graph(node_count: int) -> Data:
    source = torch.arange(node_count - 1)
    target = source + 1
    edge_index = torch.cat(
        (torch.stack((source, target)), torch.stack((target, source))), dim=1
    )
    return Data(
        x=torch.zeros((node_count, 9), dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.zeros((edge_index.shape[1], 3), dtype=torch.long),
        y=torch.tensor([1.0]),
    )


def test_frozen_gptrans_t_contract() -> None:
    config = GPTransScaleConfig()
    config.validate()
    assert (config.num_layers, config.node_channels, config.pair_channels) == (
        12,
        256,
        32,
    )
    assert (config.num_heads, config.batch_size, config.epochs) == (8, 128, 60)
    assert config.learning_rate == 1.0e-3
    assert config.ema_decay == 0.9999


def test_gptrans_forward_backward_and_parameter_budget() -> None:
    model = OGBGPTransTiny()
    batch = Batch.from_data_list([_graph(4), _graph(6)])
    prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()
    prediction.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert parameter_count(GPTransScaleConfig()) == 5_246_817


def test_thin_entrypoint_and_sealed_protocol() -> None:
    runner = ROOT / "platforms" / "scnet" / "run_pcqm_gptrans_t_500k.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    protocol = (
        ROOT / "experiments" / "pcqm_gptrans_t_500k" / "protocol.md"
    ).read_text(encoding="utf-8")
    assert "Official validation, test-dev, and test-challenge are not read" in protocol
    assert "physical batch 128" in protocol
    assert "offline multi-hop edge-sequence" in protocol


def test_largest_graph_gate_uses_every_training_shard(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save([_graph(2), _graph(8)], first)
    torch.save([_graph(5), _graph(11)], second)
    manifest = {
        "shards": [
            {"role": "train", "file": first.name},
            {"role": "train", "file": second.name},
        ]
    }
    largest = _largest_training_graphs(tmp_path, manifest, 2)
    assert [graph.num_nodes for graph in largest] == [11, 8]
