from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def _load_train_encoder():
    path = (
        Path(__file__).parents[1]
        / "production"
        / "03_train"
        / "scripts"
        / "training"
        / "train_encoder.py"
    )
    spec = importlib.util.spec_from_file_location("phase8_train_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_roles_are_seeded_and_balanced() -> None:
    module = _load_train_encoder()
    first = module._split_roles(100, 42)
    repeated = module._split_roles(100, 42)
    alternate = module._split_roles(100, 43)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, alternate)
    assert [(first == role).sum() for role in range(3)] == [80, 10, 10]


def test_packed_gps_matches_legacy_training_step() -> None:
    module = _load_train_encoder()
    graphs = [
        Data(x=torch.rand(n, 9), edge_index=torch.tensor([[0, 1], [1, 0]]),
             edge_attr=torch.rand(2, 4), y=torch.rand(1, 3),
             source_idx=torch.tensor([i]))
        for i, n in enumerate((2, 5, 3))
    ]
    batch = next(iter(module._encoder_loader(
        graphs, batch_size=3, retention_boundary=2, retention_targets=torch.cat([g.y for g in graphs]),
    )))
    torch.testing.assert_close(batch.retention_rows, torch.tensor([0, 1]))
    torch.testing.assert_close(batch.retention_teacher, batch.y[:2])
    model = module.GPSWrapper(hidden_channels=16, num_heads=4, num_layers=2)
    reference = deepcopy(model)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        actual = module._forward("gps", model, batch)
        actual.square().sum().backward()
        torch.manual_seed(42)
        expected = reference(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)

    model.eval()
    reference.eval()
    inputs = (batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    with torch.no_grad():
        torch.testing.assert_close(module._forward("gps", model, batch), reference(*inputs))
        torch.testing.assert_close(module._encode("gps", model, batch), reference.encode(*inputs))

    conv = type(model.convs[0])(16, deepcopy(model.convs[0].conv), heads=4, norm="graph_norm").eval()
    with torch.no_grad():
        x, edges = model.node_emb(batch.x), model.edge_emb(batch.edge_attr)
        args = (x, batch.edge_index, batch.batch)
        torch.testing.assert_close(conv(*args, layout=batch.gps_layout, edge_attr=edges), conv(*args, edge_attr=edges))
