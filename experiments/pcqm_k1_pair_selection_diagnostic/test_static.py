"""No-training checks for the frozen PairToken diagnostic."""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from molgap.pcqm_k1_pair_selection_diagnostic import attention_metrics


ROOT = Path(__file__).resolve().parents[2]


def test_attention_mass_partition_on_two_atom_bond():
    assignment = torch.tensor([[[0.1, 0.4], [0.4, 0.1]]])
    valid = torch.tensor([[True, True]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    node_batch = torch.zeros(2, dtype=torch.long)
    result = attention_metrics(assignment, valid, edge_index, node_batch)
    assert torch.allclose(result["top20_mass"], torch.tensor([0.4]))
    assert torch.allclose(result["bonded_mass"], torch.tensor([0.8]))
    assert torch.allclose(result["diagonal_mass"], torch.tensor([0.2]))
    assert torch.allclose(result["two_hop_mass"], torch.tensor([0.0]))
    assert torch.allclose(result["nonlocal_mass"], torch.tensor([0.0]))


def test_two_hop_pair_not_double_counted():
    assignment = torch.full((1, 3, 3), 1.0 / 9.0)
    valid = torch.ones((1, 3), dtype=torch.bool)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    result = attention_metrics(
        assignment, valid, edge_index, torch.zeros(3, dtype=torch.long)
    )
    assert torch.allclose(result["bonded_mass"], torch.tensor([4.0 / 9.0]))
    assert torch.allclose(result["two_hop_mass"], torch.tensor([2.0 / 9.0]))
    assert torch.allclose(result["diagonal_mass"], torch.tensor([3.0 / 9.0]))


def test_diagnostic_has_no_optimizer_or_training_step():
    source = (ROOT / "src/molgap/pcqm_k1_pair_selection_diagnostic.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    for forbidden in ("optimizer.step(", "loss.backward(", "model.train("):
        assert forbidden not in source
