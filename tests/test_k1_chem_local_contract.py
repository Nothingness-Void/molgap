"""Synthetic topology/identity checks; no PCQM role or full model is loaded."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from molgap.k1_chem_local import COLORS, MODES, PARAMETERS, edge_colors


def _fixture():
    # OGB x[:, 0] = Z - 1; two disconnected molecules and one isolate.
    x = torch.zeros((7, 9), dtype=torch.long)
    x[:, 0] = torch.tensor([5, 5, 6, 7, 5, 15, 5])
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5],
        [1, 0, 2, 1, 3, 2, 5, 4],
    ])
    edge_attr = torch.zeros((8, 3), dtype=torch.long)
    edge_attr[:, 0] = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return x, edge_index, edge_attr


def test_atom_pair_colors_cover_real_directed_bonds_and_isolate():
    x, edges, attrs = _fixture()
    got = edge_colors(MODES[0], x, edges, attrs)
    assert got.tolist() == [0, 0, 1, 1, 2, 2, 1, 1]
    assert got.numel() == edges.shape[1]
    assert torch.all((got >= 0) & (got < COLORS))


def test_bond_type_control_and_node_permutation():
    x, edges, attrs = _fixture()
    assert edge_colors(MODES[1], x, edges, attrs).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    order = torch.tensor([3, 0, 5, 2, 6, 1, 4])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order))
    remapped = inverse[edges]
    for mode in MODES:
        assert torch.equal(edge_colors(mode, x[order], remapped, attrs),
                           edge_colors(mode, x, edges, attrs))


def test_distinct_modes_equal_capacity_and_frozen_contract():
    contract = json.loads((Path(__file__).parents[1] / "experiments" /
        "pcqm_k1_chem_local_100k" / "training_contract.json").read_text())
    assert contract["arms"] == list(MODES)
    assert PARAMETERS[MODES[0]] == PARAMETERS[MODES[1]] == 3_717_121
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32" and contract["tf32_enabled"] is False
    assert contract["total_optimizer_steps_per_arm"] == 31_240
    assert contract["official_validation_role_read"] is False
    assert contract["test_dev_role_read"] is False
