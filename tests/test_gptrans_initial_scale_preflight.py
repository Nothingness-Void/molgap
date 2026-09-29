"""Small deterministic checks for the GPTrans input-scale CPU diagnostic."""

import pytest
import torch

from molgap.gptrans_initial_scale_preflight import component_energy


def test_component_energy_counts_actual_degree_and_atom_categories():
    state = {
        f"atom_encoder.atom_embedding_list.{column}.weight": torch.full((3, 256), float(column + 1))
        for column in range(9)
    }
    state["in_degree_encoder.weight"] = torch.arange(512).float()[:, None].repeat(1, 256)
    state["out_degree_encoder.weight"] = state["in_degree_encoder.weight"].clone()
    x = torch.zeros((3, 9), dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    result = component_energy(x, edge_index, state)
    assert result["nodes"] == 3
    assert result["degree_max"] == 2
    assert result["atom_squared_sum"] == pytest.approx(3 * 256 * 45 ** 2)
    assert result["degree_squared_sum"] == pytest.approx(256 * (2 ** 2 + 4 ** 2 + 2 ** 2))
    assert result["cross_sum"] == pytest.approx(256 * 45 * (2 + 4 + 2))


def test_component_energy_rejects_unavailable_category():
    state = {f"atom_encoder.atom_embedding_list.{column}.weight": torch.zeros(2, 256)
             for column in range(9)}
    state["in_degree_encoder.weight"] = torch.zeros(512, 256)
    state["out_degree_encoder.weight"] = torch.zeros(512, 256)
    x = torch.zeros((1, 9), dtype=torch.long)
    x[0, 0] = 2
    with pytest.raises(ValueError, match="category"):
        component_energy(x, torch.empty((2, 0), dtype=torch.long), state)
