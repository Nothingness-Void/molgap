"""Small synthetic encoders only; no repository molecular model is run."""
from dataclasses import replace
import pytest
import torch
from torch import nn
from torch_geometric.data import Data, Batch

from molgap.gptrans_cycle_pair import CommonCycleGPTrans, cycle_categories, CYCLE, PARENT
from molgap.gptrans_execution_backend import prepare_layout
from molgap.family_training_runner import _probe_recipe, family_arm_bindings
from molgap.family_training_data import DatasetSelection
from test_family_training_recipe import recipe


def synthetic_batch():
    graph = Data(x=torch.zeros(4, 9, dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]),
        edge_attr=torch.zeros(8, 3, dtype=torch.long),
        cycle_pair_index=torch.tensor([(i, j) for i in range(4) for j in range(4) if i != j]).t(),
        cycle_pair_category=torch.tensor([1 if (i-j) % 4 in (1, 3) else 2
            for i in range(4) for j in range(4) if i != j]))
    return Batch.from_data_list([graph, graph])


def test_cpu_layout_offsets_virtual_and_padding():
    batch = prepare_layout(synthetic_batch())
    categories = batch.execution_layout["cycle_categories"]
    assert categories.shape == (2, 5, 5)
    assert categories[:, 0].count_nonzero() == categories[:, :, 0].count_nonzero() == 0
    assert torch.equal(categories[0], categories[1])
    assert categories[0, 1, 3] == 2 and categories[0, 1, 2] == 1
    batch.cycle_pair_index[1, 0] = 4
    with pytest.raises(ValueError, match="crosses"):
        cycle_categories(batch, batch.execution_layout)


class MixPair(nn.Module):
    def forward(self, node, pair, padding, edges):
        return node + pair.sum((1, 2, 3))[:, None, None], pair


class SmallCycle(CommonCycleGPTrans):
    def __init__(self, categories):
        nn.Module.__init__(self)
        self.cycle_embedding = nn.Embedding(3, 32, padding_idx=0)
        nn.init.zeros_(self.cycle_embedding.weight)
        self.blocks = nn.ModuleList([MixPair()])
        self.readout = nn.Linear(36, 1, bias=False)
        nn.init.ones_(self.readout.weight)
        self._execution_layout = {"cycle_categories": categories}

    def _dense_inputs(self, *args):
        return torch.ones(2, 5, 4), torch.ones(2, 32, 5, 5), torch.zeros(2, 1, 1, 5, dtype=torch.bool)

    def _local_edges(self, *args):
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty


def test_zero_start_active_gradient_and_no_virtual_encoding():
    batch = prepare_layout(synthetic_batch())
    model = SmallCycle(batch.execution_layout["cycle_categories"])
    output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert torch.equal(output, torch.full((2, 1), 3236.))
    output.sum().backward()
    grad = model.cycle_embedding.weight.grad
    assert grad[0].count_nonzero() == 0
    assert grad[1:].abs().min() > 0
    torch.optim.AdamW(model.parameters(), lr=.001).step()
    assert model.cycle_embedding.weight[0].count_nonzero() == 0
    assert not torch.equal(output, model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))


def test_optional_sidecar_does_not_change_legacy_recipe_bytes():
    baseline = replace(recipe("gptrans_t"), mode=PARENT)
    assert "input_sidecar" not in baseline.to_dict()
    candidate = replace(baseline, mode=CYCLE,
        input_sidecar=DatasetSelection("cycles/manifest.json", "3" * 64, True))
    assert type(candidate).from_dict(candidate.to_dict()) == candidate
    probe = _probe_recipe(candidate)
    assert not probe.input_sidecar.development and not probe.dataset.development
    assert family_arm_bindings(baseline)["objective"] == family_arm_bindings(candidate)["objective"]
    with pytest.raises(ValueError):
        replace(baseline, mode=CYCLE)
