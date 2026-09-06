import ast
from pathlib import Path
import subprocess

import torch
from torch_geometric.data import Batch

from molgap.pcqm_graph_state import make_graph_state
from molgap.pcqm_geometry_scratch import forward, config_for
from molgap.pcqm_wedge import WedgeData, directed_nonbacktracking_wedges


def test_classes_are_exact_ast_copy_of_confirmed_server_source():
    source = subprocess.check_output(["git", "show", "9068ddb82e6bdf16b841570abbff023b90c07f07:src/molgap/pcqm_gap_architecture.py"], text=True, encoding="utf-8")
    old = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)}
    import molgap.pcqm_graph_state as module
    current = ast.parse(Path(module.__file__).read_text())
    for node in current.body:
        if isinstance(node, ast.ClassDef):
            assert ast.dump(node) == ast.dump(old[node.name])


def test_frozen_graphstate_count_and_real_batched_gradients():
    torch.manual_seed(42)
    model = make_graph_state()
    assert sum(p.numel() for p in model.parameters()) == 3665809
    assert len(model.convs) == 9
    assert all(block.attn is None for block in model.convs)
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    wedge = directed_nonbacktracking_wedges(edge)
    graph = WedgeData(x=torch.zeros(3, 9, dtype=torch.long), edge_index=edge,
                     edge_attr=torch.zeros(4, 3, dtype=torch.long), random_walk_pe=torch.zeros(3, 16),
                     wedge_edge_ids=wedge, edge_distance=torch.ones(4, 1),
                     wedge_angle_cos=torch.zeros(len(wedge), 1), geometry_valid=torch.tensor([True]))
    value = forward(model, Batch.from_data_list([graph, graph]), "graphstate")
    assert value.shape == (2,)
    value.sum().backward()
    grad = model.graph_context.graph_to_atom.value.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert config_for("graphstate").learning_rate == 1.6e-4
