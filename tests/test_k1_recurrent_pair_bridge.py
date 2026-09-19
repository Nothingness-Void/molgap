import torch
from torch_geometric.data import Batch, Data

from molgap.k1_recurrent_pair_bridge import (
    ADDED_PARAMETERS,
    EXCHANGE_LAYERS,
    MODE,
    PARAMETERS,
    check_mechanism,
)
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS, make_encoder


def _batch():
    graphs = []
    for offset, nodes in enumerate((3, 5)):
        source = torch.arange(nodes - 1)
        target = source + 1
        edge_index = torch.cat(
            [torch.stack([source, target]), torch.stack([target, source])], dim=1
        )
        graphs.append(
            Data(
                x=torch.zeros((nodes, 9), dtype=torch.long),
                edge_index=edge_index,
                edge_attr=torch.zeros((edge_index.shape[1], 3), dtype=torch.long),
                random_walk_pe=torch.zeros((nodes, 16)),
                y=torch.tensor([float(offset)]),
                source_idx=torch.tensor([offset]),
            )
        )
    return Batch.from_data_list(graphs)


def test_recurrent_pair_bridge_is_exactly_nested_in_k1():
    batch = _batch()
    torch.manual_seed(42)
    reference = make_encoder("neural_atom_k1_v4").eval()
    torch.manual_seed(42)
    candidate = make_encoder(MODE).eval()
    with torch.no_grad():
        expected = reference(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
        observed = candidate(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
    assert torch.equal(expected, observed)
    assert candidate.base.state_dict().keys() == reference.state_dict().keys()
    assert sum(parameter.numel() for parameter in candidate.parameters()) == PARAMETERS[MODE]
    assert PARAMETERS[MODE] - 3_658_817 == ADDED_PARAMETERS
    assert ARCHITECTURE_CONFIGS[MODE]["exchange_layers"] == list(EXCHANGE_LAYERS)


def test_recurrent_pair_bridge_mechanism_and_gradient_are_live():
    batch = _batch()
    torch.manual_seed(42)
    candidate = make_encoder(MODE)
    checks = check_mechanism(candidate, batch)
    assert checks["return_projections_zero"] is True
    assert all(
        layer["recurrent_addition_exact"]
        and layer["valid_target_mass_one"]
        and layer["padding_assignment_zero"]
        and layer["padding_pair_state_zero"]
        and layer["zero_initial_update_exact"]
        for layer in checks["layers"]
    )

    optimizer = torch.optim.AdamW(candidate.parameters(), lr=4e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = candidate(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        ).square().mean()
        loss.backward()
        optimizer.step()
    bridge_gradients = [
        parameter.grad
        for parameter in candidate.recurrent_pair_bridge.parameters()
        if parameter.grad is not None
    ]
    assert bridge_gradients
    assert all(torch.isfinite(gradient).all() for gradient in bridge_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in bridge_gradients) > 0
