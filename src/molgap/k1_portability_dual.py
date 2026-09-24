"""Two isolated K1 interventions for topology portability screening.

Both arms retain the frozen K1 atom/bond/RWSE inputs and are exactly K1 at
initialization. They change different information paths, never each other.
"""
from __future__ import annotations


MODES = (
    "neural_atom_k1_rwse_refresh",
    "neural_atom_k1_degree_balance",
)
EXCHANGE_LAYERS = (3, 6, 9)
HIDDEN = 192
RWSE = 16


def make_encoder(mode: str):
    if mode not in MODES:
        raise ValueError(mode)

    import torch
    import torch.nn as nn
    from torch_geometric.nn import global_mean_pool

    from .qm9_neural_atom import make_encoder as make_k1

    class K1PortabilityArm(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.base = make_k1("neural_atom_k1")
            if mode == MODES[0]:
                self.rwse_refresh = nn.ModuleDict({
                    str(layer): nn.Linear(RWSE, HIDDEN, bias=False)
                    for layer in EXCHANGE_LAYERS
                })
                for projection in self.rwse_refresh.values():
                    nn.init.zeros_(projection.weight)
            else:
                self.degree_balance = nn.Parameter(torch.zeros(9, HIDDEN))

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            if tuple(random_walk_pe.shape) != (x.shape[0], RWSE):
                raise ValueError("RWSE16 shape changed")
            rwse = random_walk_pe.float()
            hidden = self.base._embed_nodes(x)
            hidden = hidden + self.base.rwse_encoder(rwse)
            edge_state = self.base._embed_edges(edge_attr)
            if self.mode == MODES[1]:
                degree = torch.bincount(
                    edge_index[1], minlength=x.shape[0]
                ).to(dtype=hidden.dtype)
                mean_degree = global_mean_pool(degree[:, None], batch).squeeze(-1)
                centered_degree = torch.log1p(degree) - torch.log1p(
                    mean_degree[batch]
                )
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(hidden, edge_index, edge_state)
                updated = block(hidden, edge_index, batch, edge_attr=edge_state)
                if self.mode == MODES[1]:
                    # Calibrate the existing local update; add no new messages.
                    update = updated - hidden
                    updated = updated + centered_degree[:, None] * update * torch.tanh(
                        self.degree_balance[layer - 1]
                    )
                hidden = updated
                if layer in EXCHANGE_LAYERS:
                    if self.mode == MODES[0]:
                        # Restore topology signal immediately before global exchange.
                        hidden = hidden + self.rwse_refresh[str(layer)](rwse)
                    hidden = self.base.neural_atom_mixers[str(layer)](hidden, batch)
            return self.base._pool(hidden, batch)

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
            )

    return K1PortabilityArm()


def check_mechanism(model, batch):
    """Check only input/topology identity and zero-start intervention."""
    import torch

    if model.mode == MODES[0]:
        zero = all(
            torch.count_nonzero(module.weight).item() == 0
            for module in model.rwse_refresh.values()
        )
        return {
            "mechanism": "layerwise-rwse16-refresh",
            "layers": list(EXCHANGE_LAYERS),
            "rwse_shape": list(batch.random_walk_pe.shape),
            "zero_start": zero,
            "source_from_fixed_graph_only": True,
        }
    degree = torch.bincount(batch.edge_index[1], minlength=batch.x.shape[0])
    return {
        "mechanism": "relative-degree-local-update-calibration",
        "layers": list(range(1, 10)),
        "real_bond_degree_shape": list(degree.shape),
        "zero_start": bool(torch.count_nonzero(model.degree_balance).item() == 0),
        "source_from_fixed_graph_only": True,
    }
