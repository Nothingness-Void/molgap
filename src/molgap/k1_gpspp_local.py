"""Zero-initialized GPS++-style directional local adapters for frozen K1."""
from __future__ import annotations

from .qm9_neural_atom import MIXER_LAYERS


MODES = (
    "neural_atom_k1_gpspp_sender",
    "neural_atom_k1_gpspp_bidirectional",
)
HIDDEN_CHANNELS = 192
EDGE_CHANNELS = 64
BOTTLENECK_CHANNELS = 128
LAYERS = 9
ADAPTER_PARAMETERS_PER_LAYER = 95_232
ADDED_PARAMETERS = LAYERS * ADAPTER_PARAMETERS_PER_LAYER
PARAMETERS = {mode: 4_515_905 for mode in MODES}


def _mean_aggregate(values, index, node_count):
    """Mean directed-bond messages at one chosen endpoint."""
    import torch

    total = values.new_zeros((node_count, values.shape[-1]))
    total.index_add_(0, index, values)
    count = values.new_zeros((node_count, 1))
    count.index_add_(0, index, torch.ones_like(values[:, :1]))
    return total / count.clamp_min_(1.0)


class _DirectionalAdapterFactory:
    @staticmethod
    def make(*, use_receiver: bool):
        import torch
        import torch.nn as nn

        class DirectionalLocalAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_receiver = use_receiver
                self.node_norm = nn.LayerNorm(HIDDEN_CHANNELS)
                self.edge_norm = nn.LayerNorm(EDGE_CHANNELS)
                self.source = nn.Linear(
                    HIDDEN_CHANNELS, EDGE_CHANNELS, bias=False
                )
                self.target = nn.Linear(
                    HIDDEN_CHANNELS, EDGE_CHANNELS, bias=False
                )
                self.edge = nn.Linear(EDGE_CHANNELS, EDGE_CHANNELS)
                self.message_norm = nn.LayerNorm(EDGE_CHANNELS)
                self.input_projection = nn.Linear(
                    HIDDEN_CHANNELS + 2 * EDGE_CHANNELS,
                    BOTTLENECK_CHANNELS,
                )
                self.activation = nn.SiLU()
                self.return_projection = nn.Linear(
                    BOTTLENECK_CHANNELS, HIDDEN_CHANNELS
                )
                nn.init.zeros_(self.return_projection.weight)
                nn.init.zeros_(self.return_projection.bias)

            def components(self, hidden, edge_index, edge_state):
                source, target = edge_index
                normalized = self.node_norm(hidden)
                message = self.message_norm(
                    self.activation(
                        self.source(normalized[source])
                        + self.target(normalized[target])
                        + self.edge(self.edge_norm(edge_state))
                    )
                )
                sender = _mean_aggregate(message, source, hidden.shape[0])
                receiver = _mean_aggregate(message, target, hidden.shape[0])
                if not self.use_receiver:
                    receiver = receiver.new_zeros(receiver.shape)
                return normalized, receiver, sender, message

            def forward(self, hidden, edge_index, edge_state):
                normalized, receiver, sender, _ = self.components(
                    hidden, edge_index, edge_state
                )
                features = self.activation(
                    self.input_projection(
                        torch.cat((normalized, receiver, sender), dim=-1)
                    )
                )
                return self.return_projection(features)

        return DirectionalLocalAdapter()


def make_encoder(mode):
    if mode not in MODES:
        raise ValueError(mode)

    import torch.nn as nn

    from .qm9_neural_atom import make_encoder as frozen_encoder

    class GPSPPDirectionalLocalK1(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.base = frozen_encoder("neural_atom_k1")
            self.local_adapters = nn.ModuleList(
                [
                    _DirectionalAdapterFactory.make(
                        use_receiver=(mode == MODES[1])
                    )
                    for _ in range(LAYERS)
                ]
            )

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            expected = (x.shape[0], self.base.rwse_dim)
            if tuple(random_walk_pe.shape) != expected:
                raise ValueError(
                    f"random_walk_pe must have shape {expected}, "
                    f"got {tuple(random_walk_pe.shape)}"
                )
            hidden = self.base._embed_nodes(x)
            hidden = hidden + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            for layer, (edge_update, block, adapter) in enumerate(
                zip(
                    self.base.edge_updates,
                    self.base.local_blocks,
                    self.local_adapters,
                ),
                start=1,
            ):
                edge_state = edge_update(hidden, edge_index, edge_state)
                adapter_update = adapter(hidden, edge_index, edge_state)
                hidden = block(
                    hidden, edge_index, batch, edge_attr=edge_state
                )
                hidden = hidden + adapter_update
                if layer in MIXER_LAYERS:
                    hidden = self.base.neural_atom_mixers[str(layer)](
                        hidden, batch
                    )
            return self.base._pool(hidden, batch)

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
            )

    return GPSPPDirectionalLocalK1()


def check_mechanism(model, batch):
    """Verify directional train-fixture equations on the remote GPU only."""
    import torch

    if len(model.local_adapters) != LAYERS:
        raise RuntimeError("Directional adapter depth changed")
    expected_receiver = model.mode == MODES[1]
    source, target = batch.edge_index
    if source.numel() == 0 or source.numel() != target.numel():
        raise RuntimeError("Expected directed real bonds")
    if not torch.equal(batch.batch[source], batch.batch[target]):
        raise RuntimeError("Cross-molecule edge detected")

    base = model.base
    with torch.no_grad():
        hidden = base._embed_nodes(batch.x)
        hidden = hidden + base.rwse_encoder(batch.random_walk_pe.float())
        edge_state = base._embed_edges(batch.edge_attr)
        checks = []
        for layer, (edge_update, block, adapter) in enumerate(
            zip(base.edge_updates, base.local_blocks, model.local_adapters),
            start=1,
        ):
            edge_state = edge_update(hidden, batch.edge_index, edge_state)
            normalized, receiver, sender, message = adapter.components(
                hidden, batch.edge_index, edge_state
            )
            update = adapter(hidden, batch.edge_index, edge_state)
            row = {
                "layer": layer,
                "use_receiver": adapter.use_receiver,
                "message_finite": bool(torch.isfinite(message).all()),
                "message_nonzero": bool(torch.count_nonzero(message).item()),
                "sender_nonzero": bool(torch.count_nonzero(sender).item()),
                "receiver_nonzero": bool(torch.count_nonzero(receiver).item()),
                "normalized_node_finite": bool(torch.isfinite(normalized).all()),
                "zero_return_exact": bool(torch.count_nonzero(update).item() == 0),
                "output_projection_zero": bool(
                    torch.count_nonzero(adapter.return_projection.weight).item()
                    == 0
                    and torch.count_nonzero(adapter.return_projection.bias).item()
                    == 0
                ),
            }
            if not expected_receiver and row["receiver_nonzero"]:
                raise RuntimeError("Sender-only adapter used receiver aggregation")
            required = (
                "message_finite",
                "message_nonzero",
                "sender_nonzero",
                "normalized_node_finite",
                "zero_return_exact",
                "output_projection_zero",
            )
            if not all(row[name] for name in required):
                raise RuntimeError(f"Directional adapter invariant failed: {row}")
            if expected_receiver and not row["receiver_nonzero"]:
                raise RuntimeError("Bidirectional receiver aggregation is inactive")
            checks.append(row)
            hidden = block(
                hidden, batch.edge_index, batch.batch, edge_attr=edge_state
            )
            if layer in MIXER_LAYERS:
                hidden = base.neural_atom_mixers[str(layer)](
                    hidden, batch.batch
                )

    adapter_parameters = sum(
        parameter.numel() for parameter in model.local_adapters.parameters()
    )
    if adapter_parameters != ADDED_PARAMETERS:
        raise RuntimeError("Directional adapter parameter identity changed")
    return {
        "equations_verified": True,
        "real_bonds_only": True,
        "directional_endpoint_aggregation": True,
        "receiver_enabled": expected_receiver,
        "adapter_parameters": adapter_parameters,
        "layers": checks,
    }
