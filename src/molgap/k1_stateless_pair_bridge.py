"""A stateless ordered-pair bridge beside the frozen K1 exchanges."""
from __future__ import annotations

import math


MODE = "neural_atom_k1_stateless_pair_bridge"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
PAIR_CHANNELS = 32
EXCHANGE_LAYERS = (3, 6, 9)
ADDED_PARAMETERS = 30_881
PARAMETERS = {MODE: 3_689_698}


class _StatelessPairBridgeFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch_geometric.utils import to_dense_batch

        class StatelessPairBridge(nn.Module):
            def __init__(self):
                super().__init__()
                self.source = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.target = nn.Linear(HIDDEN_CHANNELS, PAIR_CHANNELS)
                self.update_norm = nn.LayerNorm(PAIR_CHANNELS)
                self.selection = nn.Linear(PAIR_CHANNELS, 1)
                self.return_projections = nn.ModuleDict(
                    {
                        str(layer): nn.Linear(
                            PAIR_CHANNELS, HIDDEN_CHANNELS, bias=False
                        )
                        for layer in EXCHANGE_LAYERS
                    }
                )
                for projection in self.return_projections.values():
                    nn.init.zeros_(projection.weight)

            def compute_update(self, hidden, batch, layer):
                if layer not in EXCHANGE_LAYERS:
                    raise ValueError(f"Unsupported bridge layer: {layer}")
                dense, valid = to_dense_batch(hidden, batch)
                source = self.source(dense).unsqueeze(2)
                target = self.target(dense).unsqueeze(1)
                pair_update = self.update_norm(
                    functional.silu(source + target)
                )
                pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
                pair_update = pair_update.masked_fill(
                    ~pair_valid.unsqueeze(-1), 0.0
                )

                logits = self.selection(pair_update).squeeze(-1)
                logits = logits / math.sqrt(PAIR_CHANNELS)
                source_valid = valid.unsqueeze(2).expand_as(logits)
                logits = logits.masked_fill(~source_valid, float("-inf"))
                assignment = torch.softmax(logits, dim=1)
                assignment = assignment * valid.unsqueeze(1).to(
                    assignment.dtype
                )
                target_message = torch.einsum(
                    "bij,bijd->bjd", assignment, pair_update
                )
                dense_update = self.return_projections[str(layer)](
                    target_message
                )
                node_update = dense_update[valid]
                return node_update, {
                    "assignment": assignment,
                    "pair_update": pair_update,
                    "pair_valid": pair_valid,
                    "valid": valid,
                }

        return StatelessPairBridge()


def make_encoder(mode: str):
    if mode != MODE:
        raise ValueError(mode)

    import torch.nn as nn

    from .qm9_neural_atom import make_encoder as make_k1

    class K1StatelessPairBridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.stateless_pair_bridge = _StatelessPairBridgeFactory.make()

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
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
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(hidden, edge_index, edge_state)
                hidden = block(hidden, edge_index, batch, edge_attr=edge_state)
                if str(layer) in self.base.neural_atom_mixers:
                    hidden = self.base.neural_atom_mixers[str(layer)](
                        hidden, batch
                    )
                if layer in EXCHANGE_LAYERS:
                    update, _ = self.stateless_pair_bridge.compute_update(
                        hidden, batch, layer
                    )
                    hidden = hidden + update
            return self.base._pool(hidden, batch)

    return K1StatelessPairBridge()


def check_mechanism(model, batch) -> dict:
    import torch

    probe = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.num_nodes) * HIDDEN_CHANNELS,
        device=batch.batch.device,
    ).reshape(int(batch.num_nodes), HIDDEN_CHANNELS)
    layer_checks = []
    for offset, layer in enumerate(EXCHANGE_LAYERS, start=1):
        layer_probe = probe / offset
        update, diagnostics = model.stateless_pair_bridge.compute_update(
            layer_probe, batch.batch, layer
        )
        assignment = diagnostics["assignment"]
        pair_valid = diagnostics["pair_valid"]
        valid = diagnostics["valid"]
        target_mass = assignment.sum(dim=1)
        expected_mass = valid.to(target_mass.dtype)
        repeated_update, repeated_diagnostics = (
            model.stateless_pair_bridge.compute_update(
                layer_probe, batch.batch, layer
            )
        )
        layer_checks.append(
            {
                "layer": layer,
                "valid_target_mass_one": bool(
                    torch.allclose(
                        target_mass,
                        expected_mass,
                        atol=1e-6,
                        rtol=0,
                    )
                ),
                "padding_assignment_zero": bool(
                    assignment.masked_select(
                        ~valid.unsqueeze(1)
                    ).abs().sum().item()
                    == 0.0
                ),
                "padding_pair_update_zero": bool(
                    diagnostics["pair_update"]
                    .masked_select(~pair_valid.unsqueeze(-1))
                    .abs()
                    .sum()
                    .item()
                    == 0.0
                ),
                "stateless_repeat_exact": bool(
                    torch.equal(update, repeated_update)
                    and torch.equal(
                        diagnostics["pair_update"],
                        repeated_diagnostics["pair_update"],
                    )
                ),
                "zero_initial_update_exact": bool(
                    torch.count_nonzero(update).item() == 0
                ),
            }
        )

    checks = {
        "exchange_layers": list(EXCHANGE_LAYERS),
        "pair_channels": PAIR_CHANNELS,
        "pair_state": "stateless-current-layer-only",
        "pair_update": "silu-source-plus-target-then-layernorm",
        "selection": "current-pair-update-softmax-over-sources",
        "pair_to_node": "normalized-current-update-to-target-node",
        "self_pairs_included": True,
        "return_projections_zero": all(
            torch.count_nonzero(projection.weight).item() == 0
            for projection in model.stateless_pair_bridge.return_projections.values()
        ),
        "layers": layer_checks,
    }
    required = {
        "valid_target_mass_one",
        "padding_assignment_zero",
        "padding_pair_update_zero",
        "stateless_repeat_exact",
        "zero_initial_update_exact",
    }
    if not checks["return_projections_zero"] or not all(
        all(layer_check[field] for field in required)
        for layer_check in layer_checks
    ):
        raise RuntimeError(f"Stateless-pair bridge invariant failed: {checks}")
    return checks
