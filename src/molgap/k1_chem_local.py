"""One-layer, zero-start, color-separated directed-bond update for K1.

This is a bounded adaptation of delayed heterogeneous mixing, not an MMGNN
implementation. The two modes differ only in their fixed 2D color assignment.
"""
from __future__ import annotations


MODES = (
    "neural_atom_k1_atom_pair_local",
    "neural_atom_k1_bond_type_local",
)
TARGET_LAYER = 6
COLORS = 4
HIDDEN = 192
EDGE = 64
ADDED_PARAMETERS = 58_304
PARAMETERS = {mode: 3_717_121 for mode in MODES}


def edge_colors(mode, x, edge_index, edge_attr):
    """Assign every real directed bond to one of four deterministic colors."""
    import torch

    if mode == MODES[1]:
        return edge_attr[:, 0].long().clamp(max=COLORS - 1)
    if mode != MODES[0]:
        raise ValueError(mode)
    source, target = edge_index
    number = x[:, 0].long() + 1  # OGB atomic-number category is Z - 1.
    src, dst = number[source], number[target]
    carbon_src, carbon_dst = src == 6, dst == 6
    hetero_src = (src == 7) | (src == 8) | (src == 16)
    hetero_dst = (dst == 7) | (dst == 8) | (dst == 16)
    colors = torch.full_like(src, 3)
    colors[hetero_src & hetero_dst] = 2
    colors[(carbon_src & hetero_dst) | (hetero_src & carbon_dst)] = 1
    colors[carbon_src & carbon_dst] = 0
    return colors


class _ColoredLocalFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn

        class ColoredLocal(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_norm = nn.LayerNorm(HIDDEN)
                self.edge_norm = nn.LayerNorm(EDGE)
                self.source = nn.Linear(HIDDEN, EDGE, bias=False)
                self.edge = nn.Linear(EDGE, EDGE)
                self.message_norm = nn.LayerNorm(EDGE)
                self.mix = nn.Linear(HIDDEN + COLORS * EDGE, EDGE)
                self.return_projection = nn.Linear(EDGE, HIDDEN)
                nn.init.zeros_(self.return_projection.weight)
                nn.init.zeros_(self.return_projection.bias)

            def components(self, hidden, edge_index, edge_state, colors):
                source, target = edge_index
                normalized = self.node_norm(hidden)
                message = self.message_norm(torch.nn.functional.silu(
                    self.source(normalized[source]) + self.edge(self.edge_norm(edge_state))
                ))
                flat_index = target * COLORS + colors
                sums = message.new_zeros((hidden.shape[0] * COLORS, EDGE))
                sums.index_add_(0, flat_index, message)
                counts = message.new_zeros((hidden.shape[0] * COLORS, 1))
                counts.index_add_(0, flat_index, torch.ones_like(message[:, :1]))
                grouped = (sums / counts.clamp_min(1.0)).reshape(hidden.shape[0], COLORS * EDGE)
                return normalized, grouped, message, counts.reshape(hidden.shape[0], COLORS)

            def forward(self, hidden, edge_index, edge_state, colors):
                normalized, grouped, _, _ = self.components(hidden, edge_index, edge_state, colors)
                return self.return_projection(torch.nn.functional.silu(
                    self.mix(torch.cat((normalized, grouped), dim=-1))
                ))

        return ColoredLocal()


def make_encoder(mode):
    if mode not in MODES:
        raise ValueError(mode)
    import torch.nn as nn
    from .qm9_neural_atom import make_encoder as make_k1

    class ColoredLocalK1(nn.Module):
        def __init__(self):
            super().__init__()
            # Construct K1 first, so seed42 produces its identical shared state.
            self.base = make_k1("neural_atom_k1")
            self.local_adapter = _ColoredLocalFactory.make()
            self.mode = mode

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            if tuple(random_walk_pe.shape) != (x.shape[0], self.base.rwse_dim):
                raise ValueError("RWSE16 graph identity changed")
            hidden = self.base._embed_nodes(x) + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            colors = edge_colors(self.mode, x, edge_index, edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(hidden, edge_index, edge_state)
                hidden = block(hidden, edge_index, batch, edge_attr=edge_state)
                if layer == TARGET_LAYER:
                    hidden = hidden + self.local_adapter(hidden, edge_index, edge_state, colors)
                if str(layer) in self.base.neural_atom_mixers:
                    hidden = self.base.neural_atom_mixers[str(layer)](hidden, batch)
            return self.base._pool(hidden, batch)

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(self.encode(x, edge_index, edge_attr, batch, random_walk_pe))

    return ColoredLocalK1()


def check_mechanism(model, batch):
    """Fail before training if color identity or nested initialization drifts."""
    import torch

    source, target = batch.edge_index
    colors = edge_colors(model.mode, batch.x, batch.edge_index, batch.edge_attr)
    if source.numel() == 0 or colors.shape != source.shape:
        raise RuntimeError("Real directed-bond coverage missing")
    if not torch.equal(batch.batch[source], batch.batch[target]):
        raise RuntimeError("Cross-molecule bond")
    if not bool(((colors >= 0) & (colors < COLORS)).all()):
        raise RuntimeError("Invalid bond color")
    if torch.unique(colors).numel() < 2:
        raise RuntimeError("Color partition collapsed on remote train preflight")
    with torch.no_grad():
        hidden = model.base._embed_nodes(batch.x)
        hidden = hidden + model.base.rwse_encoder(batch.random_walk_pe.float())
        edge_state = model.base._embed_edges(batch.edge_attr)
        normalized, grouped, message, counts = model.local_adapter.components(
            hidden, batch.edge_index, edge_state, colors
        )
        update = model.local_adapter(hidden, batch.edge_index, edge_state, colors)
    if (not bool(torch.isfinite(grouped).all()) or not bool(torch.isfinite(message).all())
            or int(counts.sum().item()) != int(source.numel())
            or int(torch.count_nonzero(update).item()) != 0
            or sum(p.numel() for p in model.local_adapter.parameters()) != ADDED_PARAMETERS):
        raise RuntimeError("Colored local mechanism identity failed")
    return {
        "target_layer": TARGET_LAYER,
        "all_real_directed_bonds_colored": True,
        "cross_molecule_bonds": False,
        "distinct_active_colors": int(torch.unique(colors).numel()),
        "zero_return_projection": True,
        "normalized_hidden_finite": bool(torch.isfinite(normalized).all()),
        "adapter_parameters": ADDED_PARAMETERS,
    }
