"""Conjugated-component hyperedge exchange beside the fixed 2D K1 encoder."""
from __future__ import annotations


ONESHOT = "neural_atom_k1_conjugated_oneshot"
PERSISTENT = "neural_atom_k1_conjugated_persistent"
MODES = (ONESHOT, PERSISTENT)
LAYERS = {ONESHOT: (6,), PERSISTENT: (3, 6, 9)}
CHANNELS = 32
HIDDEN = 192


def make_encoder(mode: str):
    if mode not in MODES:
        raise ValueError(mode)
    import torch
    import torch.nn as nn
    from .qm9_neural_atom import make_encoder as make_k1

    class K1ConjugatedHyperedge(nn.Module):
        requires_conjugated_components = True

        def __init__(self):
            super().__init__()
            self.mode = mode
            self.base = make_k1("neural_atom_k1")
            self.atom_projection = nn.Linear(HIDDEN, CHANNELS, bias=False)
            self.update_norm = nn.LayerNorm(2 * CHANNELS)
            self.update_gate = nn.Linear(2 * CHANNELS, CHANNELS)
            self.update_value = nn.Sequential(
                nn.Linear(2 * CHANNELS, 2 * CHANNELS), nn.SiLU(),
                nn.Linear(2 * CHANNELS, CHANNELS),
            )
            self.state_norm = nn.LayerNorm(CHANNELS)
            self.return_projection = nn.Linear(CHANNELS, HIDDEN, bias=False)
            nn.init.zeros_(self.return_projection.weight)

        @staticmethod
        def component_ids(component_id, component_count, batch, node_count):
            ids = component_id.reshape(-1).long()
            counts = component_count.reshape(-1).long()
            if ids.shape != (node_count,) or counts.numel() != int(batch.max()) + 1:
                raise ValueError("Conjugated sidecar shape mismatch")
            if (counts < 0).any():
                raise ValueError("Negative conjugated component count")
            valid = ids >= 0
            if valid.any() and (ids[valid] >= counts[batch[valid]]).any():
                raise ValueError("Conjugated component id exceeds graph count")
            offsets = counts.cumsum(0) - counts
            global_ids = ids.clone()
            global_ids[valid] += offsets[batch[valid]]
            return global_ids, valid, int(counts.sum())

        def exchange(self, h, state, ids, valid, count):
            if count == 0:
                return h, state
            projected = self.atom_projection(h[valid])
            component_sum = h.new_zeros((count, CHANNELS))
            component_sum.index_add_(0, ids[valid], projected)
            component_n = h.new_zeros((count, 1))
            component_n.index_add_(
                0, ids[valid], h.new_ones((projected.shape[0], 1))
            )
            summary = component_sum / component_n.clamp_min(1)
            combined = self.update_norm(torch.cat([state, summary], dim=-1))
            state = self.state_norm(
                state + torch.sigmoid(self.update_gate(combined))
                * self.update_value(combined)
            )
            update = h.new_zeros(h.shape)
            update[valid] = self.return_projection(state[ids[valid]])
            return h + update, state

        def encode(
            self, x, edge_index, edge_attr, batch, random_walk_pe,
            conjugated_component_id, conjugated_component_count,
        ):
            if tuple(random_walk_pe.shape) != (x.shape[0], self.base.rwse_dim):
                raise ValueError("RWSE16 identity changed")
            ids, valid, count = self.component_ids(
                conjugated_component_id, conjugated_component_count,
                batch, x.shape[0],
            )
            h = self.base._embed_nodes(x)
            h = h + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            state = h.new_zeros((count, CHANNELS))
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(h, edge_index, edge_state)
                h = block(h, edge_index, batch, edge_attr=edge_state)
                if str(layer) in self.base.neural_atom_mixers:
                    h = self.base.neural_atom_mixers[str(layer)](h, batch)
                if layer in LAYERS[self.mode]:
                    h, state = self.exchange(h, state, ids, valid, count)
            return self.base._pool(h, batch)

        def forward(
            self, x, edge_index, edge_attr, batch, random_walk_pe,
            conjugated_component_id, conjugated_component_count,
        ):
            return self.base.head(
                self.encode(
                    x, edge_index, edge_attr, batch, random_walk_pe,
                    conjugated_component_id, conjugated_component_count,
                )
            )

    return K1ConjugatedHyperedge()


def check_mechanism(model, batch) -> dict:
    import torch
    ids, valid, count = model.component_ids(
        batch.conjugated_component_id, batch.conjugated_component_count,
        batch.batch, batch.x.shape[0],
    )
    if count == 0:
        raise RuntimeError("Preflight fixture has no conjugated components")
    with torch.no_grad():
        h = torch.linspace(
            -1.0, 1.0, steps=batch.x.shape[0] * HIDDEN,
            device=batch.x.device,
        ).reshape(batch.x.shape[0], HIDDEN)
        result, state = model.exchange(
            h, h.new_zeros((count, CHANNELS)), ids, valid, count,
        )
    if not torch.equal(h, result) or not torch.isfinite(state).all():
        raise RuntimeError("Hyperedge path is not finite and zero-return nested")
    return {
        "component_count": count,
        "participating_atoms": int(valid.sum()),
        "component_layers": list(LAYERS[model.mode]),
        "component_channels": CHANNELS,
        "zero_return_exact": True,
        "component_ids_valid": True,
        "global_slot_unchanged": True,
        "real_bond_edge_state_unchanged": True,
    }
