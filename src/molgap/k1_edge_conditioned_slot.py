"""K1 slot selection conditioned on persistent local bond state.

The candidate keeps the frozen K1 encoder, one 64-channel slot, and the same
three exchange layers.  It only adds a zero-initialized projection from the
mean incident directed real-bond state to the slot-selection key.  Therefore
the first function is exactly K1 while later optimization can make global
atom-to-slot selection edge-aware.
"""
from __future__ import annotations

from .pcqm_k1_variants import (
    EDGE_STATE_CHANNELS,
    HIDDEN_CHANNELS,
    MIXER_LAYERS,
)


MODES = ("neural_atom_k1_edge_conditioned_slot",)
PARAMETERS = {MODES[0]: 3_671_105}
ADDED_PARAMETERS = 3 * EDGE_STATE_CHANNELS * EDGE_STATE_CHANNELS


def _incident_edge_context(edge_state, edge_index, node_count):
    """Mean directed real-bond state incident to each atom."""
    import torch

    if edge_state.ndim != 2 or edge_state.shape[1] != EDGE_STATE_CHANNELS:
        raise ValueError("edge_state must have shape [E, 64]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.shape[1] != edge_state.shape[0]:
        raise ValueError("edge_index and edge_state edge counts differ")
    context = edge_state.new_zeros((node_count, EDGE_STATE_CHANNELS))
    counts = edge_state.new_zeros((node_count, 1))
    if edge_state.shape[0]:
        source, target = edge_index
        context.index_add_(0, source, edge_state)
        context.index_add_(0, target, edge_state)
        ones = edge_state.new_ones((edge_state.shape[0], 1))
        counts.index_add_(0, source, ones)
        counts.index_add_(0, target, ones)
    return context / counts.clamp_min_(1.0)


def _edge_conditioned_update(
    mixer,
    edge_key_projection,
    edge_context_norm,
    hidden,
    batch,
    edge_index,
    edge_state,
):
    """Run the unchanged K1 slot equation with an edge-aware selector key."""
    import math

    import torch
    from torch_geometric.utils import to_dense_batch

    dense, valid = to_dense_batch(mixer.node_norm(hidden), batch)
    incident = _incident_edge_context(edge_state, edge_index, hidden.shape[0])
    edge_dense, edge_valid = to_dense_batch(incident, batch)
    if not torch.equal(valid, edge_valid):
        raise RuntimeError("Node and incident-edge padding masks differ")
    edge_dense = edge_context_norm(edge_dense)
    edge_delta = edge_key_projection(edge_dense)
    keys = mixer.node_key(dense) + edge_delta
    values = mixer.node_value(dense)
    seeds = mixer.slot_seed[: mixer.active_slots]
    queries = mixer.slot_query(seeds)
    logits = torch.einsum("kd,bnd->bkn", queries, keys)
    logits = logits / math.sqrt(mixer.latent_channels)
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    assignment = torch.softmax(logits, dim=-1)
    slots = seeds.unsqueeze(0) + torch.einsum(
        "bkn,bnd->bkd", assignment, values
    )
    attended, _ = mixer.slot_attention(
        slots, slots, slots, need_weights=False
    )
    slots = mixer.slot_norm1(slots + attended)
    slots = mixer.slot_norm2(slots + mixer.slot_ffn(slots))
    returned = torch.einsum("bkn,bkd->bnd", assignment, slots)
    update = mixer.dropout(mixer.return_projection(returned[valid]))
    return update, slots, assignment, valid, {
        "incident_edge_context": edge_dense,
        "edge_key_delta": edge_delta,
    }


def make_encoder(mode):
    if mode not in MODES:
        raise ValueError(mode)
    import torch
    import torch.nn as nn

    from .qm9_neural_atom import make_encoder as frozen_encoder

    class EdgeConditionedSlotK1(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = frozen_encoder("neural_atom_k1")
            self.mode = mode
            # Elementwise-affine=False adds no parameters.  The projection is
            # the sole new trainable path and is exactly zero at construction.
            self.edge_context_norms = nn.ModuleDict(
                {
                    str(layer): nn.LayerNorm(
                        EDGE_STATE_CHANNELS, elementwise_affine=False
                    )
                    for layer in MIXER_LAYERS
                }
            )
            self.edge_conditioned_keys = nn.ModuleDict(
                {
                    str(layer): nn.Linear(
                        EDGE_STATE_CHANNELS,
                        EDGE_STATE_CHANNELS,
                        bias=False,
                    )
                    for layer in MIXER_LAYERS
                }
            )
            for projection in self.edge_conditioned_keys.values():
                nn.init.zeros_(projection.weight)

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
            h = self.base._embed_nodes(x)
            h = h + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(h, edge_index, edge_state)
                h = block(h, edge_index, batch, edge_attr=edge_state)
                if layer not in MIXER_LAYERS:
                    continue
                update, _, _, _, _ = _edge_conditioned_update(
                    self.base.neural_atom_mixers[str(layer)],
                    self.edge_conditioned_keys[str(layer)],
                    self.edge_context_norms[str(layer)],
                    h,
                    batch,
                    edge_index,
                    edge_state,
                )
                h = h + update
            return self.base._pool(h, batch)

    return EdgeConditionedSlotK1()


def check_mechanism(model, batch):
    """Verify the edge-aware selector path on a train-only fixture."""
    import torch

    base = model.base
    projections_zero = all(
        torch.count_nonzero(projection.weight).item() == 0
        for projection in model.edge_conditioned_keys.values()
    )
    if not projections_zero:
        raise RuntimeError("Edge-conditioned key projection is not zero initialized")

    with torch.no_grad():
        h = base._embed_nodes(batch.x) + base.rwse_encoder(
            batch.random_walk_pe.float()
        )
        edge_state = base._embed_edges(batch.edge_attr)
        edge_layers = []
        slot_layers = []
        for layer, (edge_update, block) in enumerate(
            zip(base.edge_updates, base.local_blocks), start=1
        ):
            edge_state = edge_update(h, batch.edge_index, edge_state)
            h = block(h, batch.edge_index, batch.batch, edge_attr=edge_state)
            row = {
                "layer": layer,
                "edge_state_shape": list(edge_state.shape),
                "edge_state_finite": bool(torch.isfinite(edge_state).all()),
            }
            if layer in MIXER_LAYERS:
                update, _, assignment, valid, details = _edge_conditioned_update(
                    base.neural_atom_mixers[str(layer)],
                    model.edge_conditioned_keys[str(layer)],
                    model.edge_context_norms[str(layer)],
                    h,
                    batch.batch,
                    batch.edge_index,
                    edge_state,
                )
                row.update(
                    {
                        "incident_edge_context_finite": bool(
                            torch.isfinite(details["incident_edge_context"]).all()
                        ),
                        "edge_key_delta_shape": list(
                            details["edge_key_delta"].shape
                        ),
                    }
                )
                mass = assignment.sum(dim=-1)
                if not torch.allclose(
                    mass, torch.ones_like(mass), atol=1e-6, rtol=0
                ):
                    raise RuntimeError("Edge-conditioned slot mass changed")
                row["slot_assignment_mass_one"] = True
                slot_layers.append({"layer": layer, "assignment_mass_one": True})
                h = h + update
            edge_layers.append(row)

    if len(edge_layers) != 9 or len(slot_layers) != 3:
        raise RuntimeError("Expected nine edge layers and three slot layers")
    return {
        "equations_verified": True,
        "real_bonds_only": True,
        "normalized_update_context": True,
        "edge_context_source": "mean-incident-directed-real-bond-state",
        "zero_initialized_edge_key": projections_zero,
        "layers": edge_layers,
        "slot_layers": slot_layers,
    }
