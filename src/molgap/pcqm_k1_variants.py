"""Isolated K1 global-allocation variants for the fixed PCQM 100K screen."""
from __future__ import annotations

import math


HIDDEN_CHANNELS = 192
EDGE_STATE_CHANNELS = 64
RELATION_CHANNELS = 32
GATE_HIDDEN_CHANNELS = 32
MIXER_LAYERS = (3, 6, 9)
GATE_RANGE = (0.5, 1.5)

ARCHITECTURE_CONFIGS = {
    "neural_atom_k1_v4": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
    },
    "neural_atom_k1_g": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "per-molecule-per-layer-bounded-global-strength-gate",
        "gate_input": "node-mean-max",
        "gate_hidden_channels": GATE_HIDDEN_CHANNELS,
        "gate_range": list(GATE_RANGE),
    },
    "neural_atom_k1_r": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64-plus-one-relation-slot-32",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "persistent-edge-state-relation-slot",
        "relation_channels": RELATION_CHANNELS,
        "relation_source": "current-directed-real-bond-edge-state",
    },
    "neural_atom_k4_cluster": {
        "backbone": "neural_atom_k1",
        "global_exchange": "four-clustered-atom-slots-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "atom-wise-soft-assignment-across-neural-atoms",
        "active_slots": 4,
        "allocation_heads": 1,
        "allocation_normalization_axis": "neural-atoms-per-original-atom",
        "back_projection": "transpose-of-the-same-allocation",
    },
}


class _MoleculeGateFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        from torch_geometric.nn import global_max_pool, global_mean_pool

        class MoleculeGate(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.LayerNorm(2 * HIDDEN_CHANNELS),
                    nn.Linear(2 * HIDDEN_CHANNELS, GATE_HIDDEN_CHANNELS),
                    nn.SiLU(),
                    nn.Linear(GATE_HIDDEN_CHANNELS, 1),
                )
                nn.init.zeros_(self.network[-1].weight)
                nn.init.zeros_(self.network[-1].bias)

            def forward(self, hidden, batch):
                summary = torch.cat(
                    [
                        global_mean_pool(hidden, batch),
                        global_max_pool(hidden, batch),
                    ],
                    dim=-1,
                )
                # Exact K1 at initialization; bounded attenuation/amplification
                # prevents an unconstrained graph descriptor from replacing it.
                return 1.0 + 0.5 * torch.tanh(self.network(summary))

        return MoleculeGate()


class _RelationSlotFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        from torch_geometric.utils import scatter, softmax

        class RelationSlot(nn.Module):
            def __init__(self):
                super().__init__()
                self.seed = nn.Parameter(torch.empty(RELATION_CHANNELS))
                nn.init.normal_(self.seed, std=RELATION_CHANNELS ** -0.5)
                self.edge_norm = nn.LayerNorm(EDGE_STATE_CHANNELS)
                self.edge_key = nn.Linear(
                    EDGE_STATE_CHANNELS, RELATION_CHANNELS, bias=False
                )
                self.edge_value = nn.Linear(
                    EDGE_STATE_CHANNELS, RELATION_CHANNELS, bias=False
                )
                self.slot_norm1 = nn.LayerNorm(RELATION_CHANNELS)
                self.slot_ffn = nn.Sequential(
                    nn.Linear(RELATION_CHANNELS, 2 * RELATION_CHANNELS),
                    nn.SiLU(),
                    nn.Linear(2 * RELATION_CHANNELS, RELATION_CHANNELS),
                )
                self.slot_norm2 = nn.LayerNorm(RELATION_CHANNELS)
                self.atom_relation_projection = nn.Linear(
                    64 + RELATION_CHANNELS, HIDDEN_CHANNELS, bias=False
                )
                nn.init.zeros_(self.atom_relation_projection.weight)

            def forward(
                self,
                edge_state,
                edge_batch,
                atom_slot,
                atom_assignment,
                valid_nodes,
                graph_count,
            ):
                if edge_state.shape[0] == 0:
                    relation = edge_state.new_zeros(
                        (graph_count, RELATION_CHANNELS)
                    )
                    has_edge = edge_state.new_zeros((graph_count, 1))
                else:
                    normalized = self.edge_norm(edge_state)
                    keys = self.edge_key(normalized)
                    values = self.edge_value(normalized)
                    logits = (keys * self.seed).sum(dim=-1) / math.sqrt(
                        RELATION_CHANNELS
                    )
                    weights = softmax(logits, edge_batch, num_nodes=graph_count)
                    relation = scatter(
                        values * weights.unsqueeze(-1),
                        edge_batch,
                        dim=0,
                        dim_size=graph_count,
                        reduce="sum",
                    )
                    has_edge = scatter(
                        edge_state.new_ones((edge_state.shape[0], 1)),
                        edge_batch,
                        dim=0,
                        dim_size=graph_count,
                        reduce="sum",
                    ).gt(0).to(edge_state.dtype)
                relation = self.slot_norm1(relation + self.seed)
                relation = self.slot_norm2(
                    relation + self.slot_ffn(relation)
                ) * has_edge
                joined = torch.cat([atom_slot.squeeze(1), relation], dim=-1)
                graph_update = self.atom_relation_projection(joined)
                node_weight = atom_assignment[:, 0, :].masked_select(valid_nodes)
                return graph_update.repeat_interleave(
                    valid_nodes.sum(dim=1), dim=0
                ) * node_weight.unsqueeze(-1)

        return RelationSlot()


def _mixer_update_with_slots(mixer, hidden, batch):
    """Evaluate the frozen K1 equation while exposing its one latent slot."""
    import torch
    from torch_geometric.utils import to_dense_batch

    dense, valid = to_dense_batch(mixer.node_norm(hidden), batch)
    keys = mixer.node_key(dense)
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
    return update, slots, assignment, valid


def _cluster_mixer_update(mixer, hidden, batch):
    """Use the paper's atom-wise soft grouping instead of slot-wise pooling.

    The published implementation normalizes every original atom's allocation
    over the neural-atom dimension.  The frozen K1/K4 implementation instead
    normalizes each slot over original atoms.  Keeping the same projections,
    slot exchange, and transposed return isolates that normalization direction.
    """
    import torch
    from torch_geometric.utils import to_dense_batch

    dense, valid = to_dense_batch(mixer.node_norm(hidden), batch)
    keys = mixer.node_key(dense)
    values = mixer.node_value(dense)
    seeds = mixer.slot_seed[: mixer.active_slots]
    queries = mixer.slot_query(seeds)
    logits = torch.einsum("kd,bnd->bkn", queries, keys)
    logits = logits / math.sqrt(mixer.latent_channels)
    assignment = torch.softmax(logits, dim=1)
    assignment = assignment * valid.unsqueeze(1).to(assignment.dtype)
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
    diagnostics = {
        "assignment": assignment,
        "valid": valid,
        "active_slots": mixer.active_slots,
        "allocation_normalization_axis": "slots",
    }
    return update, slots, assignment, valid, diagnostics


def make_encoder(mode: str):
    """Build frozen K1 or one isolated global-allocation candidate."""
    if mode not in ARCHITECTURE_CONFIGS:
        raise ValueError(f"Unknown K1 variant: {mode}")

    from .qm9_neural_atom import make_encoder as make_k1

    if mode == "neural_atom_k1_v4":
        return make_k1("neural_atom_k1")

    import torch.nn as nn

    class K1Variant(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.base = make_k1("neural_atom_k1")
            if mode == "neural_atom_k4_cluster":
                for mixer in self.base.neural_atom_mixers.values():
                    mixer.active_slots = 4
            if mode == "neural_atom_k1_g":
                self.molecule_gates = nn.ModuleDict(
                    {
                        str(layer): _MoleculeGateFactory.make()
                        for layer in MIXER_LAYERS
                    }
                )
            elif mode == "neural_atom_k1_r":
                self.relation_slots = nn.ModuleDict(
                    {
                        str(layer): _RelationSlotFactory.make()
                        for layer in MIXER_LAYERS
                    }
                )

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
                mixer = self.base.neural_atom_mixers[str(layer)]
                if self.mode == "neural_atom_k1_g":
                    update, _ = mixer.compute_update(h, batch)
                    scale = self.molecule_gates[str(layer)](h, batch)
                    h = h + scale[batch] * update
                elif self.mode == "neural_atom_k1_r":
                    update, slots, assignment, valid = _mixer_update_with_slots(
                        mixer, h, batch
                    )
                    edge_batch = batch[edge_index[0]]
                    graph_count = int(valid.shape[0])
                    relation_update = self.relation_slots[str(layer)](
                        edge_state,
                        edge_batch,
                        slots,
                        assignment,
                        valid,
                        graph_count,
                    )
                    h = h + update + relation_update
                else:
                    update, _, _, _, _ = _cluster_mixer_update(
                        mixer, h, batch
                    )
                    h = h + update
            return self.base._pool(h, batch)

    return K1Variant()
