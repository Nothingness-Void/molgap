"""Isolated K1 global-allocation variants for the fixed PCQM 100K screen."""
from __future__ import annotations

import copy
import math


HIDDEN_CHANNELS = 192
EDGE_STATE_CHANNELS = 64
RELATION_CHANNELS = 32
GATE_HIDDEN_CHANNELS = 32
MIXER_LAYERS = (3, 6, 9)
GATE_RANGE = (0.5, 1.5)
REPSET_HIDDEN_SETS = 8
REPSET_ELEMENTS = 8
REPSET_CHANNELS = 64

ARCHITECTURE_CONFIGS = {
    "neural_atom_k1_mose": {
        "backbone": "neural_atom_k1_v4",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "replace-rwse16-with-rooted-mose31",
        "structural_encoding": "all-connected-2-through-5-node-homomorphisms-plus-c6",
        "structural_channels": 31,
        "input_transform": "log1p",
        "expected_parameters": 3_661_697,
        "geometry": False,
        "teacher": False,
    },
    "neural_atom_k1_mose_hidden_bn": {
        "backbone": "neural_atom_k1_v4",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "replace-rwse16-with-rooted-mose31-and-hidden-batchnorm",
        "structural_encoding": "all-connected-2-through-5-node-homomorphisms-plus-c6",
        "structural_channels": 31,
        "input_transform": "log1p",
        "structural_mlp": "linear192-batchnorm192-silu-linear192",
        "raw_input_batchnorm": False,
        "expected_parameters": 3_662_081,
        "geometry": False,
        "teacher": False,
    },
    "neural_atom_k1_pair_token": {
        "backbone": "neural_atom_k1_v4",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "one-layer6-pre-normalized-all-pair-relation-token",
        "target_layer": 6,
        "pair_channels": 32,
        "relation_tokens": 1,
        "dense_atom_to_atom_attention": False,
        "added_parameters": 22_848,
        "initialization_policy": "nested-function",
        "geometry": False,
    },
    "neural_atom_k1_gpspp_sender": {
        "backbone": "neural_atom_k1",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "zero-initialized-edge-conditioned-sender-return",
        "local_adapter_layers": 9,
        "directional_aggregation": "sender-only",
        "added_parameters": 857088,
    },
    "neural_atom_k1_gpspp_bidirectional": {
        "backbone": "neural_atom_k1",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "zero-initialized-edge-conditioned-bidirectional-return",
        "local_adapter_layers": 9,
        "directional_aggregation": "receiver-and-sender-separate",
        "added_parameters": 857088,
    },
    "neural_atom_k1_edge_context_no_slot_attention": {
        "backbone": "neural_atom_k1",
        "exchange_layers": list(MIXER_LAYERS),
        "changes": ["raw-real-bond-storage-normalized-context-read", "remove-length-one-slot-self-attention"],
        "added_parameters": 0,
    },
    "neural_atom_k1_edge_context_uniform_return": {
        "backbone": "neural_atom_k1",
        "exchange_layers": list(MIXER_LAYERS),
        "changes": ["raw-real-bond-storage-normalized-context-read", "uniform-slot-return"],
        "added_parameters": 0,
    },
    "neural_atom_k1_edge_read_norm": {
        "backbone": "neural_atom_k1", "exchange_layers": list(MIXER_LAYERS),
        "change": "real-bond-raw-residual-storage-normalized-node-read",
        "normalized_update_context": False, "added_parameters": 0,
    },
    "neural_atom_k1_edge_context_read_norm": {
        "backbone": "neural_atom_k1", "exchange_layers": list(MIXER_LAYERS),
        "change": "real-bond-raw-residual-storage-normalized-context-and-node-read",
        "normalized_update_context": True, "added_parameters": 0,
    },
    "neural_atom_k1_v4": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
    },
    "neural_atom_k1_edge_conditioned_slot": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "incident-real-bond-state-conditioned-slot-key",
        "edge_context": "mean-incident-directed-real-bond-state",
        "edge_context_channels": EDGE_STATE_CHANNELS,
        "edge_key_projection": "zero-initialized-64-to-64",
        "added_parameters": 12_288,
        "initialization_policy": "identical-tensors-altered-edge-dataflow",
        "geometry": False,
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
    "neural_atom_k1_h4": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "four-head-atom-selection-within-one-latent-token",
        "active_slots": 1,
        "allocation_heads": 4,
        "head_channels": 16,
        "allocation_normalization_axis": "original-atoms-per-head",
        "back_projection": "head-wise-transpose-of-the-same-allocation",
    },
    "neural_atom_k1_dynamic_query": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "molecule-conditioned-single-allocation-query",
        "active_slots": 1,
        "allocation_heads": 1,
        "query_context": "mean-current-node-state",
        "context_projection": "zero-initialized-linear-192-to-64",
        "allocation_normalization_axis": "original-atoms",
        "back_projection": "transpose-of-the-same-allocation",
    },
    "neural_atom_k1_repset_readout": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "repset-final-node-set-readout-residual",
        "readout": "mean-plus-zero-initialized-repset",
        "repset_hidden_sets": REPSET_HIDDEN_SETS,
        "repset_elements_per_hidden_set": REPSET_ELEMENTS,
        "repset_output_channels": REPSET_CHANNELS,
        "target_residual": False,
    },
    "neural_atom_k1_tied_selector": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "shared-atom-selector-across-exchange-layers",
        "shared_selector": ["node_norm", "node_key", "direct_query"],
        "independent_per_layer": [
            "node_value",
            "slot_seed",
            "slot_attention",
            "slot_ffn",
            "return_projection",
        ],
    },
    "neural_atom_k1_collapsed_mha": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "collapse-length-one-slot-self-attention",
        "removed": ["slot-query-projection", "slot-key-projection"],
        "retained": ["slot-value-projection", "slot-output-projection"],
        "attention_sequence_length": 1,
    },
    "neural_atom_k1_no_slot_attention": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "remove-length-one-slot-self-attention",
        "removed": ["slot-self-attention"],
        "retained": ["atom-selection", "slot-ffn", "slot-return"],
        "attention_sequence_length": 1,
    },
    "neural_atom_k1_uniform_return": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "decouple-learned-source-from-uniform-recipient-allocation",
        "source_allocation": "learned-softmax-over-atoms",
        "return_allocation": "uniform-over-valid-atoms",
        "return_mass_per_graph": 1.0,
        "added_parameters": 0,
    },
    "neural_atom_k1_inverse_return": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "decouple-learned-source-from-complementary-recipient-allocation",
        "source_allocation": "learned-softmax-over-atoms",
        "return_allocation": "softmax-of-negative-source-logits",
        "return_mass_per_graph": 1.0,
        "added_parameters": 0,
    },
    "neural_atom_k1_no_attention_uniform_return": {
        "backbone": "neural_atom_k1",
        "global_exchange": "one-atom-slot-64",
        "exchange_layers": list(MIXER_LAYERS),
        "change": "combine-two-directional-k1-simplifications",
        "source_allocation": "learned-softmax-over-atoms",
        "slot_processor": "no-self-attention-plus-slot-ffn",
        "return_allocation": "uniform-over-valid-atoms",
        "return_mass_per_graph": 1.0,
        "removed": ["slot-self-attention"],
    },
}


class _RepSetReadoutFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        from torch_geometric.utils import to_dense_batch

        class RepSetReadout(nn.Module):
            """Published RepSet equation with a nested representation return.

            The set transform follows the authors' open implementation.  Its
            final map returns a graph representation correction and is zero
            initialized, so this remains a single-encoder readout rather than
            a target-space residual model.
            """

            def __init__(self):
                super().__init__()
                self.n_hidden_sets = REPSET_HIDDEN_SETS
                self.n_elements = REPSET_ELEMENTS
                self.prototype = nn.Parameter(
                    torch.empty(
                        HIDDEN_CHANNELS,
                        self.n_hidden_sets * self.n_elements,
                    )
                )
                nn.init.normal_(self.prototype)
                self.hidden_set_norm = nn.BatchNorm1d(self.n_hidden_sets)
                self.hidden_set_projection = nn.Linear(
                    self.n_hidden_sets,
                    REPSET_CHANNELS,
                )
                self.activation = nn.LeakyReLU()
                self.return_projection = nn.Linear(
                    REPSET_CHANNELS,
                    HIDDEN_CHANNELS,
                    bias=False,
                )
                nn.init.zeros_(self.return_projection.weight)

            def forward(self, hidden, batch):
                dense, valid = to_dense_batch(hidden, batch)
                scores = self.activation(dense @ self.prototype)
                scores = scores.reshape(
                    dense.shape[0],
                    dense.shape[1],
                    self.n_elements,
                    self.n_hidden_sets,
                )
                scores = scores.max(dim=2).values
                scores = scores * valid.unsqueeze(-1).to(scores.dtype)
                set_embedding = scores.sum(dim=1)
                set_embedding = self.hidden_set_norm(set_embedding)
                set_embedding = self.activation(
                    self.hidden_set_projection(set_embedding)
                )
                return self.return_projection(set_embedding)

        return RepSetReadout()


class _SharedSelectorFactory:
    @staticmethod
    def make(template):
        import torch.nn as nn

        class SharedSelector(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_norm = copy.deepcopy(template.node_norm)
                self.node_key = copy.deepcopy(template.node_key)
                query = template.slot_query(template.slot_seed[:1]).detach()
                self.query = nn.Parameter(query.squeeze(0).clone())

        return SharedSelector()


class _CollapsedSlotProjectionFactory:
    @staticmethod
    def make(attention):
        import torch
        import torch.nn as nn

        class CollapsedSlotProjection(nn.Module):
            """The active V and output maps of length-one self-attention."""

            def __init__(self):
                super().__init__()
                channels = int(attention.embed_dim)
                self.value = nn.Linear(channels, channels, bias=True)
                self.output = nn.Linear(channels, channels, bias=True)
                with torch.no_grad():
                    self.value.weight.copy_(attention.in_proj_weight[2 * channels :])
                    self.value.bias.copy_(attention.in_proj_bias[2 * channels :])
                    self.output.weight.copy_(attention.out_proj.weight)
                    self.output.bias.copy_(attention.out_proj.bias)

            def forward(self, slots):
                return self.output(self.value(slots))

        return CollapsedSlotProjection()


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


def _multihead_single_slot_update(mixer, hidden, batch):
    """Let four channel heads select atoms while retaining one latent token.

    K1 uses one 64-channel query and one atom distribution.  This equation
    reshapes the existing projections into four 16-channel heads, normalizes
    each head over original atoms, concatenates the four pooled views into one
    64-channel token, and reuses each head's assignment for return.  It adds no
    parameters, slots, exchange layers, or optimization steps.
    """
    import torch
    from torch_geometric.utils import to_dense_batch

    allocation_heads = 4
    head_channels = mixer.latent_channels // allocation_heads
    if head_channels * allocation_heads != mixer.latent_channels:
        raise ValueError("latent channels must divide allocation heads")
    if mixer.active_slots != 1:
        raise ValueError("multi-head single-slot allocation requires one slot")

    dense, valid = to_dense_batch(mixer.node_norm(hidden), batch)
    batch_size, max_nodes, _ = dense.shape
    keys = mixer.node_key(dense).reshape(
        batch_size, max_nodes, allocation_heads, head_channels
    )
    values = mixer.node_value(dense).reshape(
        batch_size, max_nodes, allocation_heads, head_channels
    )
    seeds = mixer.slot_seed[:1]
    queries = mixer.slot_query(seeds).reshape(
        1, allocation_heads, head_channels
    )
    logits = torch.einsum("khd,bnhd->bhkn", queries, keys)
    logits = logits / math.sqrt(head_channels)
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    assignment = torch.softmax(logits, dim=-1)
    pooled = torch.einsum("bhkn,bnhd->bkhd", assignment, values).reshape(
        batch_size, 1, mixer.latent_channels
    )
    slots = seeds.unsqueeze(0) + pooled
    attended, _ = mixer.slot_attention(slots, slots, slots, need_weights=False)
    slots = mixer.slot_norm1(slots + attended)
    slots = mixer.slot_norm2(slots + mixer.slot_ffn(slots))
    slot_heads = slots.reshape(
        batch_size, 1, allocation_heads, head_channels
    )
    returned = torch.einsum(
        "bhkn,bkhd->bnhd", assignment, slot_heads
    ).reshape(batch_size, max_nodes, mixer.latent_channels)
    update = mixer.dropout(mixer.return_projection(returned[valid]))
    diagnostics = {
        "assignment": assignment,
        "valid": valid,
        "active_slots": 1,
        "allocation_heads": allocation_heads,
        "head_channels": head_channels,
        "allocation_normalization_axis": "atoms-per-head",
    }
    return update, slots, assignment, valid, diagnostics


def _dynamic_query_single_slot_update(mixer, conditioner, hidden, batch):
    """Condition K1's sole atom distribution on the current molecule.

    The mean current node state changes only the single slot query.  Pooling,
    slot processing, transposed return, and the local path remain K1-identical.
    A zero-initialized context projection nests the initial graph function
    exactly inside the frozen reference.
    """
    import torch
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.utils import to_dense_batch

    if mixer.active_slots != 1:
        raise ValueError("dynamic K1 query requires exactly one slot")
    normalized_hidden = mixer.node_norm(hidden)
    dense, valid = to_dense_batch(normalized_hidden, batch)
    keys = mixer.node_key(dense)
    values = mixer.node_value(dense)
    seeds = mixer.slot_seed[:1]
    fixed_query = mixer.slot_query(seeds).unsqueeze(0)
    graph_context = global_mean_pool(normalized_hidden, batch)
    query = fixed_query + conditioner(graph_context).unsqueeze(1)
    logits = torch.einsum("bkd,bnd->bkn", query, keys)
    logits = logits / math.sqrt(mixer.latent_channels)
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    assignment = torch.softmax(logits, dim=-1)
    slots = seeds.unsqueeze(0) + torch.einsum(
        "bkn,bnd->bkd", assignment, values
    )
    attended, _ = mixer.slot_attention(slots, slots, slots, need_weights=False)
    slots = mixer.slot_norm1(slots + attended)
    slots = mixer.slot_norm2(slots + mixer.slot_ffn(slots))
    returned = torch.einsum("bkn,bkd->bnd", assignment, slots)
    update = mixer.dropout(mixer.return_projection(returned[valid]))
    diagnostics = {
        "assignment": assignment,
        "valid": valid,
        "active_slots": 1,
        "allocation_heads": 1,
        "query_context": "mean-current-node-state",
        "allocation_normalization_axis": "atoms",
    }
    return update, slots, assignment, valid, diagnostics


def _tied_selector_single_slot_update(mixer, selector, hidden, batch):
    """Reuse one atom selector while retaining layer-specific slot content."""
    import torch
    from torch_geometric.utils import to_dense_batch

    if mixer.active_slots != 1:
        raise ValueError("tied K1 selector requires exactly one slot")
    selector_dense, valid = to_dense_batch(selector.node_norm(hidden), batch)
    value_dense, value_valid = to_dense_batch(mixer.node_norm(hidden), batch)
    if not torch.equal(valid, value_valid):
        raise RuntimeError("Selector and value layouts differ")
    keys = selector.node_key(selector_dense)
    values = mixer.node_value(value_dense)
    logits = torch.einsum("d,bnd->bn", selector.query, keys)
    logits = logits / math.sqrt(mixer.latent_channels)
    logits = logits.masked_fill(~valid, float("-inf"))
    assignment = torch.softmax(logits, dim=-1).unsqueeze(1)
    seeds = mixer.slot_seed[:1]
    slots = seeds.unsqueeze(0) + torch.einsum(
        "bkn,bnd->bkd", assignment, values
    )
    attended, _ = mixer.slot_attention(slots, slots, slots, need_weights=False)
    slots = mixer.slot_norm1(slots + attended)
    slots = mixer.slot_norm2(slots + mixer.slot_ffn(slots))
    returned = torch.einsum("bkn,bkd->bnd", assignment, slots)
    update = mixer.dropout(mixer.return_projection(returned[valid]))
    diagnostics = {
        "assignment": assignment,
        "valid": valid,
        "active_slots": 1,
        "selector_scope": "shared-across-layers-3-6-9",
        "value_scope": "independent-per-layer",
    }
    return update, slots, assignment, valid, diagnostics


def _single_slot_processor_update(mixer, processor, hidden, batch):
    """Run K1 pooling/return with a chosen length-one slot processor."""
    import torch
    from torch_geometric.utils import to_dense_batch

    if mixer.active_slots != 1:
        raise ValueError("single-slot processor variants require one slot")
    dense, valid = to_dense_batch(mixer.node_norm(hidden), batch)
    keys = mixer.node_key(dense)
    values = mixer.node_value(dense)
    seeds = mixer.slot_seed[:1]
    queries = mixer.slot_query(seeds)
    logits = torch.einsum("kd,bnd->bkn", queries, keys)
    logits = logits / math.sqrt(mixer.latent_channels)
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    assignment = torch.softmax(logits, dim=-1)
    slots = seeds.unsqueeze(0) + torch.einsum(
        "bkn,bnd->bkd", assignment, values
    )
    if processor is None:
        slots = mixer.slot_norm1(slots)
        processor_name = "none"
    else:
        slots = mixer.slot_norm1(slots + processor(slots))
        processor_name = "collapsed-value-output"
    slots = mixer.slot_norm2(slots + mixer.slot_ffn(slots))
    returned = torch.einsum("bkn,bkd->bnd", assignment, slots)
    update = mixer.dropout(mixer.return_projection(returned[valid]))
    diagnostics = {
        "assignment": assignment,
        "valid": valid,
        "active_slots": 1,
        "slot_processor": processor_name,
        "slot_attention_module_removed": not hasattr(mixer, "slot_attention"),
    }
    return update, slots, assignment, valid, diagnostics


def _return_allocation_update(
    mixer, hidden, batch, return_mode, *, remove_slot_attention=False
):
    """Separate atoms that build K1's slot from atoms that receive it.

    The frozen K1 equation reuses one learned atom distribution in both
    directions.  This variant keeps its source pooling and slot processor
    unchanged, but gives the normalized return path either uniform recipients
    or the complementary distribution implied by the negative source logits.
    Both choices preserve unit return mass per graph and add no parameters.
    """
    import torch
    from torch_geometric.utils import to_dense_batch

    if mixer.active_slots != 1:
        raise ValueError("return-allocation variants require one slot")
    if return_mode not in {"uniform", "inverse-score"}:
        raise ValueError(f"Unknown return allocation: {return_mode}")

    dense, valid = to_dense_batch(mixer.node_norm(hidden), batch)
    keys = mixer.node_key(dense)
    values = mixer.node_value(dense)
    seeds = mixer.slot_seed[:1]
    queries = mixer.slot_query(seeds)
    logits = torch.einsum("kd,bnd->bkn", queries, keys)
    logits = logits / math.sqrt(mixer.latent_channels)
    masked_logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    source_assignment = torch.softmax(masked_logits, dim=-1)
    slots = seeds.unsqueeze(0) + torch.einsum(
        "bkn,bnd->bkd", source_assignment, values
    )
    if remove_slot_attention:
        if hasattr(mixer, "slot_attention"):
            raise RuntimeError("Combined simplification retained slot attention")
        slots = mixer.slot_norm1(slots)
        slot_processor = "none"
    else:
        attended, _ = mixer.slot_attention(slots, slots, slots, need_weights=False)
        slots = mixer.slot_norm1(slots + attended)
        slot_processor = "length-one-self-attention"
    slots = mixer.slot_norm2(slots + mixer.slot_ffn(slots))

    if return_mode == "uniform":
        return_assignment = valid.unsqueeze(1).to(logits.dtype)
        return_assignment = return_assignment / return_assignment.sum(
            dim=-1, keepdim=True
        )
    else:
        inverse_logits = (-logits).masked_fill(
            ~valid.unsqueeze(1), float("-inf")
        )
        return_assignment = torch.softmax(inverse_logits, dim=-1)

    returned = torch.einsum("bkn,bkd->bnd", return_assignment, slots)
    update = mixer.dropout(mixer.return_projection(returned[valid]))
    diagnostics = {
        "source_assignment": source_assignment,
        "return_assignment": return_assignment,
        "valid": valid,
        "active_slots": 1,
        "source_allocation": "learned-softmax-over-atoms",
        "return_allocation": return_mode,
        "return_mass_per_graph": 1.0,
        "slot_processor": slot_processor,
        "slot_attention_module_removed": not hasattr(mixer, "slot_attention"),
    }
    return update, slots, source_assignment, return_assignment, valid, diagnostics


def make_encoder(mode: str):
    """Build frozen K1 or one isolated global-allocation candidate."""
    if mode not in ARCHITECTURE_CONFIGS:
        raise ValueError(f"Unknown K1 variant: {mode}")

    from .k1_edge_memory import MODES as EDGE_MEMORY_MODES
    if mode in EDGE_MEMORY_MODES:
        from .k1_edge_memory import make_encoder as make_edge_memory
        return make_edge_memory(mode)
    from .k1_edge_slot_interaction import MODES as EDGE_SLOT_MODES
    if mode in EDGE_SLOT_MODES:
        from .k1_edge_slot_interaction import make_encoder as make_edge_slot
        return make_edge_slot(mode)
    from .k1_edge_conditioned_slot import MODES as EDGE_CONDITIONED_MODES
    if mode in EDGE_CONDITIONED_MODES:
        from .k1_edge_conditioned_slot import make_encoder as make_edge_conditioned
        return make_edge_conditioned(mode)
    from .k1_gpspp_local import MODES as GPSPP_LOCAL_MODES
    if mode in GPSPP_LOCAL_MODES:
        from .k1_gpspp_local import make_encoder as make_gpspp_local
        return make_gpspp_local(mode)
    from .k1_pair_token import MODES as PAIR_TOKEN_MODES
    if mode in PAIR_TOKEN_MODES:
        from .k1_pair_token import make_encoder as make_pair_token
        return make_pair_token(mode)

    from .qm9_neural_atom import make_encoder as make_k1

    if mode == "neural_atom_k1_v4":
        return make_k1("neural_atom_k1")

    if mode in {"neural_atom_k1_mose", "neural_atom_k1_mose_hidden_bn"}:
        import torch.nn as nn

        from .pcqm_mose import MOSE_DIM

        model = make_k1("neural_atom_k1")
        hidden_channels = HIDDEN_CHANNELS
        model.rwse_dim = MOSE_DIM
        layers = [nn.Linear(MOSE_DIM, hidden_channels)]
        if mode == "neural_atom_k1_mose_hidden_bn":
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.extend(
            [
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ]
        )
        model.rwse_encoder = nn.Sequential(*layers)
        return model

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
            elif mode == "neural_atom_k1_dynamic_query":
                self.query_conditioners = nn.ModuleDict(
                    {
                        str(layer): nn.Linear(
                            HIDDEN_CHANNELS,
                            64,
                            bias=False,
                        )
                        for layer in MIXER_LAYERS
                    }
                )
                for conditioner in self.query_conditioners.values():
                    nn.init.zeros_(conditioner.weight)
            elif mode == "neural_atom_k1_repset_readout":
                self.repset_readout = _RepSetReadoutFactory.make()
            elif mode == "neural_atom_k1_tied_selector":
                template = self.base.neural_atom_mixers[str(MIXER_LAYERS[0])]
                self.shared_selector = _SharedSelectorFactory.make(template)
                for mixer in self.base.neural_atom_mixers.values():
                    del mixer.node_key
                    del mixer.slot_query
            elif mode == "neural_atom_k1_collapsed_mha":
                self.collapsed_slot_projections = nn.ModuleDict()
                for layer in MIXER_LAYERS:
                    mixer = self.base.neural_atom_mixers[str(layer)]
                    self.collapsed_slot_projections[str(layer)] = (
                        _CollapsedSlotProjectionFactory.make(
                            mixer.slot_attention
                        )
                    )
                    del mixer.slot_attention
            elif mode in {
                "neural_atom_k1_no_slot_attention",
                "neural_atom_k1_no_attention_uniform_return",
            }:
                for mixer in self.base.neural_atom_mixers.values():
                    del mixer.slot_attention

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
                elif self.mode == "neural_atom_k4_cluster":
                    update, _, _, _, _ = _cluster_mixer_update(
                        mixer, h, batch
                    )
                    h = h + update
                elif self.mode == "neural_atom_k1_h4":
                    update, _, _, _, _ = _multihead_single_slot_update(
                        mixer, h, batch
                    )
                    h = h + update
                elif self.mode == "neural_atom_k1_repset_readout":
                    h = mixer(h, batch)
                elif self.mode == "neural_atom_k1_tied_selector":
                    update, _, _, _, _ = _tied_selector_single_slot_update(
                        mixer,
                        self.shared_selector,
                        h,
                        batch,
                    )
                    h = h + update
                elif self.mode == "neural_atom_k1_collapsed_mha":
                    update, _, _, _, _ = _single_slot_processor_update(
                        mixer,
                        self.collapsed_slot_projections[str(layer)],
                        h,
                        batch,
                    )
                    h = h + update
                elif self.mode == "neural_atom_k1_no_slot_attention":
                    update, _, _, _, _ = _single_slot_processor_update(
                        mixer,
                        None,
                        h,
                        batch,
                    )
                    h = h + update
                elif self.mode in {
                    "neural_atom_k1_uniform_return",
                    "neural_atom_k1_inverse_return",
                    "neural_atom_k1_no_attention_uniform_return",
                }:
                    return_mode = (
                        "uniform"
                        if self.mode in {
                            "neural_atom_k1_uniform_return",
                            "neural_atom_k1_no_attention_uniform_return",
                        }
                        else "inverse-score"
                    )
                    update, _, _, _, _, _ = _return_allocation_update(
                        mixer,
                        h,
                        batch,
                        return_mode,
                        remove_slot_attention=(
                            self.mode
                            == "neural_atom_k1_no_attention_uniform_return"
                        ),
                    )
                    h = h + update
                else:
                    update, _, _, _, _ = _dynamic_query_single_slot_update(
                        mixer,
                        self.query_conditioners[str(layer)],
                        h,
                        batch,
                    )
                    h = h + update
            pooled = self.base._pool(h, batch)
            if self.mode == "neural_atom_k1_repset_readout":
                pooled = pooled + self.repset_readout(h, batch)
            return pooled

    return K1Variant()
