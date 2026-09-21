"""Persistent sparse topological triplet state beside frozen K1."""
from __future__ import annotations


MODE = "neural_atom_k1_sparse_triplet"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
EDGE_CHANNELS = 64
TRIPLET_CHANNELS = 16
LAYERS = 9
# Filled from the architecture identity test; kept explicit for remote fail-closed use.
PARAMETERS = {MODE: 3_766_001}


def _aggregate(state, index, size: int):
    context = state.new_zeros((size, state.shape[1]))
    counts = state.new_zeros((size, 1))
    context.index_add_(0, index, state)
    counts.index_add_(0, index, state.new_ones((state.shape[0], 1)))
    return context / counts.clamp_min_(1.0)


def make_encoder(mode: str):
    if mode not in MODES:
        raise ValueError(mode)

    import torch
    import torch.nn as nn

    from .pcqm_gap_architecture import _SparseWedgeStateUpdate
    from .qm9_neural_atom import make_encoder as make_k1

    class K1SparseTriplet(nn.Module):
        requires_wedge_topology = True

        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            context_channels = HIDDEN_CHANNELS + 2 * EDGE_CHANNELS
            self.triplet_initial = nn.Sequential(
                nn.LayerNorm(context_channels),
                nn.Linear(context_channels, TRIPLET_CHANNELS),
                nn.LayerNorm(TRIPLET_CHANNELS),
            )
            self.triplet_updates = nn.ModuleList(
                _SparseWedgeStateUpdate(
                    HIDDEN_CHANNELS,
                    EDGE_CHANNELS,
                    TRIPLET_CHANNELS,
                    0.05,
                )
                for _ in range(LAYERS)
            )
            self.triplet_to_edge = nn.ModuleList(
                nn.Linear(TRIPLET_CHANNELS, EDGE_CHANNELS)
                for _ in range(LAYERS)
            )
            self.triplet_to_node = nn.ModuleList(
                nn.Linear(TRIPLET_CHANNELS, HIDDEN_CHANNELS)
                for _ in range(LAYERS)
            )
            for projection in (*self.triplet_to_edge, *self.triplet_to_node):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

        @staticmethod
        def _validate(edge_index, wedge_edge_ids):
            if wedge_edge_ids.ndim != 2 or wedge_edge_ids.shape[1] != 2:
                raise ValueError("wedge_edge_ids must have shape [W, 2]")
            if wedge_edge_ids.numel() == 0:
                return
            if int(wedge_edge_ids.min()) < 0 or int(wedge_edge_ids.max()) >= edge_index.shape[1]:
                raise ValueError("wedge edge id outside batched edge range")
            first, second = wedge_edge_ids.unbind(dim=1)
            if not bool((edge_index[1, first] == edge_index[0, second]).all()):
                raise ValueError("wedge edges are not adjacent")
            if not bool((edge_index[0, first] != edge_index[1, second]).all()):
                raise ValueError("wedge contains a backtracking triplet")

        def forward(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
        ):
            return self.base.head(
                self.encode(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    random_walk_pe,
                    wedge_edge_ids,
                )
            )

        def encode(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
        ):
            expected = (x.shape[0], self.base.rwse_dim)
            if tuple(random_walk_pe.shape) != expected:
                raise ValueError(
                    f"random_walk_pe must have shape {expected}, got {tuple(random_walk_pe.shape)}"
                )
            self._validate(edge_index, wedge_edge_ids)
            hidden = self.base._embed_nodes(x)
            hidden = hidden + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            first, second = wedge_edge_ids.unbind(dim=1)
            centers = edge_index[1, first]
            triplet_state = None
            if wedge_edge_ids.shape[0]:
                triplet_state = self.triplet_initial(
                    torch.cat(
                        (edge_state[first], edge_state[second], hidden[centers]), dim=-1
                    )
                )
            for layer, modules in enumerate(
                zip(
                    self.base.edge_updates,
                    self.base.local_blocks,
                    self.triplet_updates,
                    self.triplet_to_edge,
                    self.triplet_to_node,
                ),
                start=1,
            ):
                edge_update, block, triplet_update, edge_projection, node_projection = modules
                edge_state = edge_update(hidden, edge_index, edge_state)
                if triplet_state is not None:
                    context = torch.cat(
                        (edge_state[first], edge_state[second], hidden[centers]),
                        dim=-1,
                    )
                    triplet_state = triplet_update(context, triplet_state)
                    edge_context = _aggregate(
                        torch.cat((triplet_state, triplet_state), dim=0),
                        torch.cat((first, second), dim=0),
                        edge_state.shape[0],
                    )
                    node_context = _aggregate(
                        triplet_state,
                        centers,
                        hidden.shape[0],
                    )
                    edge_state = edge_state + edge_projection(edge_context)
                    hidden = hidden + node_projection(node_context)
                hidden = block(hidden, edge_index, batch, edge_attr=edge_state)
                if str(layer) in self.base.neural_atom_mixers:
                    hidden = self.base.neural_atom_mixers[str(layer)](hidden, batch)
            return self.base._pool(hidden, batch)

    return K1SparseTriplet()


def check_mechanism(model, batch) -> dict:
    import torch
    from .pcqm_wedge import directed_nonbacktracking_wedges

    wedge = batch.wedge_edge_ids
    model._validate(batch.edge_index, wedge)
    first, second = wedge.unbind(dim=1)
    centers = batch.edge_index[1, first]
    expected_parts = []
    edge_offset = 0
    for graph_id in range(int(batch.num_graphs)):
        node_mask = batch.batch == graph_id
        node_ids = torch.nonzero(node_mask, as_tuple=False).view(-1)
        edge_mask = node_mask[batch.edge_index[0]]
        local_edges = batch.edge_index[:, edge_mask] - int(node_ids[0])
        expected = directed_nonbacktracking_wedges(local_edges.cpu()).to(wedge.device)
        expected_parts.append(expected + edge_offset)
        edge_offset += int(edge_mask.sum())
    expected_wedge = torch.cat(expected_parts, dim=0)
    encoded = wedge[:, 0] * batch.edge_index.shape[1] + wedge[:, 1]
    expected_encoded = (
        expected_wedge[:, 0] * batch.edge_index.shape[1] + expected_wedge[:, 1]
    )
    checks = {
        "triplet_semantics": "directed-non-backtracking-i-to-j-to-k",
        "triplet_channels": TRIPLET_CHANNELS,
        "triplet_layers": LAYERS,
        "topology_only": True,
        "adjacency_exact": bool((batch.edge_index[1, first] == batch.edge_index[0, second]).all()),
        "non_backtracking_exact": bool((batch.edge_index[0, first] != batch.edge_index[1, second]).all()),
        "center_identity_exact": bool(torch.equal(centers, batch.edge_index[0, second])),
        "wedge_count_exact": int(wedge.shape[0]) == int(expected_wedge.shape[0]),
        "wedge_pairs_unique": int(torch.unique(encoded).numel()) == int(wedge.shape[0]),
        "wedge_set_exact": bool(
            torch.equal(torch.sort(encoded).values, torch.sort(expected_encoded).values)
        ),
        "zero_edge_returns": all(torch.count_nonzero(p.weight).item() == 0 and torch.count_nonzero(p.bias).item() == 0 for p in model.triplet_to_edge),
        "zero_node_returns": all(torch.count_nonzero(p.weight).item() == 0 and torch.count_nonzero(p.bias).item() == 0 for p in model.triplet_to_node),
        "dense_atom_attention_added": False,
        "geometry_used": False,
    }
    required = (
        "adjacency_exact",
        "non_backtracking_exact",
        "center_identity_exact",
        "wedge_count_exact",
        "wedge_pairs_unique",
        "wedge_set_exact",
        "zero_edge_returns",
        "zero_node_returns",
    )
    if wedge.shape[0] == 0 or not all(checks[name] is True for name in required):
        raise RuntimeError(f"Sparse triplet invariant failed: {checks}")
    return checks
