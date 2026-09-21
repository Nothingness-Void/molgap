"""Graphormer-style shortest-path bias for the accepted K1 PairToken."""
from __future__ import annotations

import math


MODE = "neural_atom_k1_spd_pair_token"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
PAIR_CHANNELS = 32
TARGET_LAYER = 6
MAX_EXACT_DISTANCE = 5
DISTANCE_BUCKETS = MAX_EXACT_DISTANCE + 2
PARAMETERS = {MODE: 3_681_672}


def _dense_shortest_path_buckets(edge_index, batch, max_nodes: int):
    import torch
    from torch_geometric.utils import to_dense_adj

    adjacency = to_dense_adj(
        edge_index,
        batch=batch,
        max_num_nodes=max_nodes,
    ).bool()
    graph_count = int(adjacency.shape[0])
    distance = torch.full(
        (graph_count, max_nodes, max_nodes),
        MAX_EXACT_DISTANCE + 1,
        dtype=torch.long,
        device=edge_index.device,
    )
    diagonal = torch.arange(max_nodes, device=edge_index.device)
    distance[:, diagonal, diagonal] = 0
    distance.masked_fill_(adjacency, 1)
    frontier = adjacency
    adjacency_float = adjacency.float()
    for hop in range(2, MAX_EXACT_DISTANCE + 1):
        frontier = torch.bmm(frontier.float(), adjacency_float) > 0
        distance.masked_fill_(frontier & (distance > hop), hop)
    return distance


def make_encoder(mode: str):
    if mode not in MODES:
        raise ValueError(mode)

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch_geometric.utils import to_dense_batch

    from .k1_pair_token import MODE as PAIR_TOKEN_MODE, _PairTokenFactory
    from .qm9_neural_atom import make_encoder as make_k1

    class SPDPairToken(nn.Module):
        def __init__(self):
            super().__init__()
            # Construct the accepted PairToken first so its seeded identity is
            # unchanged; the new path can affect only pair-selection logits.
            self.base_token = _PairTokenFactory.make(PAIR_TOKEN_MODE)
            self.distance_bias = nn.Embedding(DISTANCE_BUCKETS, 1)
            nn.init.zeros_(self.distance_bias.weight)

        def compute_update(self, hidden, edge_index, batch):
            dense, valid = to_dense_batch(hidden, batch)
            source = self.base_token.source(dense).unsqueeze(2)
            target = self.base_token.target(dense).unsqueeze(1)
            pair = self.base_token.pair_norm(functional.silu(source + target))
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
            base_logits = torch.einsum(
                "bijd,d->bij", pair, self.base_token.query
            ) / math.sqrt(PAIR_CHANNELS)
            buckets = _dense_shortest_path_buckets(
                edge_index, batch, dense.shape[1]
            )
            spd_bias = self.distance_bias(buckets).squeeze(-1)
            logits = (base_logits + spd_bias).masked_fill(
                ~pair_valid, float("-inf")
            )
            assignment = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
            token = torch.einsum("bij,bijd->bd", assignment, pair)
            token = self.base_token.token_norm(
                token + self.base_token.token_ffn(token)
            )
            update = self.base_token.return_projection(token[batch])
            return update, {
                "assignment": assignment,
                "base_logits": base_logits,
                "spd_bias": spd_bias,
                "distance_buckets": buckets,
                "pair_valid": pair_valid,
                "valid": valid,
            }

        def forward(self, hidden, edge_index, batch):
            update, _ = self.compute_update(hidden, edge_index, batch)
            return hidden + update

    class K1SPDPairToken(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.relation_token = SPDPairToken()

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
            )

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            expected = (x.shape[0], self.base.rwse_dim)
            if tuple(random_walk_pe.shape) != expected:
                raise ValueError(
                    f"random_walk_pe must have shape {expected}, got {tuple(random_walk_pe.shape)}"
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
                    hidden = self.base.neural_atom_mixers[str(layer)](hidden, batch)
                if layer == TARGET_LAYER:
                    hidden = self.relation_token(hidden, edge_index, batch)
            return self.base._pool(hidden, batch)

    return K1SPDPairToken()


def check_mechanism(model, batch) -> dict:
    import torch

    probe = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.num_nodes) * HIDDEN_CHANNELS,
        device=batch.batch.device,
    ).reshape(int(batch.num_nodes), HIDDEN_CHANNELS)
    update, diagnostics = model.relation_token.compute_update(
        probe, batch.edge_index, batch.batch
    )
    assignment = diagnostics["assignment"]
    pair_valid = diagnostics["pair_valid"]
    valid = diagnostics["valid"]
    buckets = diagnostics["distance_buckets"]
    valid_nodes = valid.sum(dim=-1)
    checks = {
        "target_layer": TARGET_LAYER,
        "pair_channels": PAIR_CHANNELS,
        "distance_buckets": DISTANCE_BUCKETS,
        "maximum_exact_distance": MAX_EXACT_DISTANCE,
        "selection_conditioning": "graphormer-shortest-path-scalar-logit-bias",
        "pair_value": "accepted-additive-pairtoken-value-unchanged",
        "path_value_aggregation": False,
        "persistent_path_state": False,
        "dense_atom_to_atom_attention": False,
        "valid_pair_count_exact": bool(
            torch.equal(pair_valid.sum(dim=(1, 2)), valid_nodes.square())
        ),
        "assignment_mass_one": bool(
            torch.allclose(
                assignment.sum(dim=(1, 2)),
                torch.ones_like(assignment.sum(dim=(1, 2))),
                atol=1e-6,
                rtol=0,
            )
        ),
        "padding_mass_zero": bool(
            assignment.masked_select(~pair_valid).abs().sum().item() == 0.0
        ),
        "distance_range_exact": bool(
            int(buckets.min()) == 0
            and int(buckets.max()) <= MAX_EXACT_DISTANCE + 1
        ),
        "zero_spd_bias": bool(
            torch.count_nonzero(model.relation_token.distance_bias.weight).item() == 0
        ),
        "zero_return_projection": bool(
            torch.count_nonzero(
                model.relation_token.base_token.return_projection.weight
            ).item()
            == 0
        ),
        "zero_initial_update_exact": bool(torch.count_nonzero(update).item() == 0),
    }
    required = (
        "valid_pair_count_exact",
        "assignment_mass_one",
        "padding_mass_zero",
        "distance_range_exact",
        "zero_spd_bias",
        "zero_return_projection",
        "zero_initial_update_exact",
    )
    if not all(checks[name] is True for name in required):
        raise RuntimeError(f"SPD PairToken invariant failed: {checks}")
    return checks
