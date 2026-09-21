"""One-shot directed triplet interaction beside the accepted K1 PairToken."""
from __future__ import annotations


MODE = "neural_atom_k1_pair_token_oneshot_triplet"
MODES = (MODE,)
HIDDEN_CHANNELS = 192
EDGE_CHANNELS = 64
TRIPLET_CHANNELS = 32
TARGET_LAYER = 6
PARAMETERS = {MODE: 3_694_753}


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
    import torch.nn.functional as functional

    from .k1_pair_token import MODE as PAIR_TOKEN_MODE, _PairTokenFactory
    from .qm9_neural_atom import make_encoder as make_k1

    class OneShotTripletAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            context_channels = HIDDEN_CHANNELS + 2 * EDGE_CHANNELS
            self.context_norm = nn.LayerNorm(context_channels)
            self.message = nn.Linear(context_channels, TRIPLET_CHANNELS)
            self.message_norm = nn.LayerNorm(TRIPLET_CHANNELS)
            self.return_projection = nn.Linear(TRIPLET_CHANNELS, EDGE_CHANNELS)
            nn.init.zeros_(self.return_projection.weight)
            nn.init.zeros_(self.return_projection.bias)

        @staticmethod
        def validate(edge_index, wedge_edge_ids):
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

        def compute_update(self, hidden, edge_index, edge_state, wedge_edge_ids):
            self.validate(edge_index, wedge_edge_ids)
            if wedge_edge_ids.shape[0] == 0:
                return edge_state.new_zeros(edge_state.shape), {
                    "triplet_message": edge_state.new_zeros((0, TRIPLET_CHANNELS)),
                    "target_edge": wedge_edge_ids.new_zeros((0,)),
                }
            first, second = wedge_edge_ids.unbind(dim=1)
            centers = edge_index[1, first]
            context = torch.cat(
                (edge_state[first], hidden[centers], edge_state[second]), dim=-1
            )
            message = self.message_norm(
                functional.silu(self.message(self.context_norm(context)))
            )
            aggregated = _aggregate(message, second, edge_state.shape[0])
            update = self.return_projection(aggregated)
            return update, {
                "triplet_message": message,
                "target_edge": second,
                "source_edge": first,
                "center_node": centers,
            }

        def forward(self, hidden, edge_index, edge_state, wedge_edge_ids):
            update, _ = self.compute_update(
                hidden, edge_index, edge_state, wedge_edge_ids
            )
            return edge_state + update

    class K1OneShotTripletPairToken(nn.Module):
        requires_wedge_topology = True

        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.triplet_adapter = OneShotTripletAdapter()
            self.relation_token = _PairTokenFactory.make(PAIR_TOKEN_MODE)

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
            self.triplet_adapter.validate(edge_index, wedge_edge_ids)
            hidden = self.base._embed_nodes(x)
            hidden = hidden + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(hidden, edge_index, edge_state)
                if layer == TARGET_LAYER:
                    edge_state = self.triplet_adapter(
                        hidden, edge_index, edge_state, wedge_edge_ids
                    )
                hidden = block(hidden, edge_index, batch, edge_attr=edge_state)
                if str(layer) in self.base.neural_atom_mixers:
                    hidden = self.base.neural_atom_mixers[str(layer)](hidden, batch)
                if layer == TARGET_LAYER:
                    hidden = self.relation_token(hidden, batch)
            return self.base._pool(hidden, batch)

    return K1OneShotTripletPairToken()


def check_mechanism(model, batch) -> dict:
    import torch

    from .pcqm_wedge import directed_nonbacktracking_wedges

    wedge = batch.wedge_edge_ids
    model.triplet_adapter.validate(batch.edge_index, wedge)
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
    encoded = first * batch.edge_index.shape[1] + second
    expected_encoded = (
        expected_wedge[:, 0] * batch.edge_index.shape[1] + expected_wedge[:, 1]
    )
    probe_hidden = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.num_nodes) * HIDDEN_CHANNELS,
        device=batch.batch.device,
    ).reshape(int(batch.num_nodes), HIDDEN_CHANNELS)
    probe_edge = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.edge_index.shape[1]) * EDGE_CHANNELS,
        device=batch.batch.device,
    ).reshape(int(batch.edge_index.shape[1]), EDGE_CHANNELS)
    update, diagnostics = model.triplet_adapter.compute_update(
        probe_hidden, batch.edge_index, probe_edge, wedge
    )
    checks = {
        "target_layer": TARGET_LAYER,
        "triplet_channels": TRIPLET_CHANNELS,
        "triplet_semantics": "directed-non-backtracking-i-to-j-to-k",
        "interaction": "one-shot-incoming-edge-to-outgoing-edge",
        "persistent_triplet_state": False,
        "direct_node_return": False,
        "topology_only": True,
        "adjacency_exact": bool(
            (batch.edge_index[1, first] == batch.edge_index[0, second]).all()
        ),
        "non_backtracking_exact": bool(
            (batch.edge_index[0, first] != batch.edge_index[1, second]).all()
        ),
        "center_identity_exact": bool(
            torch.equal(centers, batch.edge_index[0, second])
        ),
        "wedge_count_exact": int(wedge.shape[0]) == int(expected_wedge.shape[0]),
        "wedge_set_exact": bool(
            torch.equal(torch.sort(encoded).values, torch.sort(expected_encoded).values)
        ),
        "message_targets_outgoing_edge": bool(
            torch.equal(diagnostics["target_edge"], second)
        ),
        "zero_triplet_return": bool(
            torch.count_nonzero(model.triplet_adapter.return_projection.weight).item() == 0
            and torch.count_nonzero(model.triplet_adapter.return_projection.bias).item() == 0
        ),
        "zero_initial_triplet_update_exact": bool(torch.count_nonzero(update).item() == 0),
        "zero_pairtoken_return": bool(
            torch.count_nonzero(
                model.relation_token.return_projection.weight
            ).item()
            == 0
        ),
        "geometry_used": False,
        "dense_atom_attention_added": False,
    }
    required = (
        "adjacency_exact",
        "non_backtracking_exact",
        "center_identity_exact",
        "wedge_count_exact",
        "wedge_set_exact",
        "message_targets_outgoing_edge",
        "zero_triplet_return",
        "zero_initial_triplet_update_exact",
        "zero_pairtoken_return",
    )
    if wedge.shape[0] == 0 or not all(checks[name] is True for name in required):
        raise RuntimeError(f"One-shot triplet invariant failed: {checks}")
    return checks
