"""Topology-conditioned sparse pair return for the frozen K1 encoder."""
from __future__ import annotations

import math


MODE = "neural_atom_k1_sparse_pair"
BASE_MODE = "neural_atom_k1"
HIDDEN_CHANNELS = 192
PAIR_CHANNELS = 32
RWSE_DIM = 16
TARGET_LAYER = 6
MAX_PAIR_DISTANCE = 3
PATH_ORDERS = 5
TOPOLOGY_CHANNELS = PATH_ORDERS + MAX_PAIR_DISTANCE + 2 * RWSE_DIM
BASE_PARAMETERS = 3_658_817
ADDED_PARAMETERS = 22_416
PARAMETERS = BASE_PARAMETERS + ADDED_PARAMETERS


class _SparsePairFactory:
    @staticmethod
    def make():
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch_geometric.utils import to_dense_batch

        class SparsePairReturn(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_norm = nn.LayerNorm(HIDDEN_CHANNELS)
                self.receiver = nn.Linear(
                    HIDDEN_CHANNELS, PAIR_CHANNELS, bias=False
                )
                self.sender = nn.Linear(
                    HIDDEN_CHANNELS, PAIR_CHANNELS, bias=False
                )
                self.topology_norm = nn.LayerNorm(TOPOLOGY_CHANNELS)
                self.topology = nn.Sequential(
                    nn.Linear(TOPOLOGY_CHANNELS, PAIR_CHANNELS),
                    nn.SiLU(),
                    nn.Linear(PAIR_CHANNELS, PAIR_CHANNELS),
                )
                self.pair_norm = nn.LayerNorm(PAIR_CHANNELS)
                self.score = nn.Linear(PAIR_CHANNELS, 1, bias=False)
                self.value = nn.Linear(PAIR_CHANNELS, PAIR_CHANNELS)
                self.return_projection = nn.Linear(
                    PAIR_CHANNELS, HIDDEN_CHANNELS, bias=False
                )
                nn.init.zeros_(self.return_projection.weight)

            @staticmethod
            def _dense_adjacency(edge_index, batch, graph_count, max_nodes):
                counts = torch.bincount(batch, minlength=graph_count)
                pointers = torch.cat(
                    [counts.new_zeros(1), counts.cumsum(dim=0)]
                )
                source, target = edge_index
                edge_graph = batch[source]
                if not torch.equal(edge_graph, batch[target]):
                    raise ValueError("edge_index crosses graph boundaries")
                local_source = source - pointers[edge_graph]
                local_target = target - pointers[edge_graph]
                adjacency = torch.zeros(
                    (graph_count, max_nodes, max_nodes),
                    dtype=torch.float32,
                    device=edge_index.device,
                )
                adjacency[edge_graph, local_source, local_target] = 1.0
                diagonal = torch.arange(max_nodes, device=edge_index.device)
                adjacency[:, diagonal, diagonal] = 0.0
                return adjacency

            @staticmethod
            def _nonbacktracking_paths(adjacency):
                """Return order-1..5 non-backtracking walk counts.

                These bounded counts are a deterministic, inexpensive proxy for
                simple-path structural encoding. They retain alternative cyclic
                routes without constructing a persistent all-pairs state.
                """
                degree = adjacency.sum(dim=-1)
                identity = torch.eye(
                    adjacency.shape[-1],
                    dtype=adjacency.dtype,
                    device=adjacency.device,
                ).unsqueeze(0)
                first = adjacency
                second = torch.bmm(first, adjacency) - degree.unsqueeze(-1) * identity
                counts = [first, second.clamp_min(0.0)]
                previous_previous, previous = first, counts[-1]
                degree_minus_one = (degree - 1.0).clamp_min(0.0).unsqueeze(1)
                for _ in range(3, PATH_ORDERS + 1):
                    current = torch.bmm(previous, adjacency)
                    current = current - previous_previous * degree_minus_one
                    current = current.clamp_min(0.0)
                    counts.append(current)
                    previous_previous, previous = previous, current
                return counts

            def compute_update(
                self,
                hidden,
                edge_index,
                batch,
                random_walk_pe,
            ):
                dense_hidden, valid = to_dense_batch(
                    self.node_norm(hidden), batch
                )
                dense_rwse, rwse_valid = to_dense_batch(
                    random_walk_pe.float(), batch
                )
                if not torch.equal(valid, rwse_valid):
                    raise ValueError("hidden and RWSE batching differ")
                graph_count, max_nodes, _ = dense_hidden.shape
                adjacency = self._dense_adjacency(
                    edge_index, batch, graph_count, max_nodes
                )
                path_counts = self._nonbacktracking_paths(adjacency)
                off_diagonal = ~torch.eye(
                    max_nodes,
                    dtype=torch.bool,
                    device=hidden.device,
                ).unsqueeze(0)
                pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & off_diagonal
                distance_masks = []
                seen = torch.zeros_like(pair_valid)
                for count in path_counts[:MAX_PAIR_DISTANCE]:
                    reached = count.gt(0.0) & pair_valid & ~seen
                    distance_masks.append(reached)
                    seen = seen | reached
                candidate = seen
                pair_index = candidate.nonzero(as_tuple=True)

                if pair_index[0].numel() == 0:
                    return hidden, {
                        "candidate": candidate,
                        "assignment": adjacency.new_zeros(adjacency.shape),
                        "path_counts": path_counts,
                        "distance_masks": distance_masks,
                    }

                graph_index, receiver_index, sender_index = pair_index
                receiver = self.receiver(
                    dense_hidden[graph_index, receiver_index]
                )
                sender = self.sender(
                    dense_hidden[graph_index, sender_index]
                )
                path_features = torch.stack(
                    [
                        torch.log1p(
                            count[graph_index, receiver_index, sender_index]
                        )
                        for count in path_counts
                    ],
                    dim=-1,
                )
                distance_features = torch.stack(
                    [
                        mask[graph_index, receiver_index, sender_index]
                        for mask in distance_masks
                    ],
                    dim=-1,
                ).to(path_features.dtype)
                receiver_rwse = dense_rwse[
                    graph_index, receiver_index
                ]
                sender_rwse = dense_rwse[graph_index, sender_index]
                topology_features = torch.cat(
                    [
                        path_features,
                        distance_features,
                        (receiver_rwse - sender_rwse).abs(),
                        receiver_rwse * sender_rwse,
                    ],
                    dim=-1,
                )
                topology = self.topology(
                    self.topology_norm(topology_features)
                )
                pair = self.pair_norm(
                    functional.silu(receiver + sender + topology)
                )
                selected_logits = self.score(pair).squeeze(-1)
                logits = selected_logits.new_full(
                    (graph_count, max_nodes, max_nodes), float("-inf")
                )
                logits[pair_index] = selected_logits / math.sqrt(PAIR_CHANNELS)
                receiver_has_pair = candidate.any(dim=-1, keepdim=True)
                logits = torch.where(
                    receiver_has_pair, logits, torch.zeros_like(logits)
                )
                assignment = torch.softmax(logits, dim=-1) * candidate.to(
                    logits.dtype
                )
                selected_assignment = assignment[pair_index]
                selected_value = self.value(pair) * selected_assignment.unsqueeze(-1)
                dense_value = selected_value.new_zeros(
                    (graph_count, max_nodes, max_nodes, PAIR_CHANNELS)
                )
                dense_value[pair_index] = selected_value
                returned = dense_value.sum(dim=2)
                update = self.return_projection(returned[valid])
                return hidden + update, {
                    "candidate": candidate,
                    "assignment": assignment,
                    "path_counts": path_counts,
                    "distance_masks": distance_masks,
                }

            def forward(
                self,
                hidden,
                edge_index,
                batch,
                random_walk_pe,
            ):
                output, _ = self.compute_update(
                    hidden, edge_index, batch, random_walk_pe
                )
                return output

        return SparsePairReturn()


def make_encoder(mode: str = MODE):
    if mode != MODE:
        raise ValueError(mode)

    import torch.nn as nn

    from .qm9_neural_atom import make_encoder as make_k1

    class K1SparsePair(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_k1(BASE_MODE)
            self.sparse_pair = _SparsePairFactory.make()

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
                if layer == TARGET_LAYER:
                    hidden = self.sparse_pair(
                        hidden, edge_index, batch, random_walk_pe
                    )
            return self.base._pool(hidden, batch)

    return K1SparsePair()


def mechanism_summary(model, hidden, edge_index, batch, random_walk_pe) -> dict:
    import torch

    output, diagnostics = model.sparse_pair.compute_update(
        hidden, edge_index, batch, random_walk_pe
    )
    candidate = diagnostics["candidate"]
    assignment = diagnostics["assignment"]
    receiver_has_pair = candidate.any(dim=-1)
    mass = assignment.sum(dim=-1)
    checks = {
        "target_layer": TARGET_LAYER,
        "pair_channels": PAIR_CHANNELS,
        "maximum_pair_distance": MAX_PAIR_DISTANCE,
        "path_orders": PATH_ORDERS,
        "pair_normalization": "per-pair-across-channels",
        "selection_normalization": "candidate-senders-per-receiver",
        "return": "directional-pair-to-receiver",
        "persistent_all_pair_state": False,
        "geometry_model_input": False,
        "self_pairs_excluded": bool(
            torch.count_nonzero(
                torch.diagonal(candidate, dim1=1, dim2=2)
            ).item()
            == 0
        ),
        "receiver_mass_one": bool(
            torch.allclose(
                mass[receiver_has_pair],
                torch.ones_like(mass[receiver_has_pair]),
                atol=1e-6,
                rtol=0,
            )
        ),
        "noncandidate_mass_zero": bool(
            assignment.masked_select(~candidate).abs().sum().item() == 0.0
        ),
        "zero_return_projection": bool(
            torch.count_nonzero(
                model.sparse_pair.return_projection.weight
            ).item()
            == 0
        ),
        "zero_initial_update_exact": bool(torch.equal(output, hidden)),
        "candidate_pairs": int(candidate.sum().item()),
    }
    required = (
        "self_pairs_excluded",
        "receiver_mass_one",
        "noncandidate_mass_zero",
        "zero_return_projection",
        "zero_initial_update_exact",
    )
    if not all(checks[name] for name in required):
        raise RuntimeError(f"Sparse-pair invariant failed: {checks}")
    return checks
