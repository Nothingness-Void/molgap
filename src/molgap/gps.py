"""GPS (General Powerful Scalable) Graph Transformer for 2D molecular graphs."""
from __future__ import annotations

import torch
import torch.nn as nn


class GPSWrapper(nn.Module):
    """GPS Graph Transformer operating on 2D bond-topology graphs.

    Input: PyG Data with x (atom features), edge_index (bonds),
           edge_attr (bond features), batch.
    """

    def __init__(self, in_channels=9, edge_dim=4, hidden_channels=128,
                 num_layers=6, num_heads=8, dropout=0.1, n_targets=3,
                 pooling="mean"):
        super().__init__()
        from torch_geometric.nn import GPSConv, GINEConv

        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported GPS pooling: {pooling}")
        self.pooling = pooling

        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.edge_emb = nn.Linear(edge_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            gin = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                edge_dim=hidden_channels,
            )
            gps = GPSConv(
                channels=hidden_channels,
                conv=gin,
                heads=num_heads,
                dropout=dropout,
                act="silu",
                norm="batch_norm",
                attn_type="multihead",
            )
            self.convs.append(gps)

        if pooling == "mean_max":
            self.pool_proj = nn.Linear(hidden_channels * 2, hidden_channels)
            # Start exactly at mean pooling; training can add max-pooled signal.
            with torch.no_grad():
                self.pool_proj.weight.zero_()
                self.pool_proj.weight[:, :hidden_channels].copy_(torch.eye(hidden_channels))
                self.pool_proj.bias.zero_()

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encode(x, edge_index, edge_attr, batch)
        return self.head(h)

    def _pool(self, h, batch):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        mean = global_mean_pool(h, batch)
        if self.pooling == "mean":
            return mean
        maximum = global_max_pool(h, batch)
        return self.pool_proj(torch.cat([mean, maximum], dim=-1))

    def _embed_nodes(self, x):
        """Embed node features while preserving the legacy float contract."""
        return self.node_emb(x.float())

    def _embed_edges(self, edge_attr):
        """Embed edge features while preserving the legacy float contract."""
        return self.edge_emb(edge_attr.float())

    def encode(self, x, edge_index, edge_attr, batch):
        """Return molecule-level embeddings [num_molecules, hidden_channels]."""
        h = self._embed_nodes(x)
        e = self._embed_edges(edge_attr)

        for conv in self.convs:
            h = conv(h, edge_index, batch, edge_attr=e)

        return self._pool(h, batch)

    def encode_layers(self, x, edge_index, edge_attr, batch, layers=(2, 4, -1)):
        """Return concatenated pooled embeddings from selected GPS layers.

        Layer indices are 1-based after each GPSConv. ``-1`` means the final
        layer. This supports lightweight layer-fusion probes without changing
        the normal production ``encode`` path.
        """
        n_layers = len(self.convs)
        wanted = {n_layers if layer == -1 else int(layer) for layer in layers}
        invalid = [layer for layer in wanted if layer < 1 or layer > n_layers]
        if invalid:
            raise ValueError(f"GPS layer index out of range: {invalid}")

        h = self._embed_nodes(x)
        e = self._embed_edges(edge_attr)
        pooled = []
        for i, conv in enumerate(self.convs, start=1):
            h = conv(h, edge_index, batch, edge_attr=e)
            if i in wanted:
                pooled.append(self._pool(h, batch))
        return torch.cat(pooled, dim=-1)


class StructuralGPSWrapper(GPSWrapper):
    """GPS with precomputed random-walk structural encodings.

    The base GPS modules are initialized before the RWSE projection. With an
    identical Torch seed, all parameters shared with :class:`GPSWrapper` have
    identical initial values for a controlled architecture comparison.
    """

    def __init__(self, *args, rwse_dim=16, **kwargs):
        super().__init__(*args, **kwargs)
        if rwse_dim <= 0:
            raise ValueError("rwse_dim must be positive")
        hidden_channels = self.node_emb.out_features
        self.rwse_dim = int(rwse_dim)
        self.rwse_encoder = nn.Sequential(
            nn.Linear(self.rwse_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
    ):
        if random_walk_pe is None:
            raise ValueError("Structural GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(
                f"random_walk_pe must have shape {expected}, "
                f"got {tuple(random_walk_pe.shape)}"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")

        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        e = self._embed_edges(edge_attr)
        for conv in self.convs:
            h = conv(h, edge_index, batch, edge_attr=e)
        return self._pool(h, batch)


class NormalizedStructuralGPSWrapper(StructuralGPSWrapper):
    """Structural GPS with normalized, gated RWSE input mixing.

    The legacy :class:`StructuralGPSWrapper` remains unchanged so its accepted
    checkpoints retain their exact state-dict contract. This variant tests the
    bounded input equation

    ``h0 = LayerNorm(atom_embedding + alpha * rwse_embedding)``.
    """

    def __init__(self, *args, rwse_alpha_init=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < rwse_alpha_init < 1.0:
            raise ValueError("rwse_alpha_init must fall strictly between 0 and 1")
        hidden_channels = self.node_emb.out_features
        self.rwse_normalizer = nn.BatchNorm1d(self.rwse_dim, affine=False)
        self.h0_norm = nn.LayerNorm(hidden_channels)
        initial = torch.tensor(float(rwse_alpha_init), dtype=torch.float32)
        self.rwse_alpha_logit = nn.Parameter(torch.logit(initial))

    @property
    def rwse_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.rwse_alpha_logit)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
    ):
        if random_walk_pe is None:
            raise ValueError("Normalized Structural GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(
                f"random_walk_pe must have shape {expected}, "
                f"got {tuple(random_walk_pe.shape)}"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")

        atom_embedding = self._embed_nodes(x)
        normalized_rwse = self.rwse_normalizer(random_walk_pe.float())
        rwse_embedding = self.rwse_encoder(normalized_rwse)
        h = self.h0_norm(atom_embedding + self.rwse_alpha * rwse_embedding)
        e = self._embed_edges(edge_attr)
        for conv in self.convs:
            h = conv(h, edge_index, batch, edge_attr=e)
        return self._pool(h, batch)


class GatedStructuralGPSWrapper(StructuralGPSWrapper):
    """Structural GPS with an edge-aware residual gated local branch.

    This leaves :class:`StructuralGPSWrapper` untouched and changes only the
    local operator inside each GPS block. The global attention, RWSE input,
    pooling, and prediction head retain the controlled screen configuration.
    """

    def __init__(
        self,
        in_channels=9,
        edge_dim=4,
        hidden_channels=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        n_targets=3,
        pooling="mean",
        rwse_dim=16,
    ):
        super().__init__(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_targets=n_targets,
            pooling=pooling,
            rwse_dim=rwse_dim,
        )
        from torch_geometric.nn import GPSConv, ResGatedGraphConv

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local = ResGatedGraphConv(
                hidden_channels,
                hidden_channels,
                edge_dim=hidden_channels,
            )
            self.convs.append(
                GPSConv(
                    channels=hidden_channels,
                    conv=local,
                    heads=num_heads,
                    dropout=dropout,
                    act="silu",
                    norm="batch_norm",
                    attn_type="multihead",
                )
            )


class _PersistentEdgeUpdate(nn.Module):
    """Update a compact directed edge state from its incident node states."""

    def __init__(self, node_channels: int, edge_channels: int, dropout: float):
        super().__init__()
        self.source = nn.Linear(node_channels, edge_channels, bias=False)
        self.target = nn.Linear(node_channels, edge_channels, bias=False)
        self.update = nn.Sequential(
            nn.LayerNorm(edge_channels),
            nn.Linear(edge_channels, edge_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_channels, edge_channels),
        )
        self.output_norm = nn.LayerNorm(edge_channels)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        context = edge_state + self.source(h[source]) + self.target(h[target])
        return self.output_norm(edge_state + self.update(context))


class EdgeStateStructuralGPSWrapper(StructuralGPSWrapper):
    """Structural GPS with compact persistent edge states across GPS blocks."""

    def __init__(
        self,
        in_channels=9,
        edge_dim=4,
        hidden_channels=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        n_targets=3,
        pooling="mean",
        rwse_dim=16,
        edge_state_channels=64,
    ):
        if edge_state_channels <= 0:
            raise ValueError("edge_state_channels must be positive")
        super().__init__(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_targets=n_targets,
            pooling=pooling,
            rwse_dim=rwse_dim,
        )
        from torch_geometric.nn import GPSConv, ResGatedGraphConv

        self.edge_state_channels = int(edge_state_channels)
        self.edge_emb = nn.Linear(edge_dim, self.edge_state_channels)
        self.edge_updates = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.edge_updates.append(
                _PersistentEdgeUpdate(
                    hidden_channels,
                    self.edge_state_channels,
                    dropout,
                )
            )
            local = ResGatedGraphConv(
                hidden_channels,
                hidden_channels,
                edge_dim=self.edge_state_channels,
            )
            self.convs.append(
                GPSConv(
                    channels=hidden_channels,
                    conv=local,
                    heads=num_heads,
                    dropout=dropout,
                    act="silu",
                    norm="batch_norm",
                    attn_type="multihead",
                )
            )

    def _condition_nodes_from_edges(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> torch.Tensor:
        return h

    def _initialize_graph_state(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
    ):
        return None

    def _condition_nodes_from_graph(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        graph_state,
    ) -> torch.Tensor:
        return h

    def _update_graph_state(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        graph_state,
    ):
        return graph_state

    def _encode_state_trace(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        capture_layers=(),
    ):
        if random_walk_pe is None:
            raise ValueError("Edge-state Structural GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(
                f"random_walk_pe must have shape {expected}, "
                f"got {tuple(random_walk_pe.shape)}"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")

        requested = tuple(int(layer) for layer in capture_layers)
        if len(set(requested)) != len(requested):
            raise ValueError("capture_layers must be unique")
        invalid = [
            layer for layer in requested if layer < 1 or layer > len(self.convs)
        ]
        if invalid:
            raise ValueError(f"Edge-state layer index out of range: {invalid}")

        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        edge_state = self._embed_edges(edge_attr)
        graph_state = self._initialize_graph_state(h, batch)
        captured = {}
        for layer, (edge_update, conv) in enumerate(
            zip(self.edge_updates, self.convs), start=1
        ):
            edge_state = edge_update(h, edge_index, edge_state)
            h = self._condition_nodes_from_edges(h, edge_index, edge_state)
            h = self._condition_nodes_from_graph(h, batch, graph_state)
            h = conv(h, edge_index, batch, edge_attr=edge_state)
            graph_state = self._update_graph_state(h, batch, graph_state)
            if layer in requested:
                captured[layer] = h
        return h, edge_state, tuple(captured[layer] for layer in requested)

    def _encode_states(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
    ):
        h, edge_state, _ = self._encode_state_trace(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
        )
        return h, edge_state

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
    ):
        h, _ = self._encode_states(
            x, edge_index, edge_attr, batch, random_walk_pe
        )
        return self._pool(h, batch)


class _SparseShortestPathAttention(nn.Module):
    """Shared low-rank attention over cached topology-distance pairs."""

    def __init__(
        self,
        hidden_channels: int,
        rank: int,
        max_distance: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0 or max_distance <= 0:
            raise ValueError("rank and max_distance must be positive")
        self.rank = int(rank)
        self.max_distance = int(max_distance)
        self.norm = nn.LayerNorm(hidden_channels)
        self.query = nn.Linear(hidden_channels, self.rank, bias=False)
        self.key = nn.Linear(hidden_channels, self.rank, bias=False)
        self.value = nn.Linear(hidden_channels, self.rank, bias=False)
        self.distance_bias = nn.Embedding(self.max_distance, 1)
        self.output = nn.Linear(self.rank, hidden_channels, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        h: torch.Tensor,
        multihop_edge_index: torch.Tensor,
        multihop_distance: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        from torch_geometric.utils import softmax

        if multihop_edge_index.ndim != 2 or multihop_edge_index.shape[0] != 2:
            raise ValueError("multihop_edge_index must have shape [2, E]")
        distance = multihop_distance.view(-1).long()
        if distance.numel() != multihop_edge_index.shape[1]:
            raise ValueError("multihop distance does not align with pair edges")
        if distance.numel() == 0:
            raise ValueError("multihop attention requires at least one pair")
        if int(distance.min()) < 1 or int(distance.max()) > self.max_distance:
            raise ValueError("multihop distance falls outside the frozen cap")
        source, target = multihop_edge_index
        if not torch.equal(batch[source], batch[target]):
            raise ValueError("multihop attention contains a cross-graph pair")
        normalized = self.norm(h)
        query = self.query(normalized[target])
        key = self.key(normalized[source])
        logits = (query * key).sum(dim=-1) / (self.rank ** 0.5)
        logits = logits + self.distance_bias(distance - 1).view(-1)
        weights = softmax(logits, target, num_nodes=h.shape[0])
        messages = self.value(normalized[source]) * self.dropout(
            weights.unsqueeze(-1)
        )
        aggregate = messages.new_zeros((h.shape[0], self.rank))
        aggregate.index_add_(0, target, messages)
        return h + self.dropout(self.output(aggregate))


class SparsePathAttentionStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Accepted real-bond EdgeState plus shared shortest-path attention."""

    def __init__(
        self,
        *args,
        path_attention_rank: int = 16,
        path_max_distance: int = 4,
        **kwargs,
    ) -> None:
        dropout = float(kwargs.get("dropout", 0.1))
        super().__init__(*args, **kwargs)
        self.path_attention = _SparseShortestPathAttention(
            hidden_channels=self.node_emb.out_features,
            rank=path_attention_rank,
            max_distance=path_max_distance,
            dropout=dropout,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        multihop_edge_index,
        multihop_distance,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            multihop_edge_index,
            multihop_distance,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        multihop_edge_index,
        multihop_distance,
    ):
        if random_walk_pe is None:
            raise ValueError("Sparse-path Structural GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(
                f"random_walk_pe must have shape {expected}, "
                f"got {tuple(random_walk_pe.shape)}"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")
        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        edge_state = self._embed_edges(edge_attr)
        for edge_update, conv in zip(self.edge_updates, self.convs):
            edge_state = edge_update(h, edge_index, edge_state)
            h = conv(h, edge_index, batch, edge_attr=edge_state)
            h = self.path_attention(
                h,
                multihop_edge_index,
                multihop_distance,
                batch,
            )
        return self._pool(h, batch)


class _DirectedPersistentEdgeUpdate(_PersistentEdgeUpdate):
    """Add non-backtracking incoming bond memory to an edge update."""

    def __init__(self, node_channels: int, edge_channels: int, dropout: float):
        super().__init__(node_channels, edge_channels, dropout)
        self.incoming = nn.Linear(edge_channels, edge_channels, bias=False)
        nn.init.zeros_(self.incoming.weight)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        reverse_edge: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        incoming_sum = edge_state.new_zeros(
            (h.shape[0], edge_state.shape[1])
        )
        incoming_sum.index_add_(0, target, edge_state)
        non_backtracking = incoming_sum[source] - edge_state[reverse_edge]
        context = (
            edge_state
            + self.source(h[source])
            + self.target(h[target])
            + self.incoming(non_backtracking)
        )
        return self.output_norm(edge_state + self.update(context))


class DirectedEdgeStateStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """EdgeState GPS with D-MPNN-style non-backtracking bond flow."""

    def __init__(self, *args, **kwargs) -> None:
        dropout = float(kwargs.get("dropout", 0.1))
        super().__init__(*args, **kwargs)
        hidden_channels = self.node_emb.out_features
        self.edge_updates = nn.ModuleList(
            [
                _DirectedPersistentEdgeUpdate(
                    hidden_channels,
                    self.edge_state_channels,
                    dropout,
                )
                for _ in self.convs
            ]
        )

    @staticmethod
    def _reverse_edge_indices(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        source, target = edge_index
        keys = source * num_nodes + target
        reverse_keys = target * num_nodes + source
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        positions = torch.searchsorted(sorted_keys, reverse_keys)
        if bool((positions >= len(sorted_keys)).any()):
            raise ValueError("Directed EdgeState requires reverse bond edges")
        reverse_edge = order[positions]
        if not torch.equal(source[reverse_edge], target) or not torch.equal(
            target[reverse_edge], source
        ):
            raise ValueError("Directed EdgeState reverse-edge mapping is invalid")
        return reverse_edge

    def _encode_state_trace(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        capture_layers=(),
    ):
        if random_walk_pe is None:
            raise ValueError("Directed EdgeState GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(
                f"random_walk_pe must have shape {expected}, "
                f"got {tuple(random_walk_pe.shape)}"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")
        requested = tuple(int(layer) for layer in capture_layers)
        if len(set(requested)) != len(requested):
            raise ValueError("capture_layers must be unique")
        invalid = [
            layer for layer in requested if layer < 1 or layer > len(self.convs)
        ]
        if invalid:
            raise ValueError(f"Directed EdgeState layer index out of range: {invalid}")
        reverse_edge = self._reverse_edge_indices(edge_index, x.shape[0])
        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        edge_state = self._embed_edges(edge_attr)
        captured = {}
        for layer, (edge_update, conv) in enumerate(
            zip(self.edge_updates, self.convs), start=1
        ):
            edge_state = edge_update(
                h,
                edge_index,
                edge_state,
                reverse_edge,
            )
            h = conv(h, edge_index, batch, edge_attr=edge_state)
            if layer in requested:
                captured[layer] = h
        return h, edge_state, tuple(captured[layer] for layer in requested)


class EdgeConditionedStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Persistent-edge GPS with shared node-level edge conditioning.

    Every updated directed edge state is averaged onto its target atom before
    the corresponding GPS block. A shared zero-initialized FiLM transform then
    exposes that bond context to both the local convolution and global
    attention branches while retaining the accepted EdgeState forward path at
    initialization.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_channels = self.node_emb.out_features
        self.edge_context_norm = nn.LayerNorm(self.edge_state_channels)
        self.node_context_norm = nn.LayerNorm(hidden_channels)
        self.edge_to_node_film = nn.Linear(
            self.edge_state_channels,
            2 * hidden_channels,
        )
        nn.init.zeros_(self.edge_to_node_film.weight)
        nn.init.zeros_(self.edge_to_node_film.bias)

    def _condition_nodes_from_edges(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> torch.Tensor:
        target = edge_index[1]
        edge_context = edge_state.new_zeros(
            (h.shape[0], self.edge_state_channels)
        )
        edge_context.index_add_(0, target, edge_state)
        degree = edge_state.new_zeros((h.shape[0], 1))
        degree.index_add_(
            0,
            target,
            edge_state.new_ones((edge_state.shape[0], 1)),
        )
        edge_context = edge_context / degree.clamp_min_(1.0)
        scale, shift = self.edge_to_node_film(
            self.edge_context_norm(edge_context)
        ).chunk(2, dim=-1)
        conditioned = torch.tanh(scale) * self.node_context_norm(h) + shift
        return h + conditioned


class GraphTokenStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Persistent-edge GPS with a recurrent molecule token.

    A compact shared update alternates node-to-graph aggregation and
    graph-to-node broadcast at every depth. Zero-initialized output projections
    preserve the accepted EdgeState path at initialization while the learned
    token provides an explicit graph memory after optimization.
    """

    def __init__(self, *args, token_channels: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if token_channels <= 0:
            raise ValueError("token_channels must be positive")
        hidden_channels = self.node_emb.out_features
        self.graph_token = nn.Parameter(torch.empty(1, hidden_channels))
        nn.init.normal_(self.graph_token, mean=0.0, std=0.02)
        self.token_update_norm = nn.LayerNorm(2 * hidden_channels)
        self.token_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, token_channels),
            nn.SiLU(),
            nn.Linear(token_channels, hidden_channels),
        )
        self.token_broadcast_norm = nn.LayerNorm(hidden_channels)
        self.token_to_node = nn.Linear(hidden_channels, hidden_channels)
        nn.init.zeros_(self.token_update[-1].weight)
        nn.init.zeros_(self.token_update[-1].bias)
        nn.init.zeros_(self.token_to_node.weight)
        nn.init.zeros_(self.token_to_node.bias)

    def _initialize_graph_state(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        return self.graph_token.expand(num_graphs, -1)

    def _condition_nodes_from_graph(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        broadcast = self.token_to_node(self.token_broadcast_norm(graph_state))
        return h + broadcast[batch]

    def _update_graph_state(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        graph_state: torch.Tensor,
    ) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        pooled = global_mean_pool(h, batch, size=graph_state.shape[0])
        update_input = self.token_update_norm(torch.cat((graph_state, pooled), dim=-1))
        return graph_state + self.token_update(update_input)


class EdgeReadoutStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Persistent-edge GPS with an identity-initialized node/edge readout.

    The encoder is identical to :class:`EdgeStateStructuralGPSWrapper`. The
    readout can learn where frontier-orbital information is concentrated and
    can expose the final bond state directly to the graph representation. Its
    final projection is zero-initialized, so initialization exactly preserves
    the accepted mean-pooled encoder rather than perturbing its optimization.
    """

    def __init__(self, *args, readout_channels: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if readout_channels <= 0:
            raise ValueError("readout_channels must be positive")
        hidden_channels = self.node_emb.out_features
        self.node_readout_score = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.node_readout_score.weight)
        nn.init.zeros_(self.node_readout_score.bias)
        self.edge_readout_norm = nn.LayerNorm(self.edge_state_channels)
        self.readout_delta = nn.Sequential(
            nn.Linear(
                hidden_channels + self.edge_state_channels,
                readout_channels,
            ),
            nn.SiLU(),
            nn.Linear(readout_channels, hidden_channels),
        )
        nn.init.zeros_(self.readout_delta[-1].weight)
        nn.init.zeros_(self.readout_delta[-1].bias)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
    ):
        from torch_geometric.nn import global_add_pool, global_mean_pool
        from torch_geometric.utils import softmax

        h, edge_state = self._encode_states(
            x, edge_index, edge_attr, batch, random_walk_pe
        )
        mean = global_mean_pool(h, batch)
        weights = softmax(self.node_readout_score(h), batch)
        attended = global_add_pool(h * weights, batch, size=mean.shape[0])
        edge_batch = batch[edge_index[0]]
        edge_mean = global_mean_pool(
            edge_state, edge_batch, size=mean.shape[0]
        )
        delta_input = torch.cat(
            (attended - mean, self.edge_readout_norm(edge_mean)), dim=-1
        )
        return mean + self.readout_delta(delta_input)


class EdgeJKReadoutStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Persistent-edge GPS with an identity-initialized multi-depth readout.

    Sparse node states from selected depths and the final directed-edge state
    enter a small bottleneck. Its final projection starts at zero, so the first
    forward pass remains exactly the accepted final-layer mean pooling path.
    """

    def __init__(
        self,
        *args,
        readout_layers=(3, 6, 9),
        readout_channels: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if readout_channels <= 0:
            raise ValueError("readout_channels must be positive")
        layers = tuple(int(layer) for layer in readout_layers)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("readout_layers must be non-empty and unique")
        invalid = [layer for layer in layers if layer < 1 or layer > len(self.convs)]
        if invalid:
            raise ValueError(f"Edge-state readout layer out of range: {invalid}")

        hidden_channels = self.node_emb.out_features
        self.readout_layers = layers
        self.layer_readout_norms = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in self.readout_layers
        )
        self.edge_readout_norm = nn.LayerNorm(self.edge_state_channels)
        input_channels = (
            len(self.readout_layers) * hidden_channels + self.edge_state_channels
        )
        self.readout_delta = nn.Sequential(
            nn.Linear(input_channels, readout_channels),
            nn.SiLU(),
            nn.Linear(readout_channels, hidden_channels),
        )
        nn.init.zeros_(self.readout_delta[-1].weight)
        nn.init.zeros_(self.readout_delta[-1].bias)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
    ):
        from torch_geometric.nn import global_mean_pool

        h, edge_state, layer_states = self._encode_state_trace(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            capture_layers=self.readout_layers,
        )
        baseline = self._pool(h, batch)
        layer_summaries = [
            normalizer(global_mean_pool(state, batch))
            for normalizer, state in zip(self.layer_readout_norms, layer_states)
        ]
        edge_batch = batch[edge_index[0]]
        edge_summary = self.edge_readout_norm(
            global_mean_pool(edge_state, edge_batch, size=baseline.shape[0])
        )
        delta_input = torch.cat((*layer_summaries, edge_summary), dim=-1)
        return baseline + self.readout_delta(delta_input)


class CategoricalFeatureEncoder(nn.Module):
    """Sum one learned embedding per categorical molecular feature field."""

    def __init__(self, feature_dims, embedding_dim: int):
        super().__init__()
        dimensions = tuple(int(value) for value in feature_dims)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ValueError("feature_dims must contain positive category counts")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.feature_dims = dimensions
        self.embeddings = nn.ModuleList(
            nn.Embedding(categories, embedding_dim) for categories in dimensions
        )
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.embeddings):
            raise ValueError(
                f"categorical features must have shape [rows, {len(self.embeddings)}]"
            )
        encoded = self.embeddings[0](features[:, 0].long())
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            encoded = encoded + embedding(features[:, column].long())
        return encoded


class CategoricalConcatFeatureEncoder(nn.Module):
    """Preserve categorical field identity before projecting to model width."""

    def __init__(self, feature_dims, embedding_dim: int, *, field_channels: int = 16):
        super().__init__()
        dimensions = tuple(int(value) for value in feature_dims)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ValueError("feature_dims must contain positive category counts")
        if embedding_dim <= 0 or field_channels <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.feature_dims = dimensions
        self.field_channels = int(field_channels)
        self.embeddings = nn.ModuleList(
            nn.Embedding(categories, self.field_channels)
            for categories in dimensions
        )
        self.projection = nn.Linear(
            len(dimensions) * self.field_channels, embedding_dim
        )
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.embeddings):
            raise ValueError(
                f"categorical features must have shape [rows, {len(self.embeddings)}]"
            )
        fields = [
            embedding(features[:, column].long())
            for column, embedding in enumerate(self.embeddings)
        ]
        return self.projection(torch.cat(fields, dim=-1))


class CategoricalEdgeStateStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Persistent EdgeState GPS using complete categorical atom/bond fields."""

    def __init__(
        self,
        *,
        atom_feature_dims,
        bond_feature_dims,
        hidden_channels=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        n_targets=3,
        pooling="mean",
        rwse_dim=16,
        edge_state_channels=64,
        categorical_encoder="sum",
        categorical_field_channels=16,
    ):
        atom_dimensions = tuple(int(value) for value in atom_feature_dims)
        bond_dimensions = tuple(int(value) for value in bond_feature_dims)
        super().__init__(
            in_channels=len(atom_dimensions),
            edge_dim=len(bond_dimensions),
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_targets=n_targets,
            pooling=pooling,
            rwse_dim=rwse_dim,
            edge_state_channels=edge_state_channels,
        )
        if categorical_encoder == "sum":
            encoder = CategoricalFeatureEncoder
            encoder_kwargs = {}
        elif categorical_encoder == "concat_project":
            encoder = CategoricalConcatFeatureEncoder
            encoder_kwargs = {"field_channels": categorical_field_channels}
        else:
            raise ValueError(
                "categorical_encoder must be 'sum' or 'concat_project'"
            )
        self.categorical_encoder = categorical_encoder
        self.node_emb = encoder(atom_dimensions, hidden_channels, **encoder_kwargs)
        self.edge_emb = encoder(
            bond_dimensions, edge_state_channels, **encoder_kwargs
        )


class CategoricalRadicalContextEdgeStateStructuralGPSWrapper(
    CategoricalEdgeStateStructuralGPSWrapper
):
    """Add a graph-level radical-electron context without perturbing closed shells."""

    def __init__(
        self,
        *,
        radical_context_channels=16,
        radical_context_gate_init=0.1,
        **kwargs,
    ):
        if radical_context_channels <= 0:
            raise ValueError("radical_context_channels must be positive")
        if not 0.0 < radical_context_gate_init < 1.0:
            raise ValueError("radical_context_gate_init must be in (0, 1)")
        super().__init__(**kwargs)
        hidden_channels = self.head[0].in_features
        radical_categories = self.node_emb.feature_dims[5]
        self.radical_context = nn.Embedding(
            radical_categories, radical_context_channels, padding_idx=0
        )
        nn.init.xavier_uniform_(self.radical_context.weight)
        with torch.no_grad():
            self.radical_context.weight[0].zero_()
        self.radical_context_projection = nn.Sequential(
            nn.Linear(radical_context_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=False),
        )
        initial = torch.tensor(float(radical_context_gate_init))
        self.radical_context_gate_logit = nn.Parameter(torch.logit(initial))

    @property
    def radical_context_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.radical_context_gate_logit)

    def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
        from torch_geometric.nn import global_add_pool

        embedding = super().encode(
            x, edge_index, edge_attr, batch, random_walk_pe
        )
        atom_context = self.radical_context(x[:, 5].long())
        graph_context = global_add_pool(atom_context, batch)
        correction = self.radical_context_projection(graph_context)
        return embedding + self.radical_context_gate * correction


class OrbitalCenterHead(nn.Module):
    """Predict the frontier-orbital center from a frozen graph embedding."""

    def __init__(self, embedding_dim: int, hidden_channels: int = 128, dropout: float = 0.05):
        super().__init__()
        if embedding_dim <= 0 or hidden_channels <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)


class FrontierCenterGapHead(nn.Module):
    """Predict center and Gap, then reconstruct a consistent frontier triple.

    The network returns values in the normalized three-target contract used by
    the existing trainers, while reconstruction happens in eV so every output
    satisfies ``LUMO - HOMO == Gap`` up to floating-point precision.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_channels: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels or embedding_dim)
        if embedding_dim <= 0 or hidden_channels <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 2),
        )
        self.register_buffer("target_mean", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))
        self.register_buffer("center_mean", torch.zeros(1))
        self.register_buffer("center_std", torch.ones(1))
        self.register_buffer("stats_configured", torch.tensor(False))

    def configure_target_stats(
        self,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        center_mean: torch.Tensor,
        center_std: torch.Tensor,
    ) -> None:
        target_mean = torch.as_tensor(target_mean).float().view(-1)
        target_std = torch.as_tensor(target_std).float().view(-1)
        center_mean = torch.as_tensor(center_mean).float().view(-1)
        center_std = torch.as_tensor(center_std).float().view(-1)
        if target_mean.shape != (3,) or target_std.shape != (3,):
            raise ValueError("target statistics must each contain three values")
        if center_mean.shape != (1,) or center_std.shape != (1,):
            raise ValueError("center statistics must each contain one value")
        values = torch.cat((target_mean, target_std, center_mean, center_std))
        if not torch.isfinite(values).all():
            raise ValueError("frontier target statistics must be finite")
        if not (target_std > 0).all() or not (center_std > 0).all():
            raise ValueError("frontier target standard deviations must be positive")
        self.target_mean.copy_(target_mean)
        self.target_std.copy_(target_std)
        self.center_mean.copy_(center_mean)
        self.center_std.copy_(center_std)
        self.stats_configured.fill_(True)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if not bool(self.stats_configured):
            raise RuntimeError("FrontierCenterGapHead target statistics are unset")
        latent = self.network(embedding)
        center = latent[:, :1] * self.center_std + self.center_mean
        gap = latent[:, 1:] * self.target_std[2] + self.target_mean[2]
        values_eV = reconstruct_frontier_orbitals(gap, center)
        return (values_eV - self.target_mean) / self.target_std


def reconstruct_frontier_orbitals(
    gap: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    """Return ``[HOMO, LUMO, Gap]`` from aligned scalar gap and center tensors."""
    if gap.ndim != 2 or center.ndim != 2 or gap.shape != center.shape or gap.shape[1] != 1:
        raise ValueError("gap and center must have the same [rows, 1] shape")
    homo = center - 0.5 * gap
    lumo = center + 0.5 * gap
    return torch.cat((homo, lumo, gap), dim=1)
