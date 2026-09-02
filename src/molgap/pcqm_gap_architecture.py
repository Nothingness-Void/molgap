"""Official-PCQM categorical encoders for bounded Gap-only architecture screens."""
from __future__ import annotations

import torch
import torch.nn as nn

from .gps import (
    EdgeStateStructuralGPSWrapper,
    GraphTokenStructuralGPSWrapper,
    StructuralGPSWrapper,
)


class OGBStructuralGPSWrapper(StructuralGPSWrapper):
    """Structural GPS over the official OGB categorical graph representation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

        hidden_channels = self.node_emb.out_features
        self.node_emb = AtomEncoder(hidden_channels)
        self.edge_emb = BondEncoder(hidden_channels)

    def _embed_nodes(self, x):
        return self.node_emb(x.long())

    def _embed_edges(self, edge_attr):
        return self.edge_emb(edge_attr.long())


class OGBEdgeStateStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Persistent real-bond EdgeState GPS using official OGB categories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

        hidden_channels = self.node_emb.out_features
        self.node_emb = AtomEncoder(hidden_channels)
        self.edge_emb = BondEncoder(self.edge_state_channels)

    def _embed_nodes(self, x):
        return self.node_emb(x.long())

    def _embed_edges(self, edge_attr):
        return self.edge_emb(edge_attr.long())


class OGBGraphTokenStructuralGPSWrapper(GraphTokenStructuralGPSWrapper):
    """Persistent-edge GPS with an OGB-encoded recurrent molecule state.

    The shared graph state is updated from the node set and broadcast before
    every GPS block. Unlike learned-query pooling, it participates in all nine
    representation updates rather than changing only the final readout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

        hidden_channels = self.node_emb.out_features
        self.node_emb = AtomEncoder(hidden_channels)
        self.edge_emb = BondEncoder(self.edge_state_channels)

    def _embed_nodes(self, x):
        return self.node_emb(x.long())

    def _embed_edges(self, edge_attr):
        return self.edge_emb(edge_attr.long())


class OGBQueryPoolStructuralGPSWrapper(OGBStructuralGPSWrapper):
    """Structural GPS with learned graph queries instead of uniform pooling.

    The node encoder and all nine GPS blocks remain matched to Structural GPS.
    Four learned queries cross-attend to the final node set, allowing a scalar
    frontier-orbital target to emphasize distinct molecular regions without a
    persistent EdgeState, virtual node, 3D input, or prediction fusion.
    """

    def __init__(self, *args, num_pool_queries=4, **kwargs):
        super().__init__(*args, **kwargs)
        if num_pool_queries <= 0:
            raise ValueError("num_pool_queries must be positive")
        hidden_channels = self.head[0].in_features
        num_heads = kwargs.get("num_heads", 4)
        dropout = kwargs.get("dropout", 0.1)
        self.pool_queries = nn.Parameter(
            torch.empty(int(num_pool_queries), hidden_channels)
        )
        nn.init.normal_(self.pool_queries, std=hidden_channels ** -0.5)
        self.pool_attention = nn.MultiheadAttention(
            hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm1 = nn.LayerNorm(hidden_channels)
        self.pool_ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.pool_norm2 = nn.LayerNorm(hidden_channels)

    def _pool(self, h, batch):
        from torch_geometric.utils import to_dense_batch

        dense, mask = to_dense_batch(h, batch)
        queries = self.pool_queries.unsqueeze(0).expand(dense.shape[0], -1, -1)
        attended, _ = self.pool_attention(
            queries,
            dense,
            dense,
            key_padding_mask=~mask,
            need_weights=False,
        )
        queries = self.pool_norm1(queries + attended)
        queries = self.pool_norm2(queries + self.pool_ffn(queries))
        return queries.mean(dim=1)


class OGBLocalOperatorStructuralGPSWrapper(OGBStructuralGPSWrapper):
    """Structural GPS with one frozen edge-aware local operator family.

    The OGB encoders, RWSE16 input, global multi-head attention, depth, width,
    pooling, and scalar head remain fixed. Only the local message-passing
    operator inside each GPS block changes.
    """

    def __init__(self, *args, local_operator: str, **kwargs):
        super().__init__(*args, **kwargs)
        from torch_geometric.nn import (
            GATv2Conv,
            GENConv,
            GPSConv,
            ResGatedGraphConv,
            TransformerConv,
        )

        hidden_channels = self.head[0].in_features
        num_heads = int(kwargs.get("num_heads", 4))
        num_layers = int(kwargs.get("num_layers", 9))
        dropout = float(kwargs.get("dropout", 0.1))
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")

        def make_local():
            if local_operator == "resgated":
                return ResGatedGraphConv(
                    hidden_channels,
                    hidden_channels,
                    edge_dim=hidden_channels,
                )
            if local_operator == "transformer":
                return TransformerConv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    concat=True,
                    beta=True,
                    dropout=dropout,
                    edge_dim=hidden_channels,
                )
            if local_operator == "gen":
                return GENConv(
                    hidden_channels,
                    hidden_channels,
                    aggr="softmax",
                    learn_t=True,
                    msg_norm=True,
                    learn_msg_scale=True,
                    norm="layer",
                    num_layers=2,
                    edge_dim=hidden_channels,
                )
            if local_operator == "gatv2":
                return GATv2Conv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=hidden_channels,
                    add_self_loops=True,
                    fill_value="mean",
                )
            raise ValueError(f"Unknown local operator: {local_operator}")

        self.local_operator = local_operator
        self.convs = nn.ModuleList(
            GPSConv(
                channels=hidden_channels,
                conv=make_local(),
                heads=num_heads,
                dropout=dropout,
                act="silu",
                norm="batch_norm",
                attn_type="multihead",
            )
            for _ in range(num_layers)
        )


class _SparseWedgeStateUpdate(nn.Module):
    """Persistent low-rank state for adjacent directed bond pairs."""

    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        wedge_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        context_channels = hidden_channels + 2 * edge_channels
        self.context_norm = nn.LayerNorm(context_channels)
        self.context_proj = nn.Linear(context_channels, wedge_channels)
        self.state_norm = nn.LayerNorm(wedge_channels)
        self.update = nn.Sequential(
            nn.Linear(wedge_channels, wedge_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(wedge_channels * 2, wedge_channels),
        )
        self.output_norm = nn.LayerNorm(wedge_channels)

    def initial(self, context: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.context_proj(self.context_norm(context)))

    def forward(
        self,
        context: torch.Tensor,
        wedge_state: torch.Tensor,
    ) -> torch.Tensor:
        proposal = self.context_proj(self.context_norm(context))
        update_input = self.state_norm(wedge_state + proposal)
        return self.output_norm(wedge_state + self.update(update_input))


class OGBSparseTriangleEdgeStateGPSWrapper(OGBEdgeStateStructuralGPSWrapper):
    """Persistent EdgeState GPS with sparse adjacent-edge interactions.

    The additional state is attached only to directed non-backtracking wedges
    ``i -> j -> k``.  It is updated from the two current bond states and the
    center-node state, then sends sparse context back to both bond states and
    the center node before the normal GPS block.  No dense all-pairs tensor,
    coordinate input, target residual, or prediction fusion is used.
    """

    def __init__(
        self,
        *args,
        wedge_channels: int = 16,
        **kwargs,
    ) -> None:
        if wedge_channels <= 0:
            raise ValueError("wedge_channels must be positive")
        super().__init__(*args, **kwargs)
        # OGB's categorical AtomEncoder intentionally does not expose the
        # Linear-style ``out_features`` attribute.  The scalar head retains
        # the frozen encoder width after the superclass swaps node encoders.
        hidden_channels = self.head[0].in_features
        dropout = float(kwargs.get("dropout", 0.1))
        self.wedge_channels = int(wedge_channels)
        context_channels = hidden_channels + 2 * self.edge_state_channels
        self.wedge_initial = nn.Sequential(
            nn.LayerNorm(context_channels),
            nn.Linear(context_channels, self.wedge_channels),
            nn.LayerNorm(self.wedge_channels),
        )
        self.wedge_updates = nn.ModuleList(
            _SparseWedgeStateUpdate(
                hidden_channels,
                self.edge_state_channels,
                self.wedge_channels,
                dropout,
            )
            for _ in self.convs
        )
        self.wedge_to_edge = nn.ModuleList(
            nn.Linear(self.wedge_channels, self.edge_state_channels)
            for _ in self.convs
        )
        self.wedge_to_node = nn.ModuleList(
            nn.Linear(self.wedge_channels, hidden_channels)
            for _ in self.convs
        )
        for projection in (*self.wedge_to_edge, *self.wedge_to_node):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _validate_wedge_ids(
        wedge_edge_ids: torch.Tensor,
        edge_count: int,
    ) -> None:
        if wedge_edge_ids.ndim != 2 or wedge_edge_ids.shape[1] != 2:
            raise ValueError("wedge_edge_ids must have shape [W, 2]")
        if wedge_edge_ids.numel() and (
            int(wedge_edge_ids.min()) < 0
            or int(wedge_edge_ids.max()) >= edge_count
        ):
            raise ValueError("wedge edge id falls outside the batched edge set")

    @staticmethod
    def _aggregate_to_edges(
        wedge_state: torch.Tensor,
        wedge_edge_ids: torch.Tensor,
        edge_count: int,
    ) -> torch.Tensor:
        context = wedge_state.new_zeros((edge_count, wedge_state.shape[1]))
        counts = wedge_state.new_zeros((edge_count, 1))
        first, second = wedge_edge_ids.unbind(dim=1)
        context.index_add_(0, first, wedge_state)
        context.index_add_(0, second, wedge_state)
        ones = wedge_state.new_ones((wedge_state.shape[0], 1))
        counts.index_add_(0, first, ones)
        counts.index_add_(0, second, ones)
        return context / counts.clamp_min_(1.0)

    @staticmethod
    def _aggregate_to_centers(
        wedge_state: torch.Tensor,
        centers: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        context = wedge_state.new_zeros((node_count, wedge_state.shape[1]))
        counts = wedge_state.new_zeros((node_count, 1))
        context.index_add_(0, centers, wedge_state)
        counts.index_add_(
            0,
            centers,
            wedge_state.new_ones((wedge_state.shape[0], 1)),
        )
        return context / counts.clamp_min_(1.0)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
        )
        return self.head(embedding)

    def _post_atom_block(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        wedge_edge_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional information-flow hook after one atom GPS block."""
        return h, edge_state

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
    ):
        if random_walk_pe is None:
            raise ValueError("Sparse triangle GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(
                f"random_walk_pe must have shape {expected}, "
                f"got {tuple(random_walk_pe.shape)}"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")
        self._validate_wedge_ids(wedge_edge_ids, edge_index.shape[1])

        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        edge_state = self._embed_edges(edge_attr)
        first, second = wedge_edge_ids.unbind(dim=1)
        centers = edge_index[1, first]
        wedge_state = None
        if wedge_edge_ids.shape[0]:
            initial_context = torch.cat(
                [edge_state[first], edge_state[second], h[centers]], dim=-1
            )
            wedge_state = self.wedge_initial(initial_context)

        for edge_update, wedge_update, edge_projection, node_projection, conv in zip(
            self.edge_updates,
            self.wedge_updates,
            self.wedge_to_edge,
            self.wedge_to_node,
            self.convs,
        ):
            edge_state = edge_update(h, edge_index, edge_state)
            if wedge_state is not None:
                context = torch.cat(
                    [edge_state[first], edge_state[second], h[centers]],
                    dim=-1,
                )
                wedge_state = wedge_update(context, wedge_state)
                edge_context = self._aggregate_to_edges(
                    wedge_state,
                    wedge_edge_ids,
                    edge_state.shape[0],
                )
                edge_state = edge_state + edge_projection(edge_context)
                center_context = self._aggregate_to_centers(
                    wedge_state,
                    centers,
                    h.shape[0],
                )
                h = h + node_projection(center_context)
            h = conv(h, edge_index, batch, edge_attr=edge_state)
            h, edge_state = self._post_atom_block(
                layer,
                h,
                edge_index,
                edge_state,
                wedge_edge_ids,
            )
        return self._pool(h, batch)


class _SparseBondAttentionBlock(nn.Module):
    """Separately normalized attention over non-backtracking bond wedges."""

    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("bond channels must be divisible by attention heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_channels = self.channels // self.heads
        self.scale = self.head_channels ** -0.5
        self.attention_norm = nn.LayerNorm(self.channels)
        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_output = nn.Linear(self.channels, self.channels)
        self.ffn_norm = nn.LayerNorm(self.channels)
        expanded = self.channels * int(expansion)
        self.ffn_value = nn.Linear(self.channels, expanded)
        self.ffn_gate = nn.Linear(self.channels, expanded)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_output = nn.Linear(expanded, self.channels)
        for projection in (self.attention_output, self.ffn_output):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        edge_state: torch.Tensor,
        wedge_edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(edge_state)
        attended = edge_state.new_zeros(edge_state.shape)
        if wedge_edge_ids.shape[0]:
            from torch_geometric.utils import softmax

            source, target = wedge_edge_ids.unbind(dim=1)
            query = self.query(normalized[target]).view(
                -1, self.heads, self.head_channels
            )
            key = self.key(normalized[source]).view(
                -1, self.heads, self.head_channels
            )
            value = self.value(normalized[source]).view(
                -1, self.heads, self.head_channels
            )
            score = (query * key).sum(dim=-1) * self.scale
            weight = softmax(score, target, num_nodes=edge_state.shape[0])
            message = self.attention_dropout(weight).unsqueeze(-1) * value
            attended = attended.view(
                -1, self.heads, self.head_channels
            )
            attended.index_add_(0, target, message)
            attended = attended.reshape(-1, self.channels)
        edge_state = edge_state + self.attention_output(attended)
        normalized = self.ffn_norm(edge_state)
        proposal = self.ffn_value(normalized) * torch.sigmoid(
            self.ffn_gate(normalized)
        )
        return edge_state + self.ffn_output(self.ffn_dropout(proposal))


class _LowRankGatedProjection(nn.Module):
    """Rank-bounded gated residual whose value path starts at zero."""

    def __init__(self, in_channels: int, out_channels: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.down = nn.Linear(in_channels, rank)
        self.value = nn.Linear(rank, out_channels)
        self.gate = nn.Linear(rank, out_channels)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = torch.nn.functional.silu(self.down(self.norm(inputs)))
        return self.value(latent) * torch.sigmoid(self.gate(latent))


class _SharedAtomBondExchange(nn.Module):
    """One shared low-rank exchange between atom and directed-bond streams."""

    def __init__(
        self,
        atom_channels: int,
        bond_channels: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.atom_to_bond = _LowRankGatedProjection(
            atom_channels, bond_channels, rank
        )
        self.bond_to_atom = _LowRankGatedProjection(
            bond_channels, atom_channels, rank
        )

    @staticmethod
    def _incoming_bond_mean(
        edge_state: torch.Tensor,
        destinations: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        context = edge_state.new_zeros((node_count, edge_state.shape[1]))
        counts = edge_state.new_zeros((node_count, 1))
        context.index_add_(0, destinations, edge_state)
        counts.index_add_(
            0,
            destinations,
            edge_state.new_ones((edge_state.shape[0], 1)),
        )
        return context / counts.clamp_min_(1.0)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        endpoint_mean = 0.5 * (h[source] + h[destination])
        edge_state = edge_state + self.atom_to_bond(endpoint_mean)
        incoming = self._incoming_bond_mean(
            edge_state, destination, h.shape[0]
        )
        h = h + self.bond_to_atom(incoming)
        return h, edge_state


class _FixedGaussianBasis(nn.Module):
    """Small fixed radial basis for deterministic scalar geometry channels."""

    def __init__(self, minimum: float, maximum: float, channels: int) -> None:
        super().__init__()
        if channels < 2 or not maximum > minimum:
            raise ValueError("Gaussian basis bounds/channels are invalid")
        centers = torch.linspace(float(minimum), float(maximum), int(channels))
        spacing = float((maximum - minimum) / (channels - 1))
        self.register_buffer("centers", centers)
        self.gamma = 0.5 / (spacing * spacing)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("geometry scalar must have shape [N, 1]")
        return torch.exp(-self.gamma * (values - self.centers.view(1, -1)) ** 2)


class OGBGeometrySparseTriangleEdgeStateGPSWrapper(
    OGBSparseTriangleEdgeStateGPSWrapper
):
    """Bottom-fused ETKDG bond/angle geometry inside Sparse Triangle GPS.

    Geometry enters the persistent real-bond and topology-wedge states before
    every GPS block.  There is no independent 3D encoder, late prediction
    fusion, or target residual.  Zero-initialized projections make the initial
    function equal to the accepted pure-2D architecture while preserving full
    gradient flow into the new geometry channels.
    """

    MODES = {"distance", "angle", "distance_angle"}

    def __init__(
        self,
        *args,
        geometry_mode: str,
        geometry_basis_channels: int = 16,
        **kwargs,
    ) -> None:
        if geometry_mode not in self.MODES:
            raise ValueError(f"Unknown geometry mode: {geometry_mode}")
        super().__init__(*args, **kwargs)
        self.geometry_mode = geometry_mode
        self.distance_basis = _FixedGaussianBasis(0.75, 2.25, geometry_basis_channels)
        self.angle_basis = _FixedGaussianBasis(-1.0, 1.0, geometry_basis_channels)
        if "distance" in geometry_mode:
            self.distance_initial = nn.Linear(
                geometry_basis_channels, self.edge_state_channels, bias=False
            )
            self.distance_updates = nn.ModuleList(
                nn.Linear(
                    geometry_basis_channels, self.edge_state_channels, bias=False
                )
                for _ in self.convs
            )
            nn.init.zeros_(self.distance_initial.weight)
            for projection in self.distance_updates:
                nn.init.zeros_(projection.weight)
        if "angle" in geometry_mode:
            self.angle_initial = nn.Linear(
                geometry_basis_channels, self.wedge_channels, bias=False
            )
            self.angle_updates = nn.ModuleList(
                nn.Linear(geometry_basis_channels, self.wedge_channels, bias=False)
                for _ in self.convs
            )
            nn.init.zeros_(self.angle_initial.weight)
            for projection in self.angle_updates:
                nn.init.zeros_(projection.weight)

    @staticmethod
    def _geometry_mask(
        geometry_valid: torch.Tensor,
        batch: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        valid = geometry_valid.reshape(-1).float()
        if valid.shape[0] <= int(batch.max()):
            raise ValueError("geometry_valid does not cover the batched graphs")
        return valid[batch[node_ids]].view(-1, 1)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
    ):
        return self._encode_geometry(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
        )

    def _initialize_geometry_auxiliary(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_payload,
    ):
        """Create optional state for a subclass without changing the base."""
        return auxiliary_payload

    def _pre_message_node_injection(
        self,
        h: torch.Tensor,
        pos: torch.Tensor | None,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_distance: torch.Tensor,
        geometry_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Allow one stateless node feature injection before message passing."""
        return h

    def _update_geometry_auxiliary(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_state,
    ):
        """Update optional state after a complete atom block."""
        return h, edge_state, auxiliary_state

    def _encode_geometry(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        auxiliary_payload=None,
        pos=None,
    ):
        if random_walk_pe is None:
            raise ValueError("Geometry Triangle GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(f"random_walk_pe must have shape {expected}")
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")
        self._validate_wedge_ids(wedge_edge_ids, edge_index.shape[1])
        if tuple(edge_distance.shape) != (edge_index.shape[1], 1):
            raise ValueError("edge_distance is not aligned to directed bonds")
        if tuple(wedge_angle_cos.shape) != (wedge_edge_ids.shape[0], 1):
            raise ValueError("wedge_angle_cos is not aligned to wedges")
        if not torch.isfinite(edge_distance).all() or not torch.isfinite(
            wedge_angle_cos
        ).all():
            raise ValueError("geometry contains non-finite values")

        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        h = self._pre_message_node_injection(
            h,
            pos,
            edge_index,
            batch,
            edge_distance,
            geometry_valid,
        )
        edge_state = self._embed_edges(edge_attr)
        first, second = wedge_edge_ids.unbind(dim=1)
        centers = edge_index[1, first]

        distance_features = None
        if "distance" in self.geometry_mode:
            edge_mask = self._geometry_mask(
                geometry_valid, batch, edge_index[0]
            )
            distance_features = self.distance_basis(edge_distance.float()) * edge_mask
            edge_state = edge_state + self.distance_initial(distance_features)

        angle_features = None
        if "angle" in self.geometry_mode and wedge_edge_ids.shape[0]:
            wedge_mask = self._geometry_mask(geometry_valid, batch, centers)
            angle_features = self.angle_basis(wedge_angle_cos.float()) * wedge_mask

        wedge_state = None
        if wedge_edge_ids.shape[0]:
            initial_context = torch.cat(
                [edge_state[first], edge_state[second], h[centers]], dim=-1
            )
            wedge_state = self.wedge_initial(initial_context)
            if angle_features is not None:
                wedge_state = wedge_state + self.angle_initial(angle_features)

        auxiliary_state = self._initialize_geometry_auxiliary(
            h, batch, auxiliary_payload
        )

        for layer, (
            edge_update,
            wedge_update,
            edge_projection,
            node_projection,
            conv,
        ) in enumerate(
            zip(
                self.edge_updates,
                self.wedge_updates,
                self.wedge_to_edge,
                self.wedge_to_node,
                self.convs,
            )
        ):
            edge_state = edge_update(h, edge_index, edge_state)
            if distance_features is not None:
                edge_state = edge_state + self.distance_updates[layer](
                    distance_features
                )
            if wedge_state is not None:
                context = torch.cat(
                    [edge_state[first], edge_state[second], h[centers]], dim=-1
                )
                wedge_state = wedge_update(context, wedge_state)
                if angle_features is not None:
                    wedge_state = wedge_state + self.angle_updates[layer](
                        angle_features
                    )
                edge_context = self._aggregate_to_edges(
                    wedge_state, wedge_edge_ids, edge_state.shape[0]
                )
                edge_state = edge_state + edge_projection(edge_context)
                center_context = self._aggregate_to_centers(
                    wedge_state, centers, h.shape[0]
                )
                h = h + node_projection(center_context)
            h = conv(h, edge_index, batch, edge_attr=edge_state)
            h, edge_state = self._post_atom_block(
                layer,
                h,
                edge_index,
                edge_state,
                wedge_edge_ids,
            )
            h, edge_state, auxiliary_state = self._update_geometry_auxiliary(
                layer,
                h,
                edge_index,
                edge_state,
                batch,
                auxiliary_state,
            )
        return self._pool(h, batch)


class _ScheduledGPSBlock(nn.Module):
    """Reuse one GPS local branch with optional global atom attention.

    The wrapper receives an already initialized :class:`GPSConv`.  This keeps
    every shared local/MLP parameter identical under the same seed while
    removing the unused attention and normalization parameters from local-only
    blocks.
    """

    def __init__(self, base, use_global_attention: bool) -> None:
        super().__init__()
        self.channels = int(base.channels)
        self.heads = int(base.heads)
        self.dropout = float(base.dropout)
        self.attn_type = base.attn_type
        self.conv = base.conv
        self.attn = base.attn if use_global_attention else None
        self.mlp = base.mlp
        self.norm1 = base.norm1
        self.norm2 = base.norm2 if use_global_attention else None
        self.norm3 = base.norm3
        self.norm_with_batch = bool(base.norm_with_batch)
        self.use_global_attention = bool(use_global_attention)

    def _normalize(self, normalizer, value, batch):
        if normalizer is None:
            return value
        if self.norm_with_batch:
            return normalizer(value, batch=batch)
        return normalizer(value)

    def forward(self, x, edge_index, batch=None, **kwargs):
        import torch.nn.functional as functional

        local = self.conv(x, edge_index, **kwargs)
        local = functional.dropout(
            local, p=self.dropout, training=self.training
        )
        local = self._normalize(self.norm1, local + x, batch)
        branches = [local]

        if self.use_global_attention:
            from torch_geometric.utils import to_dense_batch

            dense, mask = to_dense_batch(x, batch)
            attended, _ = self.attn(
                dense,
                dense,
                dense,
                key_padding_mask=~mask,
                need_weights=False,
            )
            attended = attended[mask]
            attended = functional.dropout(
                attended, p=self.dropout, training=self.training
            )
            attended = self._normalize(self.norm2, attended + x, batch)
            branches.append(attended)

        output = sum(branches)
        output = output + self.mlp(output)
        return self._normalize(self.norm3, output, batch)


class _SharedGraphContext(nn.Module):
    """One compact molecule state updated and broadcast at fixed depths."""

    def __init__(
        self,
        atom_channels: int,
        graph_channels: int,
        exchange_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pooled_channels = 3 * atom_channels
        self.initial = nn.Sequential(
            nn.LayerNorm(pooled_channels),
            nn.Linear(pooled_channels, graph_channels),
            nn.LayerNorm(graph_channels),
        )
        combined_channels = 2 * graph_channels
        self.atom_summary = nn.Sequential(
            nn.LayerNorm(pooled_channels),
            nn.Linear(pooled_channels, graph_channels),
        )
        self.update_norm = nn.LayerNorm(combined_channels)
        self.update_gate = nn.Linear(combined_channels, graph_channels)
        self.update_value = nn.Sequential(
            nn.Linear(combined_channels, graph_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_channels, graph_channels),
        )
        self.output_norm = nn.LayerNorm(graph_channels)
        self.graph_to_atom = _LowRankGatedProjection(
            graph_channels, atom_channels, exchange_rank
        )

    @staticmethod
    def _pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        from torch_geometric.nn import (
            global_add_pool,
            global_max_pool,
            global_mean_pool,
        )

        return torch.cat(
            [
                global_mean_pool(h, batch),
                global_add_pool(h, batch),
                global_max_pool(h, batch),
            ],
            dim=-1,
        )

    def initialize(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.initial(self._pool(h, batch))

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        graph_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_summary = self.atom_summary(self._pool(h, batch))
        combined = self.update_norm(
            torch.cat([graph_state, atom_summary], dim=-1)
        )
        gate = torch.sigmoid(self.update_gate(combined))
        proposal = self.update_value(combined)
        graph_state = self.output_norm(graph_state + gate * proposal)
        return h + self.graph_to_atom(graph_state[batch]), graph_state


class OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper(
    OGBGeometrySparseTriangleEdgeStateGPSWrapper
):
    """Allocate global communication sparsely or through a graph state."""

    GLOBAL_MODES = {"sparse_attention", "graph_state"}
    GLOBAL_BLOCKS = (3, 6, 9)

    def __init__(
        self,
        *args,
        global_mode: str,
        graph_state_channels: int = 64,
        graph_exchange_rank: int = 32,
        **kwargs,
    ) -> None:
        if global_mode not in self.GLOBAL_MODES:
            raise ValueError(f"Unknown local/global mode: {global_mode}")
        super().__init__(*args, geometry_mode="distance_angle", **kwargs)
        self.global_mode = global_mode
        self.convs = nn.ModuleList(
            _ScheduledGPSBlock(
                conv,
                use_global_attention=(
                    global_mode == "sparse_attention"
                    and layer in self.GLOBAL_BLOCKS
                ),
            )
            for layer, conv in enumerate(self.convs, start=1)
        )
        if global_mode == "graph_state":
            hidden_channels = self.head[0].in_features
            dropout = float(kwargs.get("dropout", 0.1))
            self.graph_context = _SharedGraphContext(
                hidden_channels,
                graph_state_channels,
                graph_exchange_rank,
                dropout,
            )

    def _initialize_geometry_auxiliary(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_payload,
    ):
        if auxiliary_payload is not None:
            raise ValueError("Local/global screen does not accept auxiliary input")
        if self.global_mode == "graph_state":
            return self.graph_context.initialize(h, batch)
        return None

    def _update_geometry_auxiliary(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_state,
    ):
        block = layer + 1
        if self.global_mode == "graph_state" and block in self.GLOBAL_BLOCKS:
            h, auxiliary_state = self.graph_context(
                h, batch, auxiliary_state
            )
        return h, edge_state, auxiliary_state


class OGBBodyOrderMomentGraphStateGeometrySparseTriangleEdgeStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState geometry GPS with one stateless rotational-invariant moment."""

    BODY_ORDER_BASIS_CHANNELS = 16

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, global_mode="graph_state", **kwargs)
        self.body_order_basis = _FixedGaussianBasis(
            0.75, 2.25, self.BODY_ORDER_BASIS_CHANNELS
        )
        self.body_order_injection = nn.Sequential(
            nn.LayerNorm(48),
            nn.Linear(48, 64),
            nn.SiLU(),
            nn.Linear(64, 192, bias=False),
        )
        nn.init.zeros_(self.body_order_injection[-1].weight)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        pos,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            pos,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        pos,
    ):
        return self._encode_geometry(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            pos=pos,
        )

    def _pre_message_node_injection(
        self,
        h: torch.Tensor,
        pos: torch.Tensor | None,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_distance: torch.Tensor,
        geometry_valid: torch.Tensor,
    ) -> torch.Tensor:
        if pos is None:
            raise ValueError("Body-order moment GPS requires pos")
        if pos.ndim != 2 or tuple(pos.shape) != (h.shape[0], 3):
            raise ValueError("pos must align to batched atom coordinates")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if tuple(edge_distance.shape) != (edge_index.shape[1], 1):
            raise ValueError("edge_distance is not aligned to directed bonds")
        valid = geometry_valid.reshape(-1).to(device=h.device, dtype=h.dtype)
        if valid.numel() == 0 or batch.numel() == 0:
            raise ValueError("geometry_valid and batch must be non-empty")
        if int(batch.max()) >= valid.shape[0]:
            raise ValueError("geometry_valid does not cover the batched graphs")
        if not torch.isfinite(valid).all():
            raise ValueError("geometry_valid contains non-finite values")
        node_mask = valid[batch].view(-1, 1)
        finite_pos = torch.isfinite(pos).all(dim=1, keepdim=True)
        if bool(((node_mask > 0) & ~finite_pos).any()):
            raise ValueError("valid geometry contains non-finite positions")
        safe_pos = torch.where(finite_pos, pos, torch.zeros_like(pos))

        node_count = h.shape[0]
        source, destination = edge_index.unbind(dim=0)
        if destination.numel() == 0:
            invariants = h.new_zeros((node_count, 48))
        else:
            displacement = safe_pos[destination] - safe_pos[source]
            direction = displacement / displacement.norm(
                dim=1, keepdim=True
            ).clamp_min_(torch.finfo(displacement.dtype).eps)
            radial = self.body_order_basis(edge_distance.float()) * node_mask[
                destination
            ]

            scalar_density = h.new_zeros(
                (node_count, self.BODY_ORDER_BASIS_CHANNELS)
            )
            scalar_density.index_add_(0, destination, radial)

            vector_moment = h.new_zeros(
                (node_count, self.BODY_ORDER_BASIS_CHANNELS, 3)
            )
            vector_moment.index_add_(
                0, destination, radial.unsqueeze(-1) * direction.unsqueeze(1)
            )

            rank2_moment = h.new_zeros(
                (node_count, self.BODY_ORDER_BASIS_CHANNELS, 3, 3)
            )
            outer = direction.unsqueeze(-1) * direction.unsqueeze(-2)
            rank2_moment.index_add_(
                0,
                destination,
                radial.unsqueeze(-1).unsqueeze(-1) * outer.unsqueeze(1),
            )

            invariants = torch.cat(
                [
                    scalar_density,
                    vector_moment.square().sum(dim=-1),
                    rank2_moment.square().sum(dim=(-1, -2)),
                ],
                dim=-1,
            )
        return h + self.body_order_injection(invariants) * node_mask


class _SharedContactStateUpdate(nn.Module):
    """One recurrent through-space relation update shared across depth."""

    def __init__(
        self,
        atom_channels: int,
        contact_channels: int,
        exchange_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.endpoint_projection = nn.Linear(atom_channels, contact_channels)
        combined_channels = 3 * contact_channels
        self.update_norm = nn.LayerNorm(combined_channels)
        self.update_gate = nn.Linear(combined_channels, contact_channels)
        self.update_value = nn.Sequential(
            nn.Linear(combined_channels, contact_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(contact_channels, contact_channels),
        )
        self.output_norm = nn.LayerNorm(contact_channels)
        self.contact_to_atom = _LowRankGatedProjection(
            contact_channels, atom_channels, exchange_rank
        )

    def forward(
        self,
        h: torch.Tensor,
        contact_state: torch.Tensor,
        contact_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if contact_state.shape[0] == 0:
            return h, contact_state
        source, target = contact_edge_index.unbind(dim=0)
        combined = self.update_norm(
            torch.cat(
                [
                    contact_state,
                    self.endpoint_projection(h[source]),
                    self.endpoint_projection(h[target]),
                ],
                dim=-1,
            )
        )
        gate = torch.sigmoid(self.update_gate(combined))
        proposal = self.update_value(combined)
        contact_state = self.output_norm(contact_state + gate * proposal)

        incoming = contact_state.new_zeros((h.shape[0], contact_state.shape[1]))
        counts = contact_state.new_zeros((h.shape[0], 1))
        incoming.index_add_(0, target, contact_state)
        counts.index_add_(
            0, target, contact_state.new_ones((target.shape[0], 1))
        )
        incoming = incoming / counts.clamp_min_(1.0)
        return h + self.contact_to_atom(incoming), contact_state


class OGBContactStateGraphStateGeometrySparseTriangleEdgeStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState winner plus one narrow non-covalent ContactState."""

    CONTACT_LAYERS = (1, 3, 5, 7)

    def __init__(
        self,
        *args,
        contact_channels: int = 32,
        contact_basis_channels: int = 16,
        contact_exchange_rank: int = 16,
        **kwargs,
    ) -> None:
        if contact_channels <= 0 or contact_basis_channels < 2:
            raise ValueError("contact channels/basis are invalid")
        super().__init__(*args, global_mode="graph_state", **kwargs)
        atom_channels = self.head[0].in_features
        dropout = float(kwargs.get("dropout", 0.1))
        self.contact_channels = int(contact_channels)
        self.contact_basis_channels = int(contact_basis_channels)
        self.contact_distance_basis = _FixedGaussianBasis(
            0.25, 5.0, self.contact_basis_channels
        )
        initial_channels = 2 * atom_channels + self.contact_basis_channels
        self.contact_initial = nn.Sequential(
            nn.LayerNorm(initial_channels),
            nn.Linear(initial_channels, self.contact_channels),
            nn.LayerNorm(self.contact_channels),
        )
        self.contact_update = _SharedContactStateUpdate(
            atom_channels,
            self.contact_channels,
            contact_exchange_rank,
            dropout,
        )

    @staticmethod
    def _validate_contact_payload(
        contact_edge_index: torch.Tensor,
        contact_distance: torch.Tensor,
        node_count: int,
    ) -> None:
        if contact_edge_index.ndim != 2 or contact_edge_index.shape[0] != 2:
            raise ValueError("contact_edge_index must have shape [2, C]")
        if tuple(contact_distance.shape) != (contact_edge_index.shape[1], 1):
            raise ValueError("contact distances are not aligned")
        if contact_edge_index.shape[1] and (
            int(contact_edge_index.min()) < 0
            or int(contact_edge_index.max()) >= node_count
        ):
            raise ValueError("contact relation has an invalid atom id")
        if not torch.isfinite(contact_distance).all() or (
            contact_distance.numel()
            and (
                bool((contact_distance <= 0).any())
                or bool((contact_distance > 5.0).any())
            )
        ):
            raise ValueError("contact distance is outside (0, 5.0]")

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        contact_edge_index,
        contact_distance,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            contact_edge_index,
            contact_distance,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        contact_edge_index,
        contact_distance,
    ):
        self._validate_contact_payload(
            contact_edge_index, contact_distance, x.shape[0]
        )
        return self._encode_geometry(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            auxiliary_payload={
                "contact_edge_index": contact_edge_index,
                "contact_distance": contact_distance,
            },
        )

    def _initialize_geometry_auxiliary(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_payload,
    ):
        if auxiliary_payload is None:
            raise ValueError("ContactState payload is required")
        contact_edge_index = auxiliary_payload["contact_edge_index"]
        contact_distance = auxiliary_payload["contact_distance"]
        if contact_edge_index.shape[1]:
            source, target = contact_edge_index.unbind(dim=0)
            distance_features = self.contact_distance_basis(
                contact_distance.float()
            )
            contact_state = self.contact_initial(
                torch.cat([h[source], h[target], distance_features], dim=-1)
            )
        else:
            contact_state = h.new_empty((0, self.contact_channels))
        return {
            **auxiliary_payload,
            "contact_state": contact_state,
            "graph_state": self.graph_context.initialize(h, batch),
        }

    def _update_geometry_auxiliary(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_state,
    ):
        if layer in self.CONTACT_LAYERS:
            h, contact_state = self.contact_update(
                h,
                auxiliary_state["contact_state"],
                auxiliary_state["contact_edge_index"],
            )
            auxiliary_state["contact_state"] = contact_state
        block = layer + 1
        if block in self.GLOBAL_BLOCKS:
            h, graph_state = self.graph_context(
                h, batch, auxiliary_state["graph_state"]
            )
            auxiliary_state["graph_state"] = graph_state
        return h, edge_state, auxiliary_state


class _SharedRingHierarchyUpdate(nn.Module):
    """One narrow recurrent atom--ring--ring update shared across depth."""

    def __init__(
        self,
        atom_channels: int,
        ring_channels: int,
        ring_edge_channels: int,
        exchange_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ring_channels = int(ring_channels)
        self.atom_projection = nn.Linear(atom_channels, ring_channels)
        self.edge_projection = nn.Linear(ring_edge_channels, ring_channels)
        combined_channels = 3 * ring_channels
        self.update_norm = nn.LayerNorm(combined_channels)
        self.update_gate = nn.Linear(combined_channels, ring_channels)
        self.update_value = nn.Sequential(
            nn.Linear(combined_channels, ring_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ring_channels, ring_channels),
        )
        self.output_norm = nn.LayerNorm(ring_channels)
        self.ring_to_atom = _LowRankGatedProjection(
            ring_channels, atom_channels, exchange_rank
        )

    @staticmethod
    def _membership_mean(
        source: torch.Tensor,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        result = source.new_zeros((target_count, source.shape[1]))
        counts = source.new_zeros((target_count, 1))
        result.index_add_(0, target_ids, source[source_ids])
        counts.index_add_(
            0, target_ids, source.new_ones((source_ids.shape[0], 1))
        )
        return result / counts.clamp_min_(1.0)

    def forward(
        self,
        h: torch.Tensor,
        ring_state: torch.Tensor,
        atom_ring_index: torch.Tensor,
        ring_edge_index: torch.Tensor,
        ring_edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ring_state.shape[0] == 0:
            return h, ring_state
        atom_ids, ring_ids = atom_ring_index.unbind(dim=0)
        atom_context = self._membership_mean(
            h, atom_ids, ring_ids, ring_state.shape[0]
        )
        atom_context = self.atom_projection(atom_context)

        neighbor_context = ring_state.new_zeros(ring_state.shape)
        if ring_edge_index.shape[1]:
            source, target = ring_edge_index.unbind(dim=0)
            messages = ring_state[source] + self.edge_projection(
                ring_edge_attr.float()
            )
            counts = ring_state.new_zeros((ring_state.shape[0], 1))
            neighbor_context.index_add_(0, target, messages)
            counts.index_add_(
                0, target, ring_state.new_ones((target.shape[0], 1))
            )
            neighbor_context = neighbor_context / counts.clamp_min_(1.0)

        combined = self.update_norm(
            torch.cat([ring_state, atom_context, neighbor_context], dim=-1)
        )
        gate = torch.sigmoid(self.update_gate(combined))
        proposal = self.update_value(combined)
        ring_state = self.output_norm(ring_state + gate * proposal)

        atom_ring_context = self._membership_mean(
            ring_state, ring_ids, atom_ids, h.shape[0]
        )
        return h + self.ring_to_atom(atom_ring_context), ring_state


class OGBRingHierarchyGeometrySparseTriangleEdgeStateGPSWrapper(
    OGBGeometrySparseTriangleEdgeStateGPSWrapper
):
    """Distance/angle GPS with persistent deterministic smallest-ring state."""

    HIERARCHY_LAYERS = (1, 3, 5, 7)

    def __init__(
        self,
        *args,
        ring_channels: int = 64,
        ring_feature_channels: int = 12,
        ring_edge_channels: int = 4,
        exchange_rank: int = 32,
        **kwargs,
    ) -> None:
        if ring_channels <= 0 or ring_feature_channels <= 0:
            raise ValueError("ring channels must be positive")
        super().__init__(*args, geometry_mode="distance_angle", **kwargs)
        hidden_channels = self.head[0].in_features
        dropout = float(kwargs.get("dropout", 0.1))
        self.ring_channels = int(ring_channels)
        self.ring_feature_channels = int(ring_feature_channels)
        self.ring_edge_channels = int(ring_edge_channels)
        self.ring_feature_encoder = nn.Sequential(
            nn.LayerNorm(self.ring_feature_channels),
            nn.Linear(self.ring_feature_channels, self.ring_channels),
            nn.LayerNorm(self.ring_channels),
        )
        self.ring_update = _SharedRingHierarchyUpdate(
            hidden_channels,
            self.ring_channels,
            self.ring_edge_channels,
            exchange_rank,
            dropout,
        )
        self.ring_initial_norm = nn.LayerNorm(self.ring_channels)

    @staticmethod
    def _validate_ring_payload(
        ring_features: torch.Tensor,
        atom_ring_index: torch.Tensor,
        ring_edge_index: torch.Tensor,
        ring_edge_attr: torch.Tensor,
        node_count: int,
        feature_channels: int,
        edge_channels: int,
    ) -> None:
        ring_count = int(ring_features.shape[0])
        if tuple(ring_features.shape) != (ring_count, feature_channels):
            raise ValueError("ring_features has the wrong shape")
        if atom_ring_index.ndim != 2 or atom_ring_index.shape[0] != 2:
            raise ValueError("atom_ring_index must have shape [2, M]")
        if ring_edge_index.ndim != 2 or ring_edge_index.shape[0] != 2:
            raise ValueError("ring_edge_index must have shape [2, R]")
        if tuple(ring_edge_attr.shape) != (
            ring_edge_index.shape[1],
            edge_channels,
        ):
            raise ValueError("ring edge attributes are not aligned")
        if not torch.isfinite(ring_features).all() or not torch.isfinite(
            ring_edge_attr
        ).all():
            raise ValueError("ring payload contains non-finite values")
        if atom_ring_index.shape[1]:
            atom_ids, ring_ids = atom_ring_index.unbind(dim=0)
            if int(atom_ids.min()) < 0 or int(atom_ids.max()) >= node_count:
                raise ValueError("atom-ring membership has an invalid atom id")
            if int(ring_ids.min()) < 0 or int(ring_ids.max()) >= ring_count:
                raise ValueError("atom-ring membership has an invalid ring id")
            counts = torch.bincount(ring_ids, minlength=ring_count)
            if counts.shape[0] != ring_count or torch.any(counts == 0):
                raise ValueError("every ring must have at least one member atom")
        elif ring_count:
            raise ValueError("ring features exist without atom membership")
        if ring_edge_index.shape[1] and (
            int(ring_edge_index.min()) < 0
            or int(ring_edge_index.max()) >= ring_count
        ):
            raise ValueError("ring relation has an invalid ring id")

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        ring_features,
        atom_ring_index,
        ring_edge_index,
        ring_edge_attr,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            ring_features,
            atom_ring_index,
            ring_edge_index,
            ring_edge_attr,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        ring_features,
        atom_ring_index,
        ring_edge_index,
        ring_edge_attr,
    ):
        self._validate_ring_payload(
            ring_features,
            atom_ring_index,
            ring_edge_index,
            ring_edge_attr,
            x.shape[0],
            self.ring_feature_channels,
            self.ring_edge_channels,
        )
        return self._encode_geometry(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            auxiliary_payload={
                "ring_features": ring_features,
                "atom_ring_index": atom_ring_index,
                "ring_edge_index": ring_edge_index,
                "ring_edge_attr": ring_edge_attr,
            },
        )

    def _initialize_geometry_auxiliary(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_payload,
    ):
        if auxiliary_payload is None:
            raise ValueError("ring hierarchy payload is required")
        ring_features = auxiliary_payload["ring_features"]
        atom_ring_index = auxiliary_payload["atom_ring_index"]
        ring_state = self.ring_feature_encoder(ring_features.float())
        if ring_state.shape[0]:
            atom_ids, ring_ids = atom_ring_index.unbind(dim=0)
            atom_context = self.ring_update._membership_mean(
                h, atom_ids, ring_ids, ring_state.shape[0]
            )
            ring_state = self.ring_initial_norm(
                ring_state + self.ring_update.atom_projection(atom_context)
            )
        return {**auxiliary_payload, "ring_state": ring_state}

    def _update_geometry_auxiliary(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_state,
    ):
        if layer not in self.HIERARCHY_LAYERS:
            return h, edge_state, auxiliary_state
        h, ring_state = self.ring_update(
            h,
            auxiliary_state["ring_state"],
            auxiliary_state["atom_ring_index"],
            auxiliary_state["ring_edge_index"],
            auxiliary_state["ring_edge_attr"],
        )
        auxiliary_state["ring_state"] = ring_state
        return h, edge_state, auxiliary_state


class OGBRingHierarchyGraphStateGeometrySparseTriangleEdgeStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState winner with one persistent deterministic ring hierarchy."""

    HIERARCHY_LAYERS = (1, 3, 5, 7)

    def __init__(
        self,
        *args,
        ring_channels: int = 64,
        ring_feature_channels: int = 12,
        ring_edge_channels: int = 4,
        ring_exchange_rank: int = 32,
        **kwargs,
    ) -> None:
        if ring_channels <= 0 or ring_feature_channels <= 0:
            raise ValueError("ring channels must be positive")
        super().__init__(*args, global_mode="graph_state", **kwargs)
        hidden_channels = self.head[0].in_features
        dropout = float(kwargs.get("dropout", 0.1))
        self.ring_channels = int(ring_channels)
        self.ring_feature_channels = int(ring_feature_channels)
        self.ring_edge_channels = int(ring_edge_channels)
        self.ring_feature_encoder = nn.Sequential(
            nn.LayerNorm(self.ring_feature_channels),
            nn.Linear(self.ring_feature_channels, self.ring_channels),
            nn.LayerNorm(self.ring_channels),
        )
        self.ring_update = _SharedRingHierarchyUpdate(
            hidden_channels,
            self.ring_channels,
            self.ring_edge_channels,
            ring_exchange_rank,
            dropout,
        )
        self.ring_initial_norm = nn.LayerNorm(self.ring_channels)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        ring_features,
        atom_ring_index,
        ring_edge_index,
        ring_edge_attr,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            ring_features,
            atom_ring_index,
            ring_edge_index,
            ring_edge_attr,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        ring_features,
        atom_ring_index,
        ring_edge_index,
        ring_edge_attr,
    ):
        OGBRingHierarchyGeometrySparseTriangleEdgeStateGPSWrapper._validate_ring_payload(
            ring_features,
            atom_ring_index,
            ring_edge_index,
            ring_edge_attr,
            x.shape[0],
            self.ring_feature_channels,
            self.ring_edge_channels,
        )
        return self._encode_geometry(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            auxiliary_payload={
                "ring_features": ring_features,
                "atom_ring_index": atom_ring_index,
                "ring_edge_index": ring_edge_index,
                "ring_edge_attr": ring_edge_attr,
            },
        )

    def _initialize_geometry_auxiliary(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_payload,
    ):
        if auxiliary_payload is None:
            raise ValueError("ring hierarchy payload is required")
        ring_features = auxiliary_payload["ring_features"]
        atom_ring_index = auxiliary_payload["atom_ring_index"]
        ring_state = self.ring_feature_encoder(ring_features.float())
        if ring_state.shape[0]:
            atom_ids, ring_ids = atom_ring_index.unbind(dim=0)
            atom_context = self.ring_update._membership_mean(
                h, atom_ids, ring_ids, ring_state.shape[0]
            )
            ring_state = self.ring_initial_norm(
                ring_state + self.ring_update.atom_projection(atom_context)
            )
        return {
            **auxiliary_payload,
            "ring_state": ring_state,
            "graph_state": self.graph_context.initialize(h, batch),
        }

    def _update_geometry_auxiliary(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_state,
    ):
        if layer in self.HIERARCHY_LAYERS:
            h, ring_state = self.ring_update(
                h,
                auxiliary_state["ring_state"],
                auxiliary_state["atom_ring_index"],
                auxiliary_state["ring_edge_index"],
                auxiliary_state["ring_edge_attr"],
            )
            auxiliary_state["ring_state"] = ring_state
        block = layer + 1
        if block in self.GLOBAL_BLOCKS:
            h, graph_state = self.graph_context(
                h, batch, auxiliary_state["graph_state"]
            )
            auxiliary_state["graph_state"] = graph_state
        return h, edge_state, auxiliary_state


class OGBDualStreamGeometrySparseTriangleEdgeStateGPSWrapper(
    OGBGeometrySparseTriangleEdgeStateGPSWrapper
):
    """Distance/angle GPS with a sparse, separately normalized bond stream."""

    DUAL_STREAM_LAYERS = (1, 3, 5, 7)

    def __init__(
        self,
        *args,
        bond_attention_heads: int = 4,
        exchange_rank: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, geometry_mode="distance_angle", **kwargs)
        hidden_channels = self.head[0].in_features
        dropout = float(kwargs.get("dropout", 0.1))
        self.bond_stream_blocks = nn.ModuleDict(
            {
                str(layer): _SparseBondAttentionBlock(
                    self.edge_state_channels,
                    bond_attention_heads,
                    dropout,
                )
                for layer in self.DUAL_STREAM_LAYERS
            }
        )
        self.atom_bond_exchange = _SharedAtomBondExchange(
            hidden_channels,
            self.edge_state_channels,
            exchange_rank,
        )

    def _post_atom_block(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        wedge_edge_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(layer)
        if key not in self.bond_stream_blocks:
            return h, edge_state
        edge_state = self.bond_stream_blocks[key](
            edge_state, wedge_edge_ids
        )
        return self.atom_bond_exchange(h, edge_index, edge_state)


class _SharedTorsionGatedUpdate(nn.Module):
    """One gated torsion-state update cell reused at every GPS block."""

    def __init__(
        self,
        context_channels: int,
        torsion_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(context_channels)
        self.context_proj = nn.Linear(context_channels, torsion_channels)
        self.state_norm = nn.LayerNorm(torsion_channels)
        combined_channels = torsion_channels * 2
        self.gate = nn.Linear(combined_channels, torsion_channels)
        self.proposal = nn.Sequential(
            nn.Linear(combined_channels, torsion_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(torsion_channels, torsion_channels),
        )
        self.output_norm = nn.LayerNorm(torsion_channels)

    def forward(
        self,
        context: torch.Tensor,
        torsion_state: torch.Tensor,
    ) -> torch.Tensor:
        context_state = self.context_proj(self.context_norm(context))
        combined = torch.cat(
            [context_state, self.state_norm(torsion_state)], dim=-1
        )
        gate = torch.sigmoid(self.gate(combined))
        proposal = self.proposal(combined)
        return self.output_norm(torsion_state + gate * proposal)


class OGBTorsionGeometrySparseTriangleEdgeStateGPSWrapper(
    OGBGeometrySparseTriangleEdgeStateGPSWrapper
):
    """Distance/angle Sparse Triangle GPS with one sparse torsion state.

    Torsions are attached only to cached non-backtracking bonded paths.  Their
    zero-initialized projections cannot perturb the accepted distance-plus-
    angle function at initialization; after optimization they exchange
    context only with their three bonds and two adjacent wedges.
    """

    TORSION_FEATURE_CHANNELS = 4

    def __init__(
        self,
        *args,
        torsion_channels: int = 16,
        **kwargs,
    ) -> None:
        if torsion_channels <= 0:
            raise ValueError("torsion_channels must be positive")
        super().__init__(*args, geometry_mode="distance_angle", **kwargs)
        self.torsion_channels = int(torsion_channels)
        context_channels = (
            3 * self.edge_state_channels
            + 2 * self.wedge_channels
            + self.TORSION_FEATURE_CHANNELS
        )
        self.torsion_initial = nn.Sequential(
            nn.LayerNorm(context_channels),
            nn.Linear(context_channels, self.torsion_channels),
            nn.LayerNorm(self.torsion_channels),
        )
        self.torsion_update = _SharedTorsionGatedUpdate(
            context_channels,
            self.torsion_channels,
            float(kwargs.get("dropout", 0.1)),
        )
        self.torsion_to_edge = nn.Linear(
            self.torsion_channels, self.edge_state_channels
        )
        self.torsion_to_wedge = nn.Linear(
            self.torsion_channels, self.wedge_channels
        )
        for projection in (self.torsion_to_edge, self.torsion_to_wedge):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _validate_torsion_inputs(
        torsion_edge_ids: torch.Tensor,
        torsion_wedge_ids: torch.Tensor,
        torsion_fourier: torch.Tensor,
        torsion_valid: torch.Tensor,
        edge_count: int,
        wedge_count: int,
    ) -> None:
        torsion_count = int(torsion_edge_ids.shape[0])
        if torsion_edge_ids.ndim != 2 or torsion_edge_ids.shape[1] != 3:
            raise ValueError("torsion_edge_ids must have shape [T, 3]")
        if torsion_wedge_ids.ndim != 2 or torsion_wedge_ids.shape[1] != 2:
            raise ValueError("torsion_wedge_ids must have shape [T, 2]")
        if torsion_wedge_ids.shape[0] != torsion_count:
            raise ValueError("torsion wedge ids are not aligned to torsions")
        if tuple(torsion_fourier.shape) != (
            torsion_count,
            OGBTorsionGeometrySparseTriangleEdgeStateGPSWrapper.TORSION_FEATURE_CHANNELS,
        ):
            raise ValueError("torsion_fourier is not aligned to torsion paths")
        if tuple(torsion_valid.shape) != (torsion_count, 1):
            raise ValueError("torsion_valid is not aligned to torsion paths")
        if torsion_count and (
            int(torsion_edge_ids.min()) < 0
            or int(torsion_edge_ids.max()) >= edge_count
        ):
            raise ValueError("torsion edge id falls outside the batched edge set")
        if torsion_count and (
            int(torsion_wedge_ids.min()) < 0
            or int(torsion_wedge_ids.max()) >= wedge_count
        ):
            raise ValueError("torsion wedge id falls outside the batched wedge set")
        if not torch.isfinite(torsion_fourier).all() or not torch.isfinite(
            torsion_valid
        ).all():
            raise ValueError("torsion payload contains non-finite values")
        if torch.any(torsion_valid < 0) or torch.any(torsion_valid > 1):
            raise ValueError("torsion_valid must lie in [0, 1]")

    @staticmethod
    def _aggregate_to_indices(
        state: torch.Tensor,
        indices: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        context = state.new_zeros((count, state.shape[1]))
        occurrences = state.new_zeros((count, 1))
        for column in range(indices.shape[1]):
            selected = indices[:, column]
            context.index_add_(0, selected, state)
            occurrences.index_add_(
                0, selected, state.new_ones((state.shape[0], 1))
            )
        return context / occurrences.clamp_min_(1.0)

    def _torsion_context(
        self,
        edge_state: torch.Tensor,
        wedge_state: torch.Tensor,
        torsion_edge_ids: torch.Tensor,
        torsion_wedge_ids: torch.Tensor,
        torsion_features: torch.Tensor,
    ) -> torch.Tensor:
        parts = [
            edge_state[torsion_edge_ids[:, column]]
            for column in range(3)
        ]
        parts.extend(
            wedge_state[torsion_wedge_ids[:, column]] for column in range(2)
        )
        parts.append(torsion_features)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        torsion_edge_ids,
        torsion_wedge_ids,
        torsion_fourier,
        torsion_valid,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            torsion_edge_ids,
            torsion_wedge_ids,
            torsion_fourier,
            torsion_valid,
        )
        return self.head(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        torsion_edge_ids,
        torsion_wedge_ids,
        torsion_fourier,
        torsion_valid,
    ):
        if random_walk_pe is None:
            raise ValueError("Torsion geometry GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(f"random_walk_pe must have shape {expected}")
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")
        self._validate_wedge_ids(wedge_edge_ids, edge_index.shape[1])
        if tuple(edge_distance.shape) != (edge_index.shape[1], 1):
            raise ValueError("edge_distance is not aligned to directed bonds")
        if tuple(wedge_angle_cos.shape) != (wedge_edge_ids.shape[0], 1):
            raise ValueError("wedge_angle_cos is not aligned to wedges")
        if not torch.isfinite(edge_distance).all() or not torch.isfinite(
            wedge_angle_cos
        ).all():
            raise ValueError("geometry contains non-finite values")
        self._validate_torsion_inputs(
            torsion_edge_ids,
            torsion_wedge_ids,
            torsion_fourier,
            torsion_valid,
            edge_index.shape[1],
            wedge_edge_ids.shape[0],
        )

        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        edge_state = self._embed_edges(edge_attr)
        first, second = wedge_edge_ids.unbind(dim=1)
        centers = edge_index[1, first]

        edge_mask = self._geometry_mask(geometry_valid, batch, edge_index[0])
        distance_features = self.distance_basis(edge_distance.float()) * edge_mask
        edge_state = edge_state + self.distance_initial(distance_features)

        angle_features = None
        if wedge_edge_ids.shape[0]:
            wedge_mask = self._geometry_mask(geometry_valid, batch, centers)
            angle_features = self.angle_basis(wedge_angle_cos.float()) * wedge_mask

        wedge_state = None
        if wedge_edge_ids.shape[0]:
            initial_context = torch.cat(
                [edge_state[first], edge_state[second], h[centers]], dim=-1
            )
            wedge_state = self.wedge_initial(initial_context)
            if angle_features is not None:
                wedge_state = wedge_state + self.angle_initial(angle_features)

        torsion_state = None
        torsion_mask = None
        torsion_features = None
        if torsion_edge_ids.shape[0]:
            if wedge_state is None:
                raise ValueError("torsion paths require cached wedges")
            torsion_mask = torsion_valid.float()
            torsion_features = torsion_fourier.float() * torsion_mask
            initial_torsion_context = self._torsion_context(
                edge_state,
                wedge_state,
                torsion_edge_ids,
                torsion_wedge_ids,
                torsion_features,
            )
            torsion_state = self.torsion_initial(initial_torsion_context)
            torsion_state = torsion_state * torsion_mask

        for layer, (
            edge_update,
            wedge_update,
            edge_projection,
            node_projection,
            conv,
        ) in enumerate(
            zip(
                self.edge_updates,
                self.wedge_updates,
                self.wedge_to_edge,
                self.wedge_to_node,
                self.convs,
            )
        ):
            edge_state = edge_update(h, edge_index, edge_state)
            edge_state = edge_state + self.distance_updates[layer](
                distance_features
            )
            if wedge_state is not None:
                context = torch.cat(
                    [edge_state[first], edge_state[second], h[centers]], dim=-1
                )
                wedge_state = wedge_update(context, wedge_state)
                if angle_features is not None:
                    wedge_state = wedge_state + self.angle_updates[layer](
                        angle_features
                    )
                if torsion_state is not None:
                    torsion_context = self._torsion_context(
                        edge_state,
                        wedge_state,
                        torsion_edge_ids,
                        torsion_wedge_ids,
                        torsion_features,
                    )
                    torsion_state = self.torsion_update(
                        torsion_context, torsion_state
                    ) * torsion_mask
                    edge_context = self._aggregate_to_indices(
                        torsion_state, torsion_edge_ids, edge_state.shape[0]
                    )
                    wedge_context = self._aggregate_to_indices(
                        torsion_state, torsion_wedge_ids, wedge_state.shape[0]
                    )
                    edge_state = edge_state + self.torsion_to_edge(edge_context)
                    wedge_state = wedge_state + self.torsion_to_wedge(
                        wedge_context
                    )
                edge_context = self._aggregate_to_edges(
                    wedge_state, wedge_edge_ids, edge_state.shape[0]
                )
                edge_state = edge_state + edge_projection(edge_context)
                center_context = self._aggregate_to_centers(
                    wedge_state, centers, h.shape[0]
                )
                h = h + node_projection(center_context)
            h = conv(h, edge_index, batch, edge_attr=edge_state)
        return self._pool(h, batch)


def make_pcqm_gap_encoder(candidate: str):
    """Build one frozen first-round candidate with a scalar Gap head."""
    common = {
        "in_channels": 9,
        "edge_dim": 3,
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.1,
        "n_targets": 1,
        "pooling": "mean",
        "rwse_dim": 16,
    }
    if candidate == "ogb_structural_gps9":
        return OGBStructuralGPSWrapper(**common)
    if candidate == "ogb_edge_state_structural_gps9":
        return OGBEdgeStateStructuralGPSWrapper(
            **common,
            edge_state_channels=64,
        )
    if candidate == "ogb_sparse_triangle_edge_state_gps9":
        return OGBSparseTriangleEdgeStateGPSWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
        )
    geometry_modes = {
        "ogb_distance_triangle_edge_state_gps9": "distance",
        "ogb_angle_triangle_edge_state_gps9": "angle",
        "ogb_distance_angle_triangle_edge_state_gps9": "distance_angle",
    }
    if candidate in geometry_modes:
        return OGBGeometrySparseTriangleEdgeStateGPSWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
            geometry_mode=geometry_modes[candidate],
            geometry_basis_channels=16,
        )
    if candidate == "ogb_distance_angle_ring_hierarchy_triangle_edge_state_gps9":
        return OGBRingHierarchyGeometrySparseTriangleEdgeStateGPSWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
            geometry_basis_channels=16,
            ring_channels=64,
            ring_feature_channels=12,
            ring_edge_channels=4,
            exchange_rank=32,
        )
    if candidate == (
        "ogb_distance_angle_ring_hierarchy_triangle_edge_state_graph_state9"
    ):
        return OGBRingHierarchyGraphStateGeometrySparseTriangleEdgeStateWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
            geometry_basis_channels=16,
            graph_state_channels=64,
            graph_exchange_rank=32,
            ring_channels=64,
            ring_feature_channels=12,
            ring_edge_channels=4,
            ring_exchange_rank=32,
        )
    if candidate == (
        "ogb_distance_angle_contact_state_triangle_edge_state_graph_state9"
    ):
        return OGBContactStateGraphStateGeometrySparseTriangleEdgeStateWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
            geometry_basis_channels=16,
            graph_state_channels=64,
            graph_exchange_rank=32,
            contact_channels=32,
            contact_basis_channels=16,
            contact_exchange_rank=16,
        )
    if candidate == (
        "ogb_distance_angle_body_order_triangle_edge_state_graph_state9"
    ):
        return OGBBodyOrderMomentGraphStateGeometrySparseTriangleEdgeStateWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
            geometry_basis_channels=16,
            graph_state_channels=64,
            graph_exchange_rank=32,
        )
    local_global_modes = {
        "ogb_distance_angle_triangle_edge_state_sparse_gps369": "sparse_attention",
        "ogb_distance_angle_triangle_edge_state_graph_state9": "graph_state",
    }
    if candidate in local_global_modes:
        return OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
            geometry_basis_channels=16,
            global_mode=local_global_modes[candidate],
            graph_state_channels=64,
            graph_exchange_rank=32,
        )
    raise ValueError(f"Unknown PCQM Gap candidate: {candidate}")
