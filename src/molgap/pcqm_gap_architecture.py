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
        return self._pool(h, batch)


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
        return self._pool(h, batch)


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
    if candidate == "ogb_recurrent_graph_state_gps9":
        return OGBGraphTokenStructuralGPSWrapper(
            **common,
            edge_state_channels=64,
            token_channels=16,
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
    if candidate == "ogb_distance_angle_torsion_triangle_edge_state_gps9":
        return OGBTorsionGeometrySparseTriangleEdgeStateGPSWrapper(
            **common,
            edge_state_channels=64,
            wedge_channels=16,
            torsion_channels=16,
            geometry_basis_channels=16,
        )
    if candidate == "ogb_query_pool_structural_gps9":
        return OGBQueryPoolStructuralGPSWrapper(
            **common,
            num_pool_queries=4,
        )
    local_operators = {
        "ogb_gated_local_gps9": "resgated",
        "ogb_edge_attention_local_gps9": "transformer",
        "ogb_gen_local_gps9": "gen",
        "ogb_gatv2_local_gps9": "gatv2",
    }
    if candidate in local_operators:
        return OGBLocalOperatorStructuralGPSWrapper(
            **common,
            local_operator=local_operators[candidate],
        )
    raise ValueError(f"Unknown PCQM Gap candidate: {candidate}")
