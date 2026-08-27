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
