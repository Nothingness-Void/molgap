"""Official-PCQM categorical encoders for bounded Gap-only architecture screens."""
from __future__ import annotations

import torch
import torch.nn as nn

from .gps import EdgeStateStructuralGPSWrapper, StructuralGPSWrapper


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
    if candidate == "ogb_query_pool_structural_gps9":
        return OGBQueryPoolStructuralGPSWrapper(
            **common,
            num_pool_queries=4,
        )
    raise ValueError(f"Unknown PCQM Gap candidate: {candidate}")
