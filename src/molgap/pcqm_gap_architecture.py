"""Official-PCQM categorical encoders for bounded Gap-only architecture screens."""
from __future__ import annotations

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
    raise ValueError(f"Unknown PCQM Gap candidate: {candidate}")
