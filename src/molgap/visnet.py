"""ViSNet 3D encoder wrapper for the A/B 3D-encoder comparison.

ViSNet (Wang et al., "Enhancing Geometric Representations for Molecules with
Equivariant Vector-Scalar Interactive Message Passing", arXiv:2210.16518) is an
equivariant vector-scalar GNN. PyG ships its representation module as
``torch_geometric.nn.models.visnet.ViSNetBlock``; we pool its invariant scalar
node features into a molecule-level embedding.

Interface mirrors ``SchNetWrapper``:
    encode(z, pos, batch, charges=None) -> [num_molecules, hidden_channels]
    forward(z, pos, batch, charges=None) -> [num_molecules, n_targets]

``charges`` is accepted for signature parity with SchNet but unused — ViSNet is
used in its native form (atomic numbers + geometry only).
"""
from __future__ import annotations

import torch.nn as nn


class ViSNetWrapper(nn.Module):
    def __init__(self, hidden_channels=192, num_layers=6, num_heads=8,
                 num_rbf=32, cutoff=6.0, dropout=0.0, n_targets=3,
                 max_num_neighbors=32, use_charges=False):
        super().__init__()
        from torch_geometric.nn.models.visnet import ViSNetBlock

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.use_charges = use_charges  # accepted for parity, intentionally unused
        self.representation = ViSNetBlock(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, z, pos, batch, charges=None):
        return self.head(self.encode(z, pos, batch, charges=charges))

    def encode(self, z, pos, batch, charges=None):
        """Pooled molecular embedding (pre-head): mean over invariant scalar
        node features from ViSNet's vector-scalar message passing."""
        from torch_geometric.nn import global_mean_pool

        x, _vec = self.representation(z, pos, batch)
        return global_mean_pool(x, batch)
