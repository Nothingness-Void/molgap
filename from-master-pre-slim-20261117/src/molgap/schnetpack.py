"""SchNetPack 2.x utilities for DCU-compatible ETKDG 3D experiments.

This path deliberately avoids the PyG ``radius_graph`` dependency.  It is kept
separate from the production PyG SchNet wrapper until a like-for-like accuracy
gate establishes that it is a viable replacement.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def load_schnetpack_components():
    """Import optional SchNetPack modules only when the alternate path is used."""
    import schnetpack as spk
    from schnetpack import properties
    from schnetpack.representation import SchNet
    from schnetpack.transform import TorchNeighborList

    return spk, properties, SchNet, TorchNeighborList


def make_schnetpack_batch(
    graphs: Sequence,
    neighbor_list,
    properties,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    """Build one concatenated SchNetPack batch from labeled PyG ETKDG graphs."""
    z_parts, pos_parts, center_parts = [], [], []
    idx_i_parts, idx_j_parts, offset_parts, y_parts = [], [], [], []
    atom_offset = 0

    for molecule_idx, graph in enumerate(graphs):
        z = graph.z.to(device=device, dtype=torch.long)
        pos = graph.pos.to(device=device, dtype=torch.float32)
        inputs = {
            properties.Z: z,
            properties.R: pos,
            properties.cell: torch.zeros((3, 3), device=device, dtype=pos.dtype),
            properties.pbc: torch.zeros(3, device=device, dtype=torch.bool),
        }
        pairs = neighbor_list(inputs)
        z_parts.append(z)
        pos_parts.append(pos)
        center_parts.append(
            torch.full((z.numel(),), molecule_idx, device=device, dtype=torch.long)
        )
        idx_i_parts.append(pairs[properties.idx_i] + atom_offset)
        idx_j_parts.append(pairs[properties.idx_j] + atom_offset)
        offset_parts.append(pairs[properties.offsets])
        y_parts.append(graph.y.view(-1).to(device=device, dtype=torch.float32))
        atom_offset += z.numel()

    return {
        properties.Z: torch.cat(z_parts),
        properties.R: torch.cat(pos_parts),
        properties.idx_i: torch.cat(idx_i_parts),
        properties.idx_j: torch.cat(idx_j_parts),
        properties.offsets: torch.cat(offset_parts),
        properties.idx_m: torch.cat(center_parts),
    }, torch.stack(y_parts)


class SchNetPackRegressor(nn.Module):
    """Three-target molecular regressor using SchNetPack's native neighbour list."""

    def __init__(
        self,
        hidden_channels: int = 192,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 6.0,
        n_targets: int = 3,
    ) -> None:
        super().__init__()
        spk, properties, SchNet, TorchNeighborList = load_schnetpack_components()
        self.properties = properties
        self.neighbor_list = TorchNeighborList(cutoff=cutoff)
        self.distances = spk.atomistic.PairwiseDistances()
        self.representation = SchNet(
            n_atom_basis=hidden_channels,
            n_interactions=num_interactions,
            radial_basis=spk.nn.GaussianRBF(n_rbf=num_gaussians, cutoff=cutoff),
            cutoff_fn=spk.nn.CosineCutoff(cutoff=cutoff),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, graphs: Sequence, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, target = make_schnetpack_batch(
            graphs, self.neighbor_list, self.properties, device
        )
        inputs = self.distances(inputs)
        encoded = self.representation(inputs)["scalar_representation"]
        molecule_count = target.shape[0]
        pooled = torch.zeros(
            (molecule_count, encoded.shape[1]), device=device, dtype=encoded.dtype
        )
        pooled.index_add_(0, inputs[self.properties.idx_m], encoded)
        counts = torch.bincount(
            inputs[self.properties.idx_m], minlength=molecule_count
        ).clamp_min(1).unsqueeze(1)
        return self.head(pooled / counts), target
