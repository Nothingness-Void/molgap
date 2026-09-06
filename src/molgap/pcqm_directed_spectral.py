"""Bounded directed-bond and spectral additions to the PCQM GraphState anchor."""
from __future__ import annotations

import torch
import torch.nn as nn

from .pcqm_gap_architecture import (
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper,
)


class OGBDirectedBondGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """Add directional non-backtracking edge-to-edge memory.

    The accepted wedge cache already enumerates directed pairs ``i->j->k``.
    Each layer averages predecessor edge states only onto the outgoing second
    edge, applies a separately normalized MLP, and returns that signal through
    a zero-initialized projection. The underlying persistent EdgeState, wedge
    geometry, local atom updates, and shared GraphState are unchanged.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, global_mode="graph_state", **kwargs)
        channels = int(self.edge_state_channels)
        dropout = float(kwargs.get("dropout", 0.1))
        self.directed_edge_returns = nn.ModuleList()
        for _ in self.convs:
            block = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, channels, bias=False),
            )
            nn.init.zeros_(block[-1].weight)
            self.directed_edge_returns.append(block)

    def _pre_edge_update(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        wedge_edge_ids: torch.Tensor,
        auxiliary_state,
    ) -> torch.Tensor:
        del h, edge_index, auxiliary_state
        if wedge_edge_ids.shape[0] == 0:
            return edge_state
        first, second = wedge_edge_ids.unbind(dim=1)
        incoming = edge_state.new_zeros(edge_state.shape)
        counts = edge_state.new_zeros((edge_state.shape[0], 1))
        incoming.index_add_(0, second, edge_state[first])
        counts.index_add_(
            0,
            second,
            edge_state.new_ones((second.shape[0], 1)),
        )
        incoming = incoming / counts.clamp_min_(1.0)
        return edge_state + self.directed_edge_returns[layer](incoming)


class OGBSignNetLapPEGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """Add sign-invariant normalized-Laplacian eigenvector encodings.

    RWSE16 is retained. For every cached eigenpair, a shared MLP processes
    ``phi(+v, lambda)`` and ``phi(-v, lambda)`` and sums the two paths. Valid modes
    are then summed per atom and returned through a zero-initialized projection.
    """

    def __init__(
        self,
        *args,
        lappe_dim: int = 8,
        signnet_channels: int = 32,
        **kwargs,
    ) -> None:
        if lappe_dim <= 0 or signnet_channels <= 0:
            raise ValueError("LapPE and SignNet widths must be positive")
        super().__init__(*args, global_mode="graph_state", **kwargs)
        hidden_channels = int(self.head[0].in_features)
        self.lappe_dim = int(lappe_dim)
        self.signnet_channels = int(signnet_channels)
        self.signnet_phi = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, self.signnet_channels),
            nn.SiLU(),
            nn.Linear(self.signnet_channels, self.signnet_channels),
        )
        self.lappe_to_atom = nn.Linear(
            self.signnet_channels,
            hidden_channels,
            bias=False,
        )
        nn.init.zeros_(self.lappe_to_atom.weight)

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
        lap_eigvec,
        lap_eigval,
        lap_mask,
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
            lap_eigvec,
            lap_eigval,
            lap_mask,
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
        lap_eigvec,
        lap_eigval,
        lap_mask,
    ):
        payload = (lap_eigvec, lap_eigval, lap_mask)
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
            auxiliary_payload=payload,
        )

    def _pre_auxiliary_node_injection(
        self,
        h: torch.Tensor,
        auxiliary_payload,
    ) -> torch.Tensor:
        if not isinstance(auxiliary_payload, tuple) or len(auxiliary_payload) != 3:
            raise ValueError("SignNet-LapPE requires eigvec/eigval/mask tensors")
        eigvec, eigval, mask = auxiliary_payload
        expected = (h.shape[0], self.lappe_dim)
        if tuple(eigvec.shape) != expected or tuple(eigval.shape) != expected:
            raise ValueError(f"LapPE tensors must have shape {expected}")
        if tuple(mask.shape) != expected:
            raise ValueError(f"LapPE mask must have shape {expected}")
        if not torch.isfinite(eigvec).all() or not torch.isfinite(eigval).all():
            raise ValueError("LapPE tensors contain non-finite values")
        mask = mask.to(device=h.device, dtype=h.dtype)
        positive = torch.stack([eigvec.float(), eigval.float()], dim=-1)
        negative = torch.stack([-eigvec.float(), eigval.float()], dim=-1)
        encoded = self.signnet_phi(positive) + self.signnet_phi(negative)
        encoded = (encoded * mask.unsqueeze(-1)).sum(dim=1)
        return h + self.lappe_to_atom(encoded)

    def _initialize_geometry_auxiliary(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_payload,
    ):
        if not isinstance(auxiliary_payload, tuple) or len(auxiliary_payload) != 3:
            raise ValueError("SignNet-LapPE auxiliary payload changed")
        return self.graph_context.initialize(h, batch)


def make_directed_spectral_encoder(candidate: str):
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
        "edge_state_channels": 64,
        "wedge_channels": 16,
        "geometry_basis_channels": 16,
        "graph_state_channels": 64,
        "graph_exchange_rank": 32,
    }
    if candidate == (
        "ogb_distance_angle_directed_bond_triangle_edge_state_graph_state9"
    ):
        return OGBDirectedBondGraphStateWrapper(**common)
    if candidate == (
        "ogb_distance_angle_signnet_lappe_triangle_edge_state_graph_state9"
    ):
        return OGBSignNetLapPEGraphStateWrapper(
            **common,
            lappe_dim=8,
            signnet_channels=32,
        )
    raise ValueError(f"Unknown directed/spectral candidate: {candidate}")
