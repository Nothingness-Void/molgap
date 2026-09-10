"""Frozen GraphState implementation extracted from server commit 9068ddb."""
import torch
import torch.nn as nn

from .pcqm_gap_architecture import (
    OGBGeometrySparseTriangleEdgeStateGPSWrapper, _LowRankGatedProjection,
)


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


def make_graph_state():
    return OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper(
        in_channels=9, edge_dim=3, hidden_channels=192, num_layers=9,
        num_heads=4, dropout=0.1, n_targets=1, pooling="mean", rwse_dim=16,
        edge_state_channels=64, wedge_channels=16, geometry_basis_channels=16,
        global_mode="graph_state", graph_state_channels=64, graph_exchange_rank=32,
    )

