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

    def encode(self, x, edge_index, edge_attr, batch):
        """Return molecule-level embeddings [num_molecules, hidden_channels]."""
        h = self.node_emb(x.float())
        e = self.edge_emb(edge_attr.float())

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

        h = self.node_emb(x.float())
        e = self.edge_emb(edge_attr.float())
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

        h = self.node_emb(x.float())
        h = h + self.rwse_encoder(random_walk_pe.float())
        e = self.edge_emb(edge_attr.float())
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

        atom_embedding = self.node_emb(x.float())
        normalized_rwse = self.rwse_normalizer(random_walk_pe.float())
        rwse_embedding = self.rwse_encoder(normalized_rwse)
        h = self.h0_norm(atom_embedding + self.rwse_alpha * rwse_embedding)
        e = self.edge_emb(edge_attr.float())
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

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
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

        h = self.node_emb(x.float())
        h = h + self.rwse_encoder(random_walk_pe.float())
        edge_state = self.edge_emb(edge_attr.float())
        for edge_update, conv in zip(self.edge_updates, self.convs):
            edge_state = edge_update(h, edge_index, edge_state)
            h = conv(h, edge_index, batch, edge_attr=edge_state)
        return self._pool(h, batch)


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
