"""Categorical desktop encoders kept separate from the frozen K1 GPS source."""

import torch
from torch import nn

from .gps import EdgeStateStructuralGPSWrapper


class CategoricalFeatureEncoder(nn.Module):
    """Sum one learned embedding per categorical molecular feature field."""

    def __init__(self, feature_dims, embedding_dim: int):
        super().__init__()
        dimensions = tuple(int(value) for value in feature_dims)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ValueError("feature_dims must contain positive category counts")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.feature_dims = dimensions
        self.embeddings = nn.ModuleList(
            nn.Embedding(categories, embedding_dim) for categories in dimensions
        )
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.embeddings):
            raise ValueError(
                f"categorical features must have shape [rows, {len(self.embeddings)}]"
            )
        encoded = self.embeddings[0](features[:, 0].long())
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            encoded = encoded + embedding(features[:, column].long())
        return encoded


class CategoricalConcatFeatureEncoder(nn.Module):
    """Preserve categorical field identity before projecting to model width."""

    def __init__(self, feature_dims, embedding_dim: int, *, field_channels: int = 16):
        super().__init__()
        dimensions = tuple(int(value) for value in feature_dims)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ValueError("feature_dims must contain positive category counts")
        if embedding_dim <= 0 or field_channels <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.feature_dims = dimensions
        self.field_channels = int(field_channels)
        self.embeddings = nn.ModuleList(
            nn.Embedding(categories, self.field_channels)
            for categories in dimensions
        )
        self.projection = nn.Linear(
            len(dimensions) * self.field_channels, embedding_dim
        )
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.embeddings):
            raise ValueError(
                f"categorical features must have shape [rows, {len(self.embeddings)}]"
            )
        fields = [
            embedding(features[:, column].long())
            for column, embedding in enumerate(self.embeddings)
        ]
        return self.projection(torch.cat(fields, dim=-1))


class CategoricalEdgeStateStructuralGPSWrapper(EdgeStateStructuralGPSWrapper):
    """Persistent EdgeState GPS using complete categorical atom/bond fields."""

    def __init__(
        self,
        *,
        atom_feature_dims,
        bond_feature_dims,
        hidden_channels=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        n_targets=3,
        pooling="mean",
        rwse_dim=16,
        edge_state_channels=64,
        categorical_encoder="sum",
        categorical_field_channels=16,
    ):
        atom_dimensions = tuple(int(value) for value in atom_feature_dims)
        bond_dimensions = tuple(int(value) for value in bond_feature_dims)
        super().__init__(
            in_channels=len(atom_dimensions),
            edge_dim=len(bond_dimensions),
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_targets=n_targets,
            pooling=pooling,
            rwse_dim=rwse_dim,
            edge_state_channels=edge_state_channels,
        )
        if categorical_encoder == "sum":
            encoder = CategoricalFeatureEncoder
            encoder_kwargs = {}
        elif categorical_encoder == "concat_project":
            encoder = CategoricalConcatFeatureEncoder
            encoder_kwargs = {"field_channels": categorical_field_channels}
        else:
            raise ValueError(
                "categorical_encoder must be 'sum' or 'concat_project'"
            )
        self.categorical_encoder = categorical_encoder
        self.node_emb = encoder(atom_dimensions, hidden_channels, **encoder_kwargs)
        self.edge_emb = encoder(
            bond_dimensions, edge_state_channels, **encoder_kwargs
        )


class CategoricalRadicalContextEdgeStateStructuralGPSWrapper(
    CategoricalEdgeStateStructuralGPSWrapper
):
    """Add a graph-level radical-electron context without perturbing closed shells."""

    def __init__(
        self,
        *,
        radical_context_channels=16,
        radical_context_gate_init=0.1,
        **kwargs,
    ):
        if radical_context_channels <= 0:
            raise ValueError("radical_context_channels must be positive")
        if not 0.0 < radical_context_gate_init < 1.0:
            raise ValueError("radical_context_gate_init must be in (0, 1)")
        super().__init__(**kwargs)
        hidden_channels = self.head[0].in_features
        radical_categories = self.node_emb.feature_dims[5]
        self.radical_context = nn.Embedding(
            radical_categories, radical_context_channels, padding_idx=0
        )
        nn.init.xavier_uniform_(self.radical_context.weight)
        with torch.no_grad():
            self.radical_context.weight[0].zero_()
        self.radical_context_projection = nn.Sequential(
            nn.Linear(radical_context_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=False),
        )
        initial = torch.tensor(float(radical_context_gate_init))
        self.radical_context_gate_logit = nn.Parameter(torch.logit(initial))

    @property
    def radical_context_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.radical_context_gate_logit)

    def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
        from torch_geometric.nn import global_add_pool

        embedding = super().encode(
            x, edge_index, edge_attr, batch, random_walk_pe
        )
        atom_context = self.radical_context(x[:, 5].long())
        graph_context = global_add_pool(atom_context, batch)
        correction = self.radical_context_projection(graph_context)
        return embedding + self.radical_context_gate * correction
