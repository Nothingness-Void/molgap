"""Spawn-pickleable owner of the legacy two-item packed graph payload."""
from pathlib import Path

from torch_geometric.data import InMemoryDataset

from .v4_runtime import torch_load_compat


class PackedGraphDataset(InMemoryDataset):
    """Keep PyG's inherited separation and lazy graph-cache semantics."""

    def __init__(self, path: Path):
        super().__init__(root=None)
        self.data, self.slices = torch_load_compat(
            path, map_location="cpu", weights_only=False
        )
