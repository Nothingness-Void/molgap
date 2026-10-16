"""Spawn-pickleable owner of the legacy two-item packed graph payload."""
from pathlib import Path

from torch_geometric.data import InMemoryDataset

from .v4_runtime import torch_load_compat


TOPOLOGY_EXCLUDED_FIELDS = ("pos", "edge_distance", "wedge_angle_cos", "wedge_edge_ids", "geometry_valid")


class PackedGraphDataset(InMemoryDataset):
    """Keep PyG's inherited separation and lazy graph-cache semantics."""

    def __init__(self, path: Path, *, mmap: bool = False):
        super().__init__(root=None)
        self.data, self.slices = torch_load_compat(
            path, map_location="cpu", weights_only=False, **({"mmap": True} if mmap else {})
        )

    def topology_only(self):
        """Drop incidental legacy geometry in memory, without changing shard bytes."""
        for field in TOPOLOGY_EXCLUDED_FIELDS:
            if field in self._data:
                del self._data[field]
                if self.slices is not None:
                    self.slices.pop(field, None)
        self._data_list = None
        return self
