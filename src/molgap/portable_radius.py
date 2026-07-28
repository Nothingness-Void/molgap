"""Vectorized PyTorch fallback for batched molecular radius graphs."""

from __future__ import annotations

import torch


BACKEND = "molgap_portable_torch_radius_v1"


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: torch.Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Build a bounded radius graph without compiled PyG extensions."""
    del num_workers
    if flow not in {"source_to_target", "target_to_source"}:
        raise ValueError(f"Unsupported flow: {flow}")
    if x.ndim != 2:
        raise ValueError(f"x must be [N, D], got {tuple(x.shape)}")
    if x.shape[0] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
        raise ValueError("batch must have one entry per coordinate row")
    if bool((batch[1:] < batch[:-1]).any()):
        raise ValueError("portable radius graph requires sorted PyG batches")

    inferred_batch_size = int(batch[-1]) + 1
    n_graphs = inferred_batch_size if batch_size is None else int(batch_size)
    if n_graphs < inferred_batch_size:
        raise ValueError("batch_size is smaller than the observed batch ids")
    counts = torch.bincount(batch, minlength=n_graphs)
    max_nodes = int(counts.max())
    offsets = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(x.shape[0], device=x.device) - offsets[batch]

    dense = x.new_zeros((n_graphs, max_nodes, x.shape[1]))
    dense[batch, local] = x
    slots = torch.arange(max_nodes, device=x.device)
    valid_nodes = slots.unsqueeze(0) < counts.unsqueeze(1)
    distances = torch.cdist(dense, dense)
    valid_pairs = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
    valid_pairs &= distances < float(r)
    if not loop:
        diagonal = torch.eye(
            max_nodes, dtype=torch.bool, device=x.device
        ).unsqueeze(0)
        valid_pairs &= ~diagonal
    distances = distances.masked_fill(~valid_pairs, float("inf"))

    k = min(int(max_num_neighbors), max_nodes)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    neighbor_distance, neighbor_local = torch.topk(
        distances, k=k, dim=2, largest=False, sorted=False
    )
    selected = torch.isfinite(neighbor_distance)
    graph_id, center_local, choice = selected.nonzero(as_tuple=True)
    neighbor_local = neighbor_local[graph_id, center_local, choice]
    center = offsets[graph_id] + center_local
    neighbor = offsets[graph_id] + neighbor_local
    if flow == "source_to_target":
        return torch.stack((neighbor, center))
    return torch.stack((center, neighbor))

