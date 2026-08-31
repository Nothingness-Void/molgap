"""Sparse bonded-path torsion cache primitives for the PCQM Gap screen.

The representation keeps only directed non-backtracking bonded paths
``i-j-k-l``.  Torsion rows refer to the three directed bond ids and the two
adjacent cached wedge ids, so PyG batching can preserve the sparse index
contract without constructing a dense four-body tensor.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


TORSION_FEATURE_DIM = 4


class TorsionData(Data):
    """PyG data with torsion ids incremented by edge and wedge counts."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in {"wedge_edge_ids", "torsion_edge_ids"}:
            return int(self.edge_index.shape[1])
        if key == "torsion_wedge_ids":
            return int(self.wedge_edge_ids.shape[0])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"wedge_edge_ids", "torsion_edge_ids", "torsion_wedge_ids"}:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def _as_numpy(value, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def directed_nonbacktracking_torsions(
    edge_index: torch.Tensor,
    wedge_edge_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[T, 3]`` bond ids and ``[T, 2]`` adjacent wedge ids.

    The first wedge is ``i -> j -> k`` and the second is ``j -> k -> l``.
    Immediate reversals are excluded at both internal atoms.  Both directed
    orientations present in the bond graph are retained.
    """
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise ValueError("edge_index must have shape [2, E]")
    if wedge_edge_ids.ndim != 2 or wedge_edge_ids.shape[1] != 2:
        raise ValueError("wedge_edge_ids must have shape [W, 2]")
    edge_index = edge_index.long()
    wedge_edge_ids = wedge_edge_ids.long()
    edge_count = int(edge_index.shape[1])
    wedge_count = int(wedge_edge_ids.shape[0])
    if wedge_count and (
        int(wedge_edge_ids.min()) < 0
        or int(wedge_edge_ids.max()) >= edge_count
    ):
        raise ValueError("wedge edge id falls outside the edge set")

    source = edge_index[0].tolist()
    target = edge_index[1].tolist()
    outgoing: list[list[int]] = [[] for _ in range(int(edge_index.max()) + 1)] if edge_count else []
    for edge_id, atom in enumerate(source):
        outgoing[atom].append(edge_id)

    wedge_lookup: dict[tuple[int, int], int] = {}
    for wedge_id, (first, second) in enumerate(wedge_edge_ids.tolist()):
        if target[first] != source[second]:
            raise ValueError("wedge ids do not describe adjacent directed bonds")
        if source[first] == target[second]:
            raise ValueError("wedge ids contain an immediate reversal")
        key = (int(first), int(second))
        if key in wedge_lookup:
            raise ValueError("wedge ids contain a duplicate directed pair")
        wedge_lookup[key] = wedge_id

    torsion_edges: list[tuple[int, int, int]] = []
    torsion_wedges: list[tuple[int, int]] = []
    for first_wedge, (first, second) in enumerate(wedge_edge_ids.tolist()):
        center = target[second]
        for third in outgoing[center]:
            if target[third] == source[second]:
                continue
            second_wedge = wedge_lookup.get((int(second), int(third)))
            if second_wedge is None:
                raise ValueError("wedge cache is incomplete for a torsion path")
            torsion_edges.append((int(first), int(second), int(third)))
            torsion_wedges.append((first_wedge, second_wedge))

    if not torsion_edges:
        return (
            torch.empty((0, 3), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
        )
    return (
        torch.tensor(torsion_edges, dtype=torch.long),
        torch.tensor(torsion_wedges, dtype=torch.long),
    )


def torsion_fourier_features(
    positions,
    edge_index: torch.Tensor,
    torsion_edge_ids: torch.Tensor,
    geometry_valid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode signed dihedrals as fixed periodic features and validity masks.

    The convention is
    ``b0 = p_i-p_j``, ``b1 = p_k-p_j``, ``b2 = p_l-p_k`` and
    ``phi = atan2(dot(cross(b1_hat, v), w), dot(v, w))`` where ``v`` and
    ``w`` are the projections of ``b0`` and ``b2`` perpendicular to
    ``b1_hat``.  Degenerate paths receive a zero feature and zero mask.
    """
    positions = _as_numpy(positions, name="positions").astype(np.float64, copy=False)
    edge_array = _as_numpy(edge_index, name="edge_index").astype(np.int64, copy=False)
    torsion_array = _as_numpy(torsion_edge_ids, name="torsion_edge_ids").astype(
        np.int64, copy=False
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if edge_array.ndim != 2 or edge_array.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if torsion_array.ndim != 2 or torsion_array.shape[1] != 3:
        raise ValueError("torsion_edge_ids must have shape [T, 3]")
    count = int(torsion_array.shape[0])
    features = np.zeros((count, TORSION_FEATURE_DIM), dtype=np.float32)
    valid = np.zeros((count, 1), dtype=np.float32)
    if count == 0 or not bool(geometry_valid):
        return features, valid
    if int(torsion_array.min()) < 0 or int(torsion_array.max()) >= edge_array.shape[1]:
        raise ValueError("torsion edge id falls outside the edge set")

    first, second, third = torsion_array.T
    atom_i = edge_array[0, first]
    atom_j = edge_array[1, first]
    atom_k = edge_array[1, second]
    atom_l = edge_array[1, third]
    if not np.array_equal(atom_j, edge_array[0, second]):
        raise ValueError("torsion first and second bond are not adjacent")
    if not np.array_equal(atom_k, edge_array[0, third]):
        raise ValueError("torsion second and third bond are not adjacent")
    if np.any(atom_i == atom_k) or np.any(atom_j == atom_l):
        raise ValueError("torsion ids contain an immediate reversal")

    p_i = positions[atom_i]
    p_j = positions[atom_j]
    p_k = positions[atom_k]
    p_l = positions[atom_l]
    b0 = p_i - p_j
    b1 = p_k - p_j
    b2 = p_l - p_k
    b1_norm = np.linalg.norm(b1, axis=1)
    valid_b1 = np.isfinite(b1_norm) & (b1_norm > 1.0e-12)
    if not valid_b1.any():
        return features, valid
    b1_hat = np.zeros_like(b1)
    b1_hat[valid_b1] = b1[valid_b1] / b1_norm[valid_b1, None]
    v = b0 - np.sum(b0 * b1_hat, axis=1, keepdims=True) * b1_hat
    w = b2 - np.sum(b2 * b1_hat, axis=1, keepdims=True) * b1_hat
    v_norm = np.linalg.norm(v, axis=1)
    w_norm = np.linalg.norm(w, axis=1)
    row_valid = valid_b1 & np.isfinite(v_norm) & np.isfinite(w_norm)
    row_valid &= (v_norm > 1.0e-12) & (w_norm > 1.0e-12)
    if not row_valid.any():
        return features, valid
    cross = np.cross(b1_hat, v)
    x = np.sum(v * w, axis=1)
    y = np.sum(cross * w, axis=1)
    phi = np.arctan2(y, x)
    row_valid &= np.isfinite(phi)
    features[row_valid, 0] = np.sin(phi[row_valid]).astype(np.float32)
    features[row_valid, 1] = np.cos(phi[row_valid]).astype(np.float32)
    features[row_valid, 2] = np.sin(2.0 * phi[row_valid]).astype(np.float32)
    features[row_valid, 3] = np.cos(2.0 * phi[row_valid]).astype(np.float32)
    valid[row_valid, 0] = 1.0
    if not np.isfinite(features).all():
        raise ValueError("torsion Fourier features are not finite")
    return features, valid


def with_torsion_cache(graph: Data) -> TorsionData:
    """Copy an accepted geometry graph and attach its sparse torsion payload."""
    required = ("edge_index", "wedge_edge_ids", "pos", "geometry_valid")
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise ValueError(f"geometry graph is missing {missing}")
    payload = graph.to_dict()
    result = TorsionData(**payload)
    edge_ids, wedge_ids = directed_nonbacktracking_torsions(
        graph.edge_index, graph.wedge_edge_ids
    )
    features, valid = torsion_fourier_features(
        graph.pos,
        graph.edge_index,
        edge_ids,
        bool(graph.geometry_valid.reshape(-1)[0]),
    )
    result.torsion_edge_ids = edge_ids
    result.torsion_wedge_ids = wedge_ids
    result.torsion_fourier = torch.from_numpy(features)
    result.torsion_valid = torch.from_numpy(valid)
    return result


def torsion_path_count(graphs) -> int:
    """Return total cached torsion paths without running a model."""
    return sum(int(graph.torsion_edge_ids.shape[0]) for graph in graphs)


__all__ = [
    "TORSION_FEATURE_DIM",
    "TorsionData",
    "directed_nonbacktracking_torsions",
    "torsion_fourier_features",
    "torsion_path_count",
    "with_torsion_cache",
]
