"""Low-cost local angle and dihedral features for 3D molecular encoders."""
from __future__ import annotations

from itertools import combinations

import torch


ANGLE_BINS = 8
ANGLE_FEATURE_DIM = ANGLE_BINS + 1
DIHEDRAL_ORDERS = 4
FULL_FEATURE_DIM = ANGLE_FEATURE_DIM + DIHEDRAL_ORDERS + 1

_COVALENT_RADII = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}


def _bond_neighbors(
    z: torch.Tensor,
    pos: torch.Tensor,
    *,
    radius_scale: float = 1.25,
) -> list[list[int]]:
    """Infer covalent neighbors from optimized bond-length geometry."""
    n_atoms = int(z.numel())
    neighbors = [[] for _ in range(n_atoms)]
    for i in range(n_atoms):
        radius_i = _COVALENT_RADII.get(int(z[i]), 0.77)
        for j in range(i + 1, n_atoms):
            radius_j = _COVALENT_RADII.get(int(z[j]), 0.77)
            distance = float(torch.linalg.vector_norm(pos[i] - pos[j]))
            if 0.4 < distance <= radius_scale * (radius_i + radius_j):
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def local_geometry_features(
    z: torch.Tensor,
    pos: torch.Tensor,
    *,
    angle_bins: int = ANGLE_BINS,
    dihedral_orders: int = DIHEDRAL_ORDERS,
) -> torch.Tensor:
    """Return per-atom invariant summaries of bonded angles and dihedrals.

    Angle values use a Gaussian basis over cos(theta). Dihedrals use
    cos(n * phi), which is invariant to rotation, translation, and reflection.
    """
    if z.ndim != 1 or pos.ndim != 2 or pos.shape != (z.numel(), 3):
        raise ValueError("Expected z=[N] and pos=[N, 3]")

    device = pos.device
    dtype = pos.dtype
    n_atoms = int(z.numel())
    neighbors = _bond_neighbors(z, pos)

    centers = torch.linspace(-1.0, 1.0, angle_bins, device=device, dtype=dtype)
    angle_sum = torch.zeros((n_atoms, angle_bins), device=device, dtype=dtype)
    angle_count = torch.zeros((n_atoms, 1), device=device, dtype=dtype)

    for center, adjacent in enumerate(neighbors):
        for first, second in combinations(adjacent, 2):
            left = pos[first] - pos[center]
            right = pos[second] - pos[center]
            denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
            if float(denominator) <= 1e-8:
                continue
            cosine = torch.clamp(torch.dot(left, right) / denominator, -1.0, 1.0)
            angle_sum[center] += torch.exp(-8.0 * (cosine - centers).square())
            angle_count[center] += 1.0

    angle_mean = angle_sum / angle_count.clamp_min(1.0)
    angle_density = torch.log1p(angle_count) / torch.log(
        torch.tensor(7.0, device=device, dtype=dtype)
    )

    torsion_sum = torch.zeros(
        (n_atoms, dihedral_orders), device=device, dtype=dtype
    )
    torsion_count = torch.zeros((n_atoms, 1), device=device, dtype=dtype)
    for center_a in range(n_atoms):
        for center_b in neighbors[center_a]:
            if center_b <= center_a:
                continue
            left_neighbors = [i for i in neighbors[center_a] if i != center_b]
            right_neighbors = [i for i in neighbors[center_b] if i != center_a]
            for outer_a in left_neighbors:
                for outer_b in right_neighbors:
                    bond_left = pos[center_a] - pos[outer_a]
                    bond_center = pos[center_b] - pos[center_a]
                    bond_right = pos[outer_b] - pos[center_b]
                    normal_left = torch.linalg.cross(
                        bond_left, bond_center, dim=0
                    )
                    normal_right = torch.linalg.cross(
                        bond_center, bond_right, dim=0
                    )
                    denominator = (
                        torch.linalg.vector_norm(normal_left)
                        * torch.linalg.vector_norm(normal_right)
                    )
                    if float(denominator) <= 1e-8:
                        continue
                    cosine = torch.clamp(
                        torch.dot(normal_left, normal_right) / denominator,
                        -1.0,
                        1.0,
                    )
                    phi = torch.acos(cosine)
                    values = torch.stack(
                        [
                            torch.cos(order * phi)
                            for order in range(1, dihedral_orders + 1)
                        ]
                    )
                    torsion_sum[center_a] += values
                    torsion_sum[center_b] += values
                    torsion_count[center_a] += 1.0
                    torsion_count[center_b] += 1.0

    torsion_mean = torsion_sum / torsion_count.clamp_min(1.0)
    torsion_density = torch.log1p(torsion_count) / torch.log(
        torch.tensor(13.0, device=device, dtype=dtype)
    )
    return torch.cat(
        (angle_mean, angle_density, torsion_mean, torsion_density), dim=-1
    )


def select_geometry_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "angle":
        return features[:, :ANGLE_FEATURE_DIM]
    if mode == "angle_dihedral":
        return features
    raise ValueError(f"Unsupported local geometry mode: {mode}")
