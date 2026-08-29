"""Deterministic ETKDGv3/MMFF geometry for the PCQM Gap100K screen.

Geometry is attached to the already accepted OGB graph and sparse-wedge
representation.  Failed embeddings remain in their original role with an
explicit zero mask; no molecule is silently filtered or replaced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


ETKDG_BASE_SEED = 42
MMFF_VARIANT = "MMFF94s"
MMFF_MAX_ITERS = 200


@dataclass(frozen=True)
class GeometryResult:
    """One aligned heavy-atom geometry payload returned by a CPU worker."""

    positions: np.ndarray
    edge_distance: np.ndarray
    wedge_angle_cos: np.ndarray
    geometry_valid: bool
    mmff_converged: bool
    embed_attempt: str
    failure_type: str | None = None
    failure_message: str | None = None


def geometry_seed(row_index: int) -> int:
    """Map an immutable PCQM row index to a valid deterministic RDKit seed."""
    value = (ETKDG_BASE_SEED * 1_000_003 + int(row_index)) % 2_147_483_647
    return int(value or ETKDG_BASE_SEED)


def _zero_result(
    node_count: int,
    edge_count: int,
    wedge_count: int,
    *,
    embed_attempt: str,
    error: Exception,
) -> GeometryResult:
    return GeometryResult(
        positions=np.zeros((node_count, 3), dtype=np.float32),
        edge_distance=np.zeros((edge_count, 1), dtype=np.float32),
        wedge_angle_cos=np.zeros((wedge_count, 1), dtype=np.float32),
        geometry_valid=False,
        mmff_converged=False,
        embed_attempt=embed_attempt,
        failure_type=type(error).__name__,
        failure_message=str(error),
    )


def compute_etkdg_geometry(
    smiles: str,
    row_index: int,
    node_count: int,
    edge_index,
    wedge_edge_ids,
) -> GeometryResult:
    """Build one ETKDGv3 conformer and aligned bond/angle observables.

    The first attempt uses standard ETKDGv3.  A deterministic random-coordinate
    ETKDGv3 attempt is allowed only when embedding fails.  MMFF94s is applied
    when parameters exist; an embedded but non-converged conformer stays
    visible through ``mmff_converged=False``.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    edge_array = np.asarray(edge_index, dtype=np.int64)
    wedge_array = np.asarray(wedge_edge_ids, dtype=np.int64)
    edge_count = int(edge_array.shape[1])
    wedge_count = int(wedge_array.shape[0])
    attempt = "etkdgv3"
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError("RDKit could not parse SMILES")
        if molecule.GetNumAtoms() != int(node_count):
            raise ValueError(
                f"heavy-atom count changed: {molecule.GetNumAtoms()} != {node_count}"
            )
        with_hydrogens = Chem.AddHs(molecule)
        params = AllChem.ETKDGv3()
        params.randomSeed = geometry_seed(row_index)
        params.useRandomCoords = False
        status = int(AllChem.EmbedMolecule(with_hydrogens, params))
        if status != 0:
            attempt = "etkdgv3_random_coords"
            params = AllChem.ETKDGv3()
            params.randomSeed = geometry_seed(row_index)
            params.useRandomCoords = True
            status = int(AllChem.EmbedMolecule(with_hydrogens, params))
        if status != 0:
            raise RuntimeError(f"ETKDGv3 embedding failed with status {status}")

        mmff_converged = False
        if AllChem.MMFFHasAllMoleculeParams(with_hydrogens):
            optimize_status = int(
                AllChem.MMFFOptimizeMolecule(
                    with_hydrogens,
                    mmffVariant=MMFF_VARIANT,
                    maxIters=MMFF_MAX_ITERS,
                )
            )
            mmff_converged = optimize_status == 0

        conformer = with_hydrogens.GetConformer()
        positions = np.asarray(
            [
                [
                    conformer.GetAtomPosition(index).x,
                    conformer.GetAtomPosition(index).y,
                    conformer.GetAtomPosition(index).z,
                ]
                for index in range(node_count)
            ],
            dtype=np.float32,
        )
        if positions.shape != (node_count, 3) or not np.isfinite(positions).all():
            raise ValueError("ETKDGv3 positions are not finite and aligned")

        if edge_count:
            source, target = edge_array
            displacement = positions[source] - positions[target]
            edge_distance = np.linalg.norm(displacement, axis=1).astype(np.float32)
            if not np.isfinite(edge_distance).all() or np.any(edge_distance <= 0):
                raise ValueError("bond distances are not finite and positive")
            edge_distance = edge_distance[:, None]
        else:
            edge_distance = np.zeros((0, 1), dtype=np.float32)

        wedge_angle_cos = np.zeros((wedge_count, 1), dtype=np.float32)
        if wedge_count:
            first = wedge_array[:, 0]
            second = wedge_array[:, 1]
            left = edge_array[0, first]
            center = edge_array[1, first]
            right = edge_array[1, second]
            if not np.array_equal(center, edge_array[0, second]):
                raise ValueError("wedge center identity changed")
            vector_left = positions[left] - positions[center]
            vector_right = positions[right] - positions[center]
            denominator = np.linalg.norm(vector_left, axis=1) * np.linalg.norm(
                vector_right, axis=1
            )
            if np.any(denominator <= 0) or not np.isfinite(denominator).all():
                raise ValueError("wedge angle denominator is invalid")
            cosine = np.sum(vector_left * vector_right, axis=1) / denominator
            wedge_angle_cos[:, 0] = np.clip(cosine, -1.0, 1.0).astype(np.float32)
        if not np.isfinite(wedge_angle_cos).all():
            raise ValueError("wedge angle cosine is not finite")

        return GeometryResult(
            positions=positions,
            edge_distance=edge_distance,
            wedge_angle_cos=wedge_angle_cos,
            geometry_valid=True,
            mmff_converged=mmff_converged,
            embed_attempt=attempt,
        )
    except Exception as error:
        return _zero_result(
            int(node_count),
            edge_count,
            wedge_count,
            embed_attempt=attempt,
            error=error,
        )


def geometry_is_finite(result: GeometryResult) -> bool:
    """Cheap acceptance predicate used before a shard is published."""
    return all(
        np.isfinite(value).all()
        for value in (
            result.positions,
            result.edge_distance,
            result.wedge_angle_cos,
        )
    ) and math.isfinite(float(result.geometry_valid))
