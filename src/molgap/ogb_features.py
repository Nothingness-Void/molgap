"""Dependency-free OGB molecular categorical feature contract.

The categories match ``ogb.utils.features``. Keeping the small contract here
lets remote jobs build official PCQM graphs without installing the full OGB
package while preserving byte-for-byte feature indices.
"""
from __future__ import annotations

from rdkit import Chem


ATOM_FEATURE_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_FEATURE_DIMS = (5, 6, 2)

_CHIRALITY = (
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
    "misc",
)
_DEGREE = (*range(11), "misc")
_FORMAL_CHARGE = (*range(-5, 6), "misc")
_NUM_H = (*range(9), "misc")
_RADICAL_ELECTRONS = (*range(5), "misc")
_HYBRIDIZATION = ("SP", "SP2", "SP3", "SP3D", "SP3D2", "misc")
_BOND_TYPE = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc")
_BOND_STEREO = (
    "STEREONONE",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
    "STEREOANY",
)


def _safe_index(values: tuple, value: object) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values) - 1


def atom_to_ogb_feature_vector(atom: Chem.Atom) -> list[int]:
    atomic_number = atom.GetAtomicNum()
    return [
        atomic_number - 1 if 1 <= atomic_number <= 118 else 118,
        _safe_index(_CHIRALITY, str(atom.GetChiralTag())),
        _safe_index(_DEGREE, atom.GetTotalDegree()),
        _safe_index(_FORMAL_CHARGE, atom.GetFormalCharge()),
        _safe_index(_NUM_H, atom.GetTotalNumHs()),
        _safe_index(_RADICAL_ELECTRONS, atom.GetNumRadicalElectrons()),
        _safe_index(_HYBRIDIZATION, str(atom.GetHybridization())),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]


def bond_to_ogb_feature_vector(bond: Chem.Bond) -> list[int]:
    return [
        _safe_index(_BOND_TYPE, str(bond.GetBondType())),
        _BOND_STEREO.index(str(bond.GetStereo())),
        int(bond.GetIsConjugated()),
    ]
