"""CPU-only chemical supervision adapter; no model or evaluation-role access."""
from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChemicalLabels:
    source_index: int
    status: str
    descriptors: np.ndarray | None
    fingerprint: np.ndarray | None


class ChemicalLabelEncoder:
    """Reuse normalized Descriptastorus descriptors and RDKit path fingerprints.

    Invalid rows remain explicit. No descriptor clipping, imputation, fitting,
    canonicalization, or conversion to a different fingerprint is performed.
    """

    def __init__(self):
        from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

        self.generator = RDKit2DNormalized()
        self.descriptor_names = tuple(name for name, _ in self.generator.GetColumns()[1:])
        if len(self.descriptor_names) != 200:
            raise RuntimeError("Expected exactly 200 named normalized descriptors")

    def identity(self) -> dict:
        return {
            "descriptastorus": importlib.metadata.version("descriptastorus"),
            "rdkit": importlib.metadata.version("rdkit"),
            "descriptor_names": list(self.descriptor_names),
            "normalization": "RDKit2DNormalized library distributions; no PCQM fitting",
            "fingerprint": {"kind": "RDKFingerprint", "minPath": 1, "maxPath": 7, "fpSize": 512},
        }

    def encode(self, source_index: int, smiles: str) -> ChemicalLabels:
        from rdkit import Chem, DataStructs

        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise ValueError("source_index must be a nonnegative integer")
        if not isinstance(smiles, str) or not smiles.strip():
            return ChemicalLabels(source_index, "invalid_smiles", None, None)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return ChemicalLabels(source_index, "invalid_smiles", None, None)
        try:
            raw = self.generator.process(smiles)
            if raw is None or len(raw) != 201 or not raw[0]:
                return ChemicalLabels(source_index, "descriptor_failure", None, None)
            descriptors = np.asarray(raw[1:], dtype=np.float32)
            if not np.isfinite(descriptors).all():
                return ChemicalLabels(source_index, "nonfinite_descriptor", None, None)
            fingerprint = np.zeros(512, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(
                Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512), fingerprint
            )
        except Exception as exc:
            return ChemicalLabels(source_index, "label_error:" + type(exc).__name__, None, None)
        return ChemicalLabels(source_index, "ok", descriptors, fingerprint)


def compact_head_parameter_count(input_dim: int = 288, hidden_dim: int = 32) -> int:
    """Two biased linear layers sharing one bottleneck for 200+512 outputs."""
    if input_dim <= 0 or hidden_dim <= 0:
        raise ValueError("head dimensions must be positive")
    return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * 712
