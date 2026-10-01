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
    descriptor_valid_mask: np.ndarray | None = None
    smiles_sanitized: bool | None = None


class ChemicalLabelEncoder:
    """Reuse normalized Descriptastorus descriptors and RDKit path fingerprints.

    Invalid rows remain explicit. No descriptor clipping, imputation, fitting,
    canonicalization, or conversion to a different fingerprint is performed.
    """

    def __init__(self, *, components=("descriptors", "fingerprints"),
                 parse_policy="strict", descriptor_missing_policy="reject"):
        if (not components or len(set(components)) != len(components) or
            not set(components) <= {"descriptors", "fingerprints"}):
            raise ValueError("Choose descriptors, fingerprints, or both")
        if parse_policy not in {"strict", "pcqm_topology"}:
            raise ValueError("Unsupported molecular parsing policy")
        if descriptor_missing_policy not in {"reject", "mask_nonfinite"}:
            raise ValueError("Unsupported descriptor missing policy")
        if "descriptors" not in components and descriptor_missing_policy != "reject":
            raise ValueError("Descriptor missing policy requires descriptors")
        self.components = tuple(c for c in ("descriptors", "fingerprints") if c in components)
        self.parse_policy = parse_policy
        self.descriptor_missing_policy = descriptor_missing_policy
        self.generator = None
        self.descriptor_names = ()
        if "descriptors" not in self.components:
            return
        from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

        self.generator = RDKit2DNormalized()
        self.descriptor_names = tuple(name for name, _ in self.generator.GetColumns()[1:])
        if len(self.descriptor_names) != 200:
            raise RuntimeError("Expected exactly 200 named normalized descriptors")
        if descriptor_missing_policy == "mask_nonfinite":
            # The upstream normalizer converts calculation exceptions to 0.0.
            # Reuse its functions and CDFs, but retain failed cells as missing.
            # This instance-local hook leaves the legacy generator unchanged.
            self.generator.calculateMol = (
                lambda mol, smiles, internalParsing=False:
                _normalized_descriptors_with_missingness(self.generator, mol)
            )

    def identity(self) -> dict:
        identity = {
            "descriptastorus": (importlib.metadata.version("descriptastorus")
                                if self.generator is not None else None),
            "rdkit": importlib.metadata.version("rdkit"),
            "descriptor_names": list(self.descriptor_names),
            "normalization": "RDKit2DNormalized library distributions; no PCQM fitting",
            "fingerprint": {"kind": "RDKFingerprint", "minPath": 1, "maxPath": 7, "fpSize": 512},
        }
        if (self.components, self.parse_policy, self.descriptor_missing_policy) != (
                ("descriptors", "fingerprints"), "strict", "reject"):
            identity.update(components=list(self.components), parse_policy=self.parse_policy,
                            descriptor_missing_policy=self.descriptor_missing_policy)
        if self.descriptor_missing_policy == "mask_nonfinite":
            identity["descriptor_failure_detection"] = "raw-and-cdf-failure-to-nan-v1"
        return identity

    def encode(self, source_index: int, smiles: str) -> ChemicalLabels:
        from rdkit import Chem, DataStructs

        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise ValueError("source_index must be a nonnegative integer")
        if not isinstance(smiles, str) or not smiles.strip():
            return ChemicalLabels(source_index, "invalid_smiles", None, None)
        sanitized = True
        if self.parse_policy == "pcqm_topology":
            # Use the existing PCQM topology policy; descriptor validity is
            # independently checked and never inferred from graph acceptance.
            from .pcqm_feature_screen import _molecule
            try:
                mol, sanitized = _molecule(smiles)
            except ValueError:
                mol = None
        else:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return ChemicalLabels(source_index, "invalid_smiles", None, None)
        try:
            descriptors = fingerprint = valid_mask = None
            if "descriptors" in self.components:
                raw = (self.generator.process(smiles) if sanitized else
                       self.generator.processMol(mol, smiles))
                if raw is None or len(raw) != 201 or not raw[0]:
                    return ChemicalLabels(source_index, "descriptor_failure", None, None)
                descriptors = np.asarray(raw[1:], dtype=np.float32)
                finite = np.isfinite(descriptors)
                if not finite.all() and self.descriptor_missing_policy == "reject":
                    return ChemicalLabels(source_index, "nonfinite_descriptor", None, None)
                if self.descriptor_missing_policy == "mask_nonfinite":
                    valid_mask = finite
                    # These are storage placeholders, excluded from the loss.
                    descriptors = np.where(finite, descriptors, 0).astype(np.float32)
            if "fingerprints" in self.components:
                fingerprint = np.zeros(512, dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(
                    Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512), fingerprint
                )
        except Exception as exc:
            return ChemicalLabels(source_index, "label_error:" + type(exc).__name__, None, None)
        return ChemicalLabels(source_index, "ok", descriptors, fingerprint, valid_mask, sanitized)


def _normalized_descriptors_with_missingness(generator, molecule) -> list[float]:
    """Apply the owning library's exact CDFs without its failed-cell defaults."""
    from descriptastorus.descriptors.rdNormalizedDescriptors import cdfs

    values = []
    for name, _ in generator.columns:
        try:
            raw = generator.funcs[name](molecule)
            if raw is None or not np.isfinite(raw):
                raise ValueError("Missing raw descriptor")
            normalized = cdfs[name](raw)
            values.append(float(normalized) if np.isfinite(normalized) else float("nan"))
        except Exception:
            values.append(float("nan"))
    return values


def compact_head_parameter_count(input_dim: int = 288, hidden_dim: int = 32) -> int:
    """Two biased linear layers sharing one bottleneck for 200+512 outputs."""
    if input_dim <= 0 or hidden_dim <= 0:
        raise ValueError("head dimensions must be positive")
    return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * 712
