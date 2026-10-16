"""Canonical identity compatibility for the accepted OGB silicon exceptions.

This is NOT a chemistry repair or a graph builder. The existing scale-cache
fallback preserves source topology with sanitize=False/strict=False. Here that
strategy is used only for identity, limited to the frozen, already audited
structures; an unknown structure never acquires an accepted identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pcqm_k1_scale import (
    UNSANITIZED_OGB_INDEX_SHA256, UNSANITIZED_OGB_SOURCE_INDICES, index_sha256,
)
from .research_memory.trace import file_digest


COMPAT_MOLECULE_RULE = "rdkit-canonical-isomeric-smiles-clear-atom-maps-known-ogb-si-v2"


def _key(mol):
    from rdkit import Chem
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _silicon_fallback(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("unparsable SMILES syntax")
    problems = Chem.DetectChemistryProblems(mol)
    if not problems or any(
        p.GetType() != "AtomValenceException" or
        mol.GetAtomWithIdx(p.GetAtomIdx()).GetAtomicNum() != 14
        for p in problems
    ):
        raise ValueError("not exclusively a known-class Si valence exception")
    # Exactly the accepted cache strategy; no charge/bond/H/stereo correction.
    mol.UpdatePropertyCache(strict=False)
    return mol


def load_fallback_keys(path, expected_sha256):
    """Pin the existing ten-row audit, not a newly invented permissive list."""
    path = Path(path)
    if file_digest(path) != expected_sha256:
        raise ValueError("ROLE_BLOCKED: fallback ledger SHA differs")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    entries = ledger["entries"]
    indices = [e["source_idx"] for e in entries]
    if (ledger.get("format") != "molgap-pcqm-k1-scale500k-parser-fallback-audit-v1" or
            tuple(indices) != UNSANITIZED_OGB_SOURCE_INDICES or
            index_sha256(indices) != UNSANITIZED_OGB_INDEX_SHA256 or
            ledger.get("ordered_index_sha256") != UNSANITIZED_OGB_INDEX_SHA256):
        raise ValueError("ROLE_BLOCKED: not the accepted OGB exception ledger")
    return frozenset(_key(_silicon_fallback(e["smiles"])) for e in entries)


def molecular_identity(smiles, *, fallback_keys):
    """Return (canonical key, anomaly evidence); no fallback flag in the key.

    Known graph-isomorphic aliases are covered too: restricting by source index
    or raw SMILES alone would miss duplicates across training/holdout roles.
    Unknown anomalies return None for a complete audit, NEVER a made-up key.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return _key(mol), None
    evidence = {"smiles_sha256": hashlib.sha256(smiles.encode()).hexdigest(),
                "status": "BLOCKED", "canonical_sha256": None}
    try:
        mol = _silicon_fallback(smiles)
        key = _key(mol)
        evidence["canonical_sha256"] = hashlib.sha256(key.encode()).hexdigest()
        if key not in fallback_keys:
            raise ValueError("Si structure absent from the frozen accepted ledger")
        # A second ordering and a serialization round trip must retain identity.
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        round_trip = Chem.MolFromSmiles(key, sanitize=False)
        if round_trip is None:
            raise ValueError("canonical identity cannot round trip")
        round_trip.UpdatePropertyCache(strict=False)
        if (_key(reordered) != key or _key(round_trip) != key or
                (round_trip.GetNumAtoms(), round_trip.GetNumBonds()) !=
                (mol.GetNumAtoms(), mol.GetNumBonds())):
            raise ValueError("unstable fallback molecular identity")
        evidence.update(status="KNOWN_FALLBACK", reason="accepted OGB Si topology; no chemistry edits")
        return key, evidence
    except ValueError as exc:
        evidence["reason"] = str(exc)
        return None, evidence
