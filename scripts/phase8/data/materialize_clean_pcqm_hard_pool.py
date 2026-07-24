"""Filter the raw PCQM hard pool to the fixed closed-shell training domain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_parquet(args.input)
    keep, reasons = [], {
        "invalid": 0,
        "disconnected": 0,
        "radical": 0,
        "heavy_atoms_lt_5": 0,
        "noble_gas": 0,
        "gap_gt_12_eV": 0,
    }
    for smiles, gap in zip(table.smiles, table.homolumogap, strict=True):
        mol = Chem.MolFromSmiles(str(smiles))
        flags = {
            "invalid": mol is None,
            "disconnected": "." in str(smiles),
            "radical": False
            if mol is None
            else sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) > 0,
            "heavy_atoms_lt_5": mol is None or mol.GetNumHeavyAtoms() < 5,
            "noble_gas": False
            if mol is None
            else any(
                atom.GetAtomicNum() in (2, 10, 18, 36, 54, 86)
                for atom in mol.GetAtoms()
            ),
            "gap_gt_12_eV": gap > 12.0,
        }
        for name, flag in flags.items():
            reasons[name] += int(flag)
        keep.append(not any(flags.values()))
    clean = table.loc[keep].reset_index(drop=True)
    if clean.canonical_smiles.nunique() != len(clean):
        raise RuntimeError("Clean pool is not canonical-SMILES unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    clean.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, args.output)
    report = {
        "status": "complete",
        "input_rows": len(table),
        "output_rows": len(clean),
        "excluded_rows": len(table) - len(clean),
        "overlapping_exclusion_counts": reasons,
        "filter": "valid and connected and closed-shell and heavy_atoms>=5 and no_noble_gas and gap<=12_eV",
        "output_sha256": sha256_file(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
