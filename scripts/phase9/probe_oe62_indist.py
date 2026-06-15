"""
Phase 9 (P9.1): probe how many OE62 GW molecules fall inside our training
distribution — i.e. how much clean Δ-learning fuel we actually have.

Δ-learning target = GW gas-phase HOMO/LUMO (OE62 df_5k, 5239 molecules). But our
B3LYP-surrogate model is only trustworthy inside its training distribution
(elements ⊆ {C,H,N,O,S,F,Cl}, MW 200-1000). Molecules outside it would give a
dirty baseline, so the usable Δ training set is OE62 ∩ training-distribution.
This script counts that intersection and characterizes what gets dropped/why.

Input: OE62 dataframe JSON (df_5k.json, or df_62k.json — GW rows auto-filtered).
       Get it from NOMAD DOI 10.17172/NOMAD/2019.12.10-8 (see --help).

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/probe_oe62_indist.py --oe62-json PATH
  .venv\\Scripts\\python.exe scripts/phase9/probe_oe62_indist.py --self-test
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from molgap.constants import RESULTS_DIR

OUT = RESULTS_DIR / "phase9" / "oe62_indist.json"

# Same in-distribution box as the training set (see scripts/phase8 + fetch_300k.py).
ALLOWED_ELEMENTS = {"C", "H", "O", "N", "S", "F", "Cl"}
MW_MIN, MW_MAX = 200.0, 1000.0

# GW orbital-energy columns, best basis first (qzvp > tzvp). HOMO = max(occ),
# LUMO = min(unocc); both are lists of eV eigenvalues per molecule.
GW_OCC_COLS = ["energies_occ_gw_qzvp", "energies_occ_gw_tzvp"]
GW_UNOCC_COLS = ["energies_unocc_gw_qzvp", "energies_unocc_gw_tzvp"]


def _as_list(x):
    """OE62 stores eigenvalue arrays as lists; tolerate None/NaN/scalars."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if np.isscalar(x):
        return [float(x)]
    arr = list(x)
    return arr if len(arr) else None


def gw_homo_lumo(row):
    """(homo, lumo) GW eigenvalues in eV from the best available basis, or None."""
    occ = next((_as_list(row[c]) for c in GW_OCC_COLS if c in row and _as_list(row[c])), None)
    unocc = next((_as_list(row[c]) for c in GW_UNOCC_COLS if c in row and _as_list(row[c])), None)
    if occ is None or unocc is None:
        return None
    return max(occ), min(unocc)  # HOMO highest occupied, LUMO lowest unoccupied


def molecule_elements(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    elements = {a.GetSymbol() for a in mol.GetAtoms()}  # implicit H not listed
    if any(a.GetTotalNumHs() > 0 for a in mol.GetAtoms()):
        elements.add("H")
    return elements, Descriptors.MolWt(mol)


def main():
    ap = argparse.ArgumentParser(
        description="Probe OE62 GW ∩ training distribution.",
        epilog="OE62 df_5k.json: NOMAD DOI 10.17172/NOMAD/2019.12.10-8 "
               "(https://nomad-lab.eu/prod/v1/gui/dataset/doi/10.17172/NOMAD/2019.12.10-8). "
               "Load format: pd.read_json(path, orient='split').")
    ap.add_argument("--oe62-json", help="path to OE62 df_5k.json (or df_62k.json)")
    ap.add_argument("--self-test", action="store_true",
                    help="run logic on a tiny synthetic OE62-schema frame")
    args = ap.parse_args()

    if args.self_test:
        df = _synthetic_frame()
        print("SELF-TEST on synthetic OE62-schema frame\n")
    else:
        if not args.oe62_json:
            ap.error("--oe62-json is required (or use --self-test)")
        df = pd.read_json(args.oe62_json, orient="split")
        print(f"Loaded {len(df)} rows from {args.oe62_json}\n")

    n_total = len(df)
    n_gw, n_smiles_ok, n_elem_ok, n_mw_ok, n_indist = 0, 0, 0, 0, 0
    rejected_elements = Counter()
    indist_gaps = []

    for _, row in df.iterrows():
        hl = gw_homo_lumo(row)
        if hl is None:
            continue
        n_gw += 1
        homo, lumo = hl

        smiles = row.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        elements, mw = molecule_elements(smiles)
        if elements is None:
            continue
        n_smiles_ok += 1

        extra = elements - ALLOWED_ELEMENTS
        elem_ok = not extra
        mw_ok = MW_MIN <= mw <= MW_MAX
        if elem_ok:
            n_elem_ok += 1
        else:
            for el in extra:
                rejected_elements[el] += 1
        if mw_ok:
            n_mw_ok += 1
        if elem_ok and mw_ok:
            n_indist += 1
            indist_gaps.append(lumo - homo)

    report = {
        "input": "self-test" if args.self_test else args.oe62_json,
        "n_total_rows": n_total,
        "n_with_gw": n_gw,
        "n_smiles_parsed": n_smiles_ok,
        "n_pass_elements": n_elem_ok,
        "n_pass_mw": n_mw_ok,
        "n_in_distribution": n_indist,
        "rejected_by_element": dict(rejected_elements.most_common()),
        "indist_gw_gap_eV": {
            "min": float(np.min(indist_gaps)) if indist_gaps else None,
            "p50": float(np.percentile(indist_gaps, 50)) if indist_gaps else None,
            "max": float(np.max(indist_gaps)) if indist_gaps else None,
        },
    }

    print(f"  total rows           {n_total}")
    print(f"  with GW values       {n_gw}")
    print(f"  SMILES parsed        {n_smiles_ok}")
    print(f"  pass element filter  {n_elem_ok}")
    print(f"  pass MW filter       {n_mw_ok}")
    print(f"  IN-DISTRIBUTION      {n_indist}   <-- clean Δ training pairs")
    if rejected_elements:
        print("  rejected by element (molecules containing each foreign element):")
        for el, c in rejected_elements.most_common():
            print(f"    {el:3s} {c}")
    if indist_gaps:
        g = report["indist_gw_gap_eV"]
        print(f"  in-dist GW gap: {g['min']:.2f} … {g['max']:.2f} eV (median {g['p50']:.2f})")

    if not args.self_test:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2))
        print(f"\nSaved {OUT}")


def _synthetic_frame():
    """Tiny frame mimicking OE62 schema, to verify filter + extraction logic.

    5 rows: 2 clean in-dist, 1 with Br (foreign element), 1 too-light MW,
    1 with no GW data. Expected in-distribution count = 2.
    """
    return pd.DataFrame([
        {  # in-dist: CBP-like, C/H/N only, MW ~370
            "canonical_smiles": "c1ccc(-n2c3ccccc3c3ccccc32)cc1",
            "energies_occ_gw_qzvp": [-12.1, -8.3, -6.2], "energies_unocc_gw_qzvp": [-1.1, 2.0],
            "energies_occ_gw_tzvp": [-12.0, -8.2, -6.1], "energies_unocc_gw_tzvp": [-1.0, 2.1],
        },
        {  # in-dist: anthracene, C/H, MW ~178 -> actually too light, expect MW fail
            "canonical_smiles": "c1ccc2cc3ccccc3cc2c1",
            "energies_occ_gw_qzvp": [-9.5, -6.0], "energies_unocc_gw_qzvp": [-1.5, 1.2],
            "energies_occ_gw_tzvp": None, "energies_unocc_gw_tzvp": None,
        },
        {  # foreign element Br -> element fail
            "canonical_smiles": "Brc1ccc(-c2ccccc2)cc1",
            "energies_occ_gw_qzvp": [-9.0, -6.5], "energies_unocc_gw_qzvp": [-1.2, 1.0],
            "energies_occ_gw_tzvp": None, "energies_unocc_gw_tzvp": None,
        },
        {  # in-dist: larger C/H/N/O, MW ~300
            "canonical_smiles": "O=C1c2ccccc2C(=O)c2cc3c(cc21)C(=O)c1ccccc1C3=O",
            "energies_occ_gw_qzvp": [-10.0, -7.0], "energies_unocc_gw_qzvp": [-3.0, 0.5],
            "energies_occ_gw_tzvp": None, "energies_unocc_gw_tzvp": None,
        },
        {  # no GW data -> dropped before filters
            "canonical_smiles": "c1ccccc1",
            "energies_occ_gw_qzvp": None, "energies_unocc_gw_qzvp": None,
            "energies_occ_gw_tzvp": None, "energies_unocc_gw_tzvp": None,
        },
    ])


if __name__ == "__main__":
    main()
