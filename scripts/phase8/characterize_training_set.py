"""
Phase 7: characterize the 300k training set's chemical space.

Defines what "in-distribution" means so we can later screen commercial molecules:
only molecules that look like the training set get trusted (high-confidence)
predictions. Answers the user's core question — "what is my training set, and
which commercial molecules share its elements / size / topology?"

Outputs:
  results/phase8/training_space.json  — element coverage, MW + descriptor ranges,
                                         label ranges, the in-distribution "box"

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/characterize_training_set.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from molgap.constants import RESULTS_DIR

TRAIN_CSV = "data/raw/phase7_chonsfcl_mw200_1000_300k.csv"
OUT = RESULTS_DIR / "phase8" / "training_space.json"

# Elements the training set was filtered to (see fetch_300k.py).
TRAIN_ELEMENTS = {"C", "H", "O", "N", "S", "F", "Cl"}
# How many molecules to sample for the (slower) RDKit topology descriptors.
RDKIT_SAMPLE = 50_000
SEED = 42


def parse_formula_elements(formula: str) -> set[str]:
    """Element symbols present in a Hill-style molecular formula."""
    elements, i, n = set(), 0, len(formula)
    while i < n:
        c = formula[i]
        if c.isupper():
            sym, i = c, i + 1
            while i < n and formula[i].islower():
                sym, i = sym + formula[i], i + 1
            elements.add(sym)
        else:
            i += 1
    return elements


def pct_box(arr, lo=1.0, hi=99.0):
    """In-distribution range as [1st, 99th] percentile (robust to outliers)."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "min": float(a.min()), "max": float(a.max()),
        "p1": float(np.percentile(a, lo)), "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, hi)), "mean": float(a.mean()),
    }


def main():
    df = pd.read_csv(TRAIN_CSV)
    n = len(df)
    print(f"Loaded {n} training molecules from {TRAIN_CSV}\n")

    report = {"n": n, "elements_filter": sorted(TRAIN_ELEMENTS)}

    # ── Element coverage (full set, fast: parse the formula column) ──
    print("Element coverage (fraction of molecules containing each element):")
    elem_counts: dict[str, int] = {}
    for formula in df["formula"].astype(str):
        for el in parse_formula_elements(formula):
            elem_counts[el] = elem_counts.get(el, 0) + 1
    report["element_fraction"] = {
        el: round(c / n, 4) for el, c in sorted(elem_counts.items(), key=lambda x: -x[1])
    }
    for el, frac in report["element_fraction"].items():
        flag = "" if el in TRAIN_ELEMENTS else "  <-- UNEXPECTED (not in filter)"
        print(f"  {el:3s} {frac*100:6.2f}%{flag}")

    # ── MW + labels (full set, from existing columns) ──
    report["mw"] = pct_box(df["mw"])
    report["labels_eV"] = {t: pct_box(df[t]) for t in ("homo", "lumo", "gap")}
    print(f"\nMW: {report['mw']['min']:.0f}–{report['mw']['max']:.0f} "
          f"(p1–p99: {report['mw']['p1']:.0f}–{report['mw']['p99']:.0f})")
    for t in ("homo", "lumo", "gap"):
        b = report["labels_eV"][t]
        print(f"  {t:4s} {b['p1']:+.2f} … {b['p99']:+.2f} eV (median {b['p50']:+.2f})")

    # ── Topology descriptors (sampled, needs RDKit) ──
    rng = np.random.RandomState(SEED)
    samp = df.sample(n=min(RDKIT_SAMPLE, n), random_state=rng).reset_index(drop=True)
    print(f"\nComputing RDKit topology descriptors on {len(samp)} sampled molecules...")
    heavy, arom_rings, arom_frac, rot_bonds, rings = [], [], [], [], []
    for smi in tqdm(samp["smiles"].astype(str), unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        na = mol.GetNumAtoms()
        heavy.append(mol.GetNumHeavyAtoms())
        arom_rings.append(Descriptors.NumAromaticRings(mol))
        n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        arom_frac.append(n_arom / na if na else 0.0)
        rot_bonds.append(Descriptors.NumRotatableBonds(mol))
        rings.append(Descriptors.RingCount(mol))

    report["topology_sampled"] = {
        "sample_n": len(samp),
        "heavy_atoms": pct_box(heavy),
        "aromatic_rings": pct_box(arom_rings),
        "aromatic_atom_fraction": pct_box(arom_frac),
        "rotatable_bonds": pct_box(rot_bonds),
        "ring_count": pct_box(rings),
    }
    print("\nTopology (sampled, p1–p99):")
    for k in ("heavy_atoms", "aromatic_rings", "aromatic_atom_fraction",
              "rotatable_bonds", "ring_count"):
        b = report["topology_sampled"][k]
        print(f"  {k:24s} {b['p1']:6.2f} … {b['p99']:6.2f}  (median {b['p50']:.2f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
