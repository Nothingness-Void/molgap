"""
Phase 8.2: design the broader-coverage sampling spec.

Reads the Phase 7 300k CSV, computes lightweight RDKit/topology descriptors, and
turns P8.1's qualitative coverage gaps into concrete top-up quotas. This does
not fetch molecules or build graphs; it defines what the next PubChemQC refetch
should target.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/design_sampling_spec.py
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/design_sampling_spec.py --topup-size 300000
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/design_sampling_spec.py --max-rows 20000
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.utils import canonicalize_smiles, ensure_dirs

RDLogger.DisableLog("rdApp.*")

TRAIN_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
OUT_DIR = RESULTS_DIR / "phase8"
DESC_CACHE = OUT_DIR / "training_gap_descriptors.csv"
SPEC_JSON = OUT_DIR / "sampling_spec.json"
SPEC_MD = OUT_DIR / "sampling_spec.md"

ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "F", "Cl"}
TARGET_COLS = ("homo", "lumo", "gap")


def formula_elements(formula: object) -> set[str]:
    if not isinstance(formula, str):
        return set()
    elements: set[str] = set()
    i, n = 0, len(formula)
    while i < n:
        c = formula[i]
        if c.isupper():
            sym = c
            i += 1
            while i < n and formula[i].islower():
                sym += formula[i]
                i += 1
            elements.add(sym)
        else:
            i += 1
    return elements


def descriptor_row(item: tuple[int, str, str]) -> dict[str, float | int | str | None]:
    idx, smiles, formula = item
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {"row_idx": idx, "valid_rdkit": 0}

    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    heavy = mol.GetNumHeavyAtoms()
    n_arom_atoms = sum(1 for atom in atoms if atom.GetIsAromatic())
    elements = {atom.GetSymbol() for atom in atoms} or formula_elements(formula)
    formal_charge = sum(atom.GetFormalCharge() for atom in atoms)
    fragments = len(Chem.GetMolFrags(mol))
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    conjugated_bonds = sum(1 for bond in bonds if bond.GetIsConjugated())
    aromatic_atom_fraction = n_arom_atoms / heavy if heavy else 0.0

    return {
        "row_idx": idx,
        "valid_rdkit": 1,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "rdkit_mw": float(Descriptors.MolWt(mol)),
        "heavy_atoms": int(heavy),
        "fragments": int(fragments),
        "formal_charge": int(formal_charge),
        "ring_count": int(ring_count),
        "aromatic_rings": int(aromatic_rings),
        "aromatic_atom_fraction": float(aromatic_atom_fraction),
        "rotatable_bonds": int(rotatable_bonds),
        "conjugated_bonds": int(conjugated_bonds),
        "frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "has_s": int("S" in elements),
        "has_cl": int("Cl" in elements),
        "has_f": int("F" in elements),
        "has_n": int("N" in elements),
        "has_o": int("O" in elements),
        "has_s_or_cl": int(("S" in elements) or ("Cl" in elements)),
        "allowed_elements": int(elements.issubset(ALLOWED_ELEMENTS)),
    }


def compute_or_load_descriptors(df: pd.DataFrame, max_rows: int | None, n_jobs: int) -> pd.DataFrame:
    if max_rows is None and DESC_CACHE.exists():
        cached = pd.read_csv(DESC_CACHE)
        if len(cached) == len(df):
            print(f"Reusing descriptor cache: {DESC_CACHE} ({len(cached)} rows)")
            return cached
        print(f"Ignoring stale cache: {len(cached)} rows != {len(df)}")

    work = df if max_rows is None else df.head(max_rows).copy()
    items = list(zip(work.index.tolist(), work["smiles"].astype(str), work["formula"].astype(str)))
    rows: list[dict[str, float | int | str | None]] = []
    if n_jobs <= 1:
        for item in tqdm(items, desc="RDKit descriptors", unit="mol"):
            rows.append(descriptor_row(item))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for row in tqdm(
                ex.map(descriptor_row, items, chunksize=512),
                total=len(items), desc="RDKit descriptors", unit="mol",
            ):
                rows.append(row)

    desc = pd.DataFrame(rows).sort_values("row_idx").reset_index(drop=True)
    if max_rows is None:
        ensure_dirs(DESC_CACHE.parent)
        desc.to_csv(DESC_CACHE, index=False, encoding="utf-8")
        print(f"Saved descriptor cache: {DESC_CACHE}")
    return desc


def pct(n: int | float, total: int) -> float:
    return float(n) / float(total) if total else 0.0


def count_mask(mask: pd.Series | np.ndarray, total: int) -> dict[str, float | int]:
    n = int(np.asarray(mask).sum())
    return {"n": n, "fraction": pct(n, total)}


def numeric_bin_counts(series: pd.Series, bins: list[float], labels: list[str]) -> list[dict[str, float | int | str]]:
    cats = pd.cut(series.astype(float), bins=bins, labels=labels, include_lowest=True, right=False)
    out = []
    total = int(series.notna().sum())
    vc = cats.value_counts(sort=False)
    for label in labels:
        n = int(vc.get(label, 0))
        out.append({"bin": label, "n": n, "fraction": pct(n, total)})
    return out


def quota_needed(current_n: int, current_total: int, desired_final_fraction: float, topup_size: int) -> int:
    final_total = current_total + topup_size
    desired_final_n = int(math.ceil(desired_final_fraction * final_total))
    return max(0, desired_final_n - current_n)


def build_priority_buckets(x: pd.DataFrame, topup_size: int) -> list[dict[str, object]]:
    """Priority buckets for the next fetch. Buckets may overlap conceptually.

    The intended fetcher should assign a candidate to the first matching bucket
    with remaining quota, so the quotas sum to the requested top-up size.
    """
    return [
        {
            "id": "very_low_gap",
            "quota": int(round(topup_size * 0.15)),
            "predicate": "gap < 2.5",
            "why": "Directly fills the lowest-gap B3LYP tail; current v1 has very few.",
            "current": count_mask(x["gap"] < 2.5, len(x)),
        },
        {
            "id": "low_gap_aromatic_edge",
            "quota": int(round(topup_size * 0.20)),
            "predicate": "2.5 <= gap < 3.2 and (aromatic_rings >= 5 or aromatic_atom_fraction >= 0.85)",
            "why": "OLED-like / charge-transfer-like region where v1 and B3LYP are weakest.",
            "current": count_mask(
                (x["gap"] >= 2.5) & (x["gap"] < 3.2) & x["aromatic_edge"], len(x)
            ),
        },
        {
            "id": "large_aromatic_edge",
            "quota": int(round(topup_size * 0.13)),
            "predicate": "mw >= 500 and (aromatic_rings >= 5 or aromatic_atom_fraction >= 0.85)",
            "why": "Fills the sparse large/conjugated corner most relevant to materials molecules.",
            "current": count_mask((x["mw"] >= 500) & x["aromatic_edge"], len(x)),
        },
        {
            "id": "very_large_general",
            "quota": int(round(topup_size * 0.10)),
            "predicate": "mw >= 700",
            "why": "Phase 7 p99 MW is ~709 despite a 1000 Da allowed max.",
            "current": count_mask(x["mw"] >= 700, len(x)),
        },
        {
            "id": "s_or_cl_hard",
            "quota": int(round(topup_size * 0.10)),
            "predicate": "(has_s or has_cl) and (gap < 3.5 or aromatic_rings >= 4 or aromatic_atom_fraction >= 0.70)",
            "why": "Keeps S/Cl coverage in the hard chemistry rather than only easy molecules; overlaps with low-gap/large buckets.",
            "current": count_mask(
                (x["has_s_or_cl"] == 1)
                & ((x["gap"] < 3.5) | (x["aromatic_rings"] >= 4) | (x["aromatic_atom_fraction"] >= 0.70)),
                len(x),
            ),
        },
        {
            "id": "aromatic_edge_general",
            "quota": int(round(topup_size * 0.09)),
            "predicate": "gap >= 3.2 and (aromatic_rings >= 5 or aromatic_atom_fraction >= 0.85)",
            "why": "Adds high-aromatic edge cases not already captured by low-gap or large buckets.",
            "current": count_mask((x["gap"] >= 3.2) & x["aromatic_edge"], len(x)),
        },
        {
            "id": "flexible_hard",
            "quota": topup_size - int(round(topup_size * (0.15 + 0.20 + 0.18 + 0.13 + 0.10 + 0.10 + 0.09))),
            "predicate": "rotatable_bonds >= 8 and (gap < 3.5 or aromatic_rings >= 4)",
            "why": "Targets the flexible donor-like cases where 2D/3D ranking flips.",
            "current": count_mask((x["rotatable_bonds"] >= 8) & ((x["gap"] < 3.5) | (x["aromatic_rings"] >= 4)), len(x)),
        },
        {
            "id": "large_mw_500_700",
            "quota": int(round(topup_size * 0.18)),
            "predicate": "500 <= mw < 700",
            "why": "Moves mass out of the overrepresented 200-500 Da region without only chasing extremes.",
            "current": count_mask((x["mw"] >= 500) & (x["mw"] < 700), len(x)),
        },
    ]


def write_markdown(spec: dict, path: Path) -> None:
    lines = [
        "# P8.2 Sampling Spec",
        "",
        "Purpose: fill sparse Phase 7 training regions before retraining a v2 B3LYP base.",
        "",
        f"- current training rows analyzed: {spec['n_analyzed']:,}",
        f"- planned targeted top-up: {spec['topup_size']:,}",
        "- keep Phase 7 300k as base distribution; do not redraw another same-source 300k",
        "- hard filters: elements subset of C/H/N/O/S/F/Cl, MW 200-1000, gap > 0, exclude Phase 7 CIDs/canonical SMILES",
        "",
        "## Current Gap Summary",
        "",
        "| region | n | fraction |",
        "|---|---:|---:|",
    ]
    for key, value in spec["coverage_flags"].items():
        lines.append(f"| {key} | {value['n']:,} | {value['fraction']:.3%} |")

    lines += [
        "",
        "## Priority Fetch Buckets",
        "",
        "Assign each fetched candidate to the first matching bucket with remaining quota.",
        "",
        "| priority | bucket | quota | current n | current fraction | predicate |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for i, b in enumerate(spec["priority_buckets"], 1):
        cur = b["current"]
        lines.append(
            f"| {i} | `{b['id']}` | {b['quota']:,} | {cur['n']:,} | "
            f"{cur['fraction']:.3%} | `{b['predicate']}` |"
        )

    lines += [
        "",
        "## Axis-Level Desired Coverage",
        "",
        "These are diagnostics for the final old+topup pool; priority bucket quotas above are the executable fetch plan.",
        "",
        "| axis | current n | current fraction | desired final fraction | needed in top-up |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in spec["axis_targets"]:
        lines.append(
            f"| {row['id']} | {row['current_n']:,} | {row['current_fraction']:.3%} | "
            f"{row['desired_final_fraction']:.1%} | {row['needed_topup']:,} |"
        )

    lines += [
        "",
        "## Next Step",
        "",
        "Implement a targeted PubChemQC skim/fetcher that computes these cheap descriptors before graph building,",
        "fills the priority buckets, writes a slim CSV, then holds out scaffold-disjoint hard eval slices per bucket.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--topup-size", type=int, default=200_000)
    ap.add_argument("--max-rows", type=int, default=0, help="debug: analyze first N rows only")
    ap.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    ensure_dirs(OUT_DIR)
    df = pd.read_csv(args.train_csv)
    for col in ("mw", *TARGET_COLS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["smiles", "mw", *TARGET_COLS]).reset_index(drop=True)
    max_rows = args.max_rows if args.max_rows > 0 else None
    if max_rows is not None:
        df = df.head(max_rows).copy()

    print(f"Analyzing {len(df):,} rows from {args.train_csv}")
    desc = compute_or_load_descriptors(df, max_rows=max_rows, n_jobs=args.n_jobs)
    merged = pd.concat([df.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
    merged = merged[merged["valid_rdkit"] == 1].copy()
    merged["high_conjugation"] = (
        (merged["aromatic_rings"] >= 5)
        | (merged["aromatic_atom_fraction"] >= 0.80)
        | (merged["conjugated_bonds"] >= 24)
    )
    merged["aromatic_edge"] = (
        (merged["aromatic_rings"] >= 5)
        | (merged["aromatic_atom_fraction"] >= 0.85)
    )

    total = len(merged)
    flags = {
        "gap_lt_2p5": count_mask(merged["gap"] < 2.5, total),
        "gap_lt_3": count_mask(merged["gap"] < 3.0, total),
        "gap_3_to_4": count_mask((merged["gap"] >= 3.0) & (merged["gap"] < 4.0), total),
        "high_conjugation": count_mask(merged["high_conjugation"], total),
        "aromatic_rings_ge_5": count_mask(merged["aromatic_rings"] >= 5, total),
        "aromatic_fraction_ge_0p8": count_mask(merged["aromatic_atom_fraction"] >= 0.80, total),
        "aromatic_edge": count_mask(merged["aromatic_edge"], total),
        "mw_ge_500": count_mask(merged["mw"] >= 500, total),
        "mw_ge_700": count_mask(merged["mw"] >= 700, total),
        "has_s": count_mask(merged["has_s"] == 1, total),
        "has_cl": count_mask(merged["has_cl"] == 1, total),
        "has_s_or_cl_hard": count_mask(
            (merged["has_s_or_cl"] == 1)
            & ((merged["gap"] < 3.5) | (merged["aromatic_rings"] >= 4) | (merged["aromatic_atom_fraction"] >= 0.70)),
            total,
        ),
        "flexible_hard": count_mask(
            (merged["rotatable_bonds"] >= 8) & ((merged["gap"] < 3.5) | (merged["aromatic_rings"] >= 4)),
            total,
        ),
    }

    desired = [
        ("gap_lt_3", flags["gap_lt_3"]["n"], 0.08),
        ("gap_lt_2p5", flags["gap_lt_2p5"]["n"], 0.03),
        ("high_conjugation", flags["high_conjugation"]["n"], 0.10),
        ("aromatic_rings_ge_5", flags["aromatic_rings_ge_5"]["n"], 0.06),
        ("aromatic_fraction_ge_0p8", flags["aromatic_fraction_ge_0p8"]["n"], 0.06),
        ("mw_ge_500", flags["mw_ge_500"]["n"], 0.18),
        ("mw_ge_700", flags["mw_ge_700"]["n"], 0.05),
        ("has_s_or_cl_hard", flags["has_s_or_cl_hard"]["n"], 0.12),
        ("flexible_hard", flags["flexible_hard"]["n"], 0.08),
    ]
    axis_targets = []
    for name, cur_n, frac in desired:
        cur_n = int(cur_n)
        axis_targets.append({
            "id": name,
            "current_n": cur_n,
            "current_fraction": pct(cur_n, total),
            "desired_final_fraction": frac,
            "needed_topup": quota_needed(cur_n, total, frac, args.topup_size),
        })

    spec = {
        "source_csv": str(args.train_csv),
        "n_analyzed": total,
        "topup_size": args.topup_size,
        "hard_filters": {
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "mw_min": 200,
            "mw_max": 1000,
            "gap_positive": True,
            "exclude_existing_phase7_cids_and_canonical_smiles": True,
        },
        "coverage_flags": flags,
        "bin_counts": {
            "gap": numeric_bin_counts(
                merged["gap"], [0, 2.5, 3.0, 4.0, 5.5, 20],
                ["<2.5", "2.5-3.0", "3.0-4.0", "4.0-5.5", ">=5.5"],
            ),
            "mw": numeric_bin_counts(
                merged["mw"], [200, 300, 500, 700, 1000.0001],
                ["200-300", "300-500", "500-700", "700-1000"],
            ),
            "aromatic_rings": numeric_bin_counts(
                merged["aromatic_rings"], [0, 1, 3, 5, 7, 100],
                ["0", "1-2", "3-4", "5-6", ">=7"],
            ),
            "aromatic_atom_fraction": numeric_bin_counts(
                merged["aromatic_atom_fraction"], [0, 0.25, 0.50, 0.75, 0.85, 1.0001],
                ["<0.25", "0.25-0.50", "0.50-0.75", "0.75-0.85", ">=0.85"],
            ),
        },
        "axis_targets": axis_targets,
        "priority_buckets": build_priority_buckets(merged, args.topup_size),
        "recommended_training_pool": {
            "keep_phase7_300k": True,
            "targeted_topup_size": args.topup_size,
            "final_pool_size_before_graph_failures": total + args.topup_size,
            "use_balanced_or_weighted_batches": True,
            "holdout_per_bucket_before_training": True,
        },
    }

    SPEC_JSON.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    write_markdown(spec, SPEC_MD)
    print(f"Saved {SPEC_JSON}")
    print(f"Saved {SPEC_MD}")
    print("\nPriority buckets:")
    for b in spec["priority_buckets"]:
        cur = b["current"]
        print(f"  {b['id']:<26s} quota={b['quota']:>6,} current={cur['n']:>7,} ({cur['fraction']:.2%})")


if __name__ == "__main__":
    main()
