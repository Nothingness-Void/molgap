"""Freeze a scaffold-disjoint 100K/10K/10K PubChemQC architecture split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.gap_specialization import scaffold_split
from molgap.router_sampling import compute_scaffold_keys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("data/raw/phase8_expansion_1m.csv")
    )
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/phase8/experiments/pubchemqc100k_architecture"),
    )
    args = parser.parse_args()

    columns = ["cid", "smiles", "homo", "lumo", "gap"]
    header = pd.read_csv(args.csv, nrows=0).columns
    if "canonical_smiles" in header:
        columns.append("canonical_smiles")
    frame = pd.read_csv(args.csv, usecols=columns)
    frame.insert(0, "source_idx", np.arange(len(frame), dtype=np.int64))
    finite = np.isfinite(frame[["homo", "lumo", "gap"]].to_numpy()).all(axis=1)
    smiles_column = (
        "canonical_smiles" if "canonical_smiles" in frame.columns else "smiles"
    )
    valid = finite & frame[smiles_column].notna().to_numpy()
    candidates = np.flatnonzero(valid)
    if len(candidates) < args.rows:
        raise ValueError(f"Only {len(candidates)} valid rows for {args.rows} requested")
    rng = np.random.default_rng(args.seed)
    selected_positions = np.sort(
        rng.choice(candidates, size=args.rows, replace=False)
    )
    selected = frame.iloc[selected_positions].copy().reset_index(drop=True)
    selected["canonical_smiles"] = selected[smiles_column].astype(str)

    roles = scaffold_split(
        selected.canonical_smiles.tolist(),
        seed=args.seed,
        workers=args.workers,
        train_fraction=100_000 / args.rows,
        validation_fraction=10_000 / args.rows,
    )
    selected["split"] = ""
    for role, indices in roles.items():
        selected.loc[indices, "split"] = role
    if selected.split.eq("").any():
        raise RuntimeError("Some selected rows were not assigned to a split")

    scaffold_keys = compute_scaffold_keys(
        selected.canonical_smiles.tolist(), workers=args.workers
    )
    scaffold_sets = {
        role: set(scaffold_keys[selected.split.eq(role).to_numpy()])
        for role in ("train", "validation", "test")
    }
    overlaps = {
        f"{left}_{right}": len(scaffold_sets[left] & scaffold_sets[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Scaffold leakage detected: {overlaps}")

    output = args.out_dir / "split.csv"
    atomic_csv(
        selected[
            [
                "source_idx",
                "split",
                "cid",
                "smiles",
                "canonical_smiles",
                "homo",
                "lumo",
                "gap",
            ]
        ],
        output,
    )
    report = {
        "experiment": "pubchemqc100k_architecture_screen",
        "source_csv": str(args.csv),
        "source_csv_sha256": sha256(args.csv),
        "source_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "split_rows": {
            role: int(selected.split.eq(role).sum())
            for role in ("train", "validation", "test")
        },
        "split_seed": args.seed,
        "scaffold_rule": "Bemis-Murcko; acyclic molecules keyed by canonical SMILES",
        "scaffold_overlap": overlaps,
        "split_csv": str(output),
        "split_csv_sha256": sha256(output),
        "sealed_20k_used": False,
    }
    atomic_json(report, args.out_dir / "split_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
