"""Build a new scaffold-novel sealed set for the static multi-2D ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.router_sampling import compute_scaffold_keys, select_descriptor_diverse


DESCRIPTOR_COLUMNS = (
    "mw",
    "heavy_atoms",
    "ring_count",
    "aromatic_rings",
    "aromatic_atom_fraction",
    "rotatable_bonds",
    "conjugated_bonds",
    "fraction_csp3",
    "amide_bonds",
    "macrocycle",
    "bridgeheads",
    "has_s",
    "has_cl",
    "has_f",
)
REQUIRED_COLUMNS = ("cid", "smiles", "canonical_smiles", "homo", "lumo", "gap")


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_exclusions(cache_dirs: list[Path]) -> tuple[set[str], set[str], set[str], int]:
    cids: set[str] = set()
    smiles: set[str] = set()
    scaffolds: set[str] = set()
    rows = 0
    for cache_dir in cache_dirs:
        parts = sorted(cache_dir.glob("part_*.csv"))
        if not parts:
            raise FileNotFoundError(f"No scaffold-cache parts in {cache_dir}")
        for path in parts:
            part = pd.read_csv(
                path,
                usecols=["cid", "canonical_smiles", "scaffold"],
                dtype="string",
            )
            rows += len(part)
            cids.update(part.cid.dropna())
            smiles.update(part.canonical_smiles.dropna())
            scaffolds.update(part.scaffold.dropna())
    return cids, smiles, scaffolds, rows


def cached_scaffolds(
    frame: pd.DataFrame,
    cache_dir: Path,
    *,
    workers: int,
    chunk_size: int,
) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    total_parts = (len(frame) + chunk_size - 1) // chunk_size
    for part_index, start in enumerate(range(0, len(frame), chunk_size)):
        stop = min(start + chunk_size, len(frame))
        expected = frame.iloc[start:stop][["cid", "canonical_smiles"]].reset_index(drop=True)
        path = cache_dir / f"part_{part_index:05d}.csv"
        if path.exists():
            part = pd.read_csv(path, dtype="string")
            if len(part) != len(expected) or not part[["cid", "canonical_smiles"]].equals(
                expected.astype("string")
            ):
                raise RuntimeError(f"Invalid cached scaffold part: {path}")
        else:
            part = expected.copy()
            part["scaffold"] = compute_scaffold_keys(
                expected.canonical_smiles.astype(str).tolist(), workers=workers
            )
            atomic_csv(part, path)
        outputs.append(part.scaffold.astype(str))
        atomic_json(
            {
                "completed_parts": part_index + 1,
                "total_parts": total_parts,
                "completed_rows": stop,
            },
            cache_dir / "progress.json",
        )
        print(f"scaffolds {part_index + 1}/{total_parts}: {stop:,}/{len(frame):,}", flush=True)
    return pd.concat(outputs, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-pattern", default="phase8_2m_round0[45]_*.csv")
    parser.add_argument("--exclude-cache", type=Path, action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_files = sorted(args.candidate_dir.glob(args.candidate_pattern))
    if not candidate_files:
        raise FileNotFoundError(
            f"No candidate files match {args.candidate_dir / args.candidate_pattern}"
        )
    candidates = pd.concat(
        [pd.read_csv(path) for path in candidate_files], ignore_index=True
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in candidates]
    if missing:
        raise ValueError(f"Candidate files are missing columns: {missing}")
    if candidates.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Candidate rows contain missing required values")
    label_mismatch = float((candidates.gap - (candidates.lumo - candidates.homo)).abs().max())
    if label_mismatch > 1e-8:
        raise ValueError(f"Gap label identity failed: max mismatch {label_mismatch}")

    raw_rows = len(candidates)
    candidates = candidates.drop_duplicates("cid").drop_duplicates("canonical_smiles")
    candidates = candidates.reset_index(drop=True)
    deduplicated_rows = len(candidates)

    excluded_cids, excluded_smiles, excluded_scaffolds, exclusion_rows = load_exclusions(
        args.exclude_cache
    )
    exact_mask = candidates.cid.astype(str).isin(excluded_cids) | candidates.canonical_smiles.astype(
        str
    ).isin(excluded_smiles)
    exact_overlap_rows = int(exact_mask.sum())
    candidates = candidates.loc[~exact_mask].reset_index(drop=True)
    candidates["scaffold"] = cached_scaffolds(
        candidates,
        args.cache_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    scaffold_mask = candidates.scaffold.isin(excluded_scaffolds)
    scaffold_overlap_rows = int(scaffold_mask.sum())
    novel = candidates.loc[~scaffold_mask].copy()
    novel = novel.drop_duplicates("scaffold").reset_index(drop=True)
    if len(novel) < args.rows:
        raise RuntimeError(f"Only {len(novel):,} unique novel scaffolds for {args.rows:,} rows")

    missing_descriptors = [column for column in DESCRIPTOR_COLUMNS if column not in novel]
    if missing_descriptors:
        raise ValueError(f"Candidate files are missing descriptors: {missing_descriptors}")
    selected_indices, probabilities = select_descriptor_diverse(
        novel,
        novel.index.to_numpy(dtype=np.int64),
        features=DESCRIPTOR_COLUMNS,
        n_select=args.rows,
        n_clusters=args.clusters,
        seed=args.seed,
    )
    sealed = novel.loc[selected_indices].copy().reset_index(drop=True)
    sealed["selection_probability"] = [probabilities[int(index)] for index in selected_indices]
    sealed["sealed_role"] = "multi2d_final_scaffold_novel"
    if sealed.scaffold.nunique() != len(sealed):
        raise RuntimeError("The sealed set does not contain one row per scaffold")
    if set(sealed.scaffold) & excluded_scaffolds:
        raise RuntimeError("The sealed set overlaps an expert-training scaffold")

    atomic_csv(sealed, args.out_csv)
    sha256 = hashlib.sha256(args.out_csv.read_bytes()).hexdigest()
    report = {
        "dataset": "phase8_multi2d_final_scaffold_novel_10k",
        "candidate_files": [str(path) for path in candidate_files],
        "candidate_raw_rows": raw_rows,
        "candidate_deduplicated_rows": deduplicated_rows,
        "exclusion_cache_rows": exclusion_rows,
        "excluded_unique_cids": len(excluded_cids),
        "excluded_unique_smiles": len(excluded_smiles),
        "excluded_unique_scaffolds": len(excluded_scaffolds),
        "exact_overlap_rows_removed": exact_overlap_rows,
        "scaffold_overlap_rows_removed": scaffold_overlap_rows,
        "novel_unique_scaffold_rows": len(novel),
        "sealed_rows": len(sealed),
        "sealed_unique_scaffolds": int(sealed.scaffold.nunique()),
        "bucket_counts": {str(key): int(value) for key, value in sealed.bucket.value_counts().items()},
        "gap_identity_max_abs_eV": label_mismatch,
        "seed": args.seed,
        "descriptor_clusters": args.clusters,
        "sha256": sha256,
        "training_use_forbidden": True,
    }
    atomic_json(report, args.report_json)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
