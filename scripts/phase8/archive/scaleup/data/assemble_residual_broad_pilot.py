"""Assemble a leakage-audited additive pilot from Kaggle acquisition rounds.

The original training table is preserved as an exact prefix. Candidate rows are
deduplicated by both CID and canonical SMILES. A deterministic held-out set is
selected only from scaffolds absent from the original training table, and every
row sharing a held-out scaffold is excluded from the top-up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from molgap.utils import scaffold_split_key


TRAIN_COLUMNS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap", "canonical_smiles"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--broad-dir", type=Path, required=True)
    parser.add_argument("--residual-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--topup-csv", type=Path, required=True)
    parser.add_argument("--sealed-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sealed-rows", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def signature(frame: pd.DataFrame) -> dict:
    edges = pd.concat(
        [frame[["cid", "canonical_smiles"]].head(3), frame[["cid", "canonical_smiles"]].tail(3)],
        ignore_index=True,
    ).to_csv(index=False)
    return {"rows": len(frame), "edge_sha256": hashlib.sha256(edges.encode()).hexdigest()}


def cached_scaffolds(frame: pd.DataFrame, label: str, cache_dir: Path, workers: int, chunk_size: int) -> pd.Series:
    root = cache_dir / label
    root.mkdir(parents=True, exist_ok=True)
    expected_signature = signature(frame)
    meta_path = root / "meta.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8"))["signature"] != expected_signature:
            raise RuntimeError(f"Scaffold cache signature mismatch: {root}")
    else:
        atomic_json({"signature": expected_signature, "chunk_size": chunk_size}, meta_path)

    parts: list[pd.Series] = []
    total = math.ceil(len(frame) / chunk_size)
    for index, start in enumerate(range(0, len(frame), chunk_size)):
        stop = min(start + chunk_size, len(frame))
        expected = frame.iloc[start:stop][["cid", "canonical_smiles"]].reset_index(drop=True)
        part_path = root / f"part_{index:05d}.csv"
        if part_path.exists():
            part = pd.read_csv(part_path, dtype={"cid": "string", "canonical_smiles": "string", "scaffold": "string"})
            if len(part) != len(expected) or not part[["cid", "canonical_smiles"]].equals(expected.astype("string")):
                raise RuntimeError(f"Invalid scaffold cache part: {part_path}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                values = list(pool.map(scaffold_split_key, expected.canonical_smiles.astype(str), chunksize=500))
            part = expected.copy()
            part["scaffold"] = values
            atomic_csv(part, part_path)
        parts.append(part["scaffold"].astype(str))
        atomic_json({"completed_parts": index + 1, "total_parts": total, "completed_rows": stop}, root / "progress.json")
        print(f"scaffold {label}: {index + 1}/{total} ({stop:,}/{len(frame):,})", flush=True)
    return pd.concat(parts, ignore_index=True)


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"phase8-residual-broad-sealed-v1::{value}".encode()).hexdigest()


def validate_labels(frame: pd.DataFrame, label: str) -> None:
    missing = [column for column in TRAIN_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    if frame[TRAIN_COLUMNS].isna().any().any():
        raise ValueError(f"{label} contains missing training values")
    mismatch = (frame.gap - (frame.lumo - frame.homo)).abs().max()
    if mismatch > 1e-8:
        raise ValueError(f"{label} violates Gap=LUMO-HOMO: max={mismatch}")


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_csv)
    broad_files = sorted(args.broad_dir.glob("*.csv"))
    if not broad_files:
        raise FileNotFoundError(f"No CSV files in {args.broad_dir}")
    broad = pd.concat([pd.read_csv(path) for path in broad_files], ignore_index=True)
    residual = pd.read_csv(args.residual_csv)
    for label, frame in (("base", base), ("broad", broad), ("residual", residual)):
        validate_labels(frame, label)

    candidates = pd.concat([broad, residual], ignore_index=True)
    input_rows = len(candidates)
    candidates = candidates.drop_duplicates("cid").drop_duplicates("canonical_smiles").reset_index(drop=True)
    base_cids = set(base.cid.astype(str))
    base_smiles = set(base.canonical_smiles.astype(str))
    candidates = candidates[
        ~candidates.cid.astype(str).isin(base_cids)
        & ~candidates.canonical_smiles.astype(str).isin(base_smiles)
    ].reset_index(drop=True)

    base_scaffolds = cached_scaffolds(base, "original_1m", args.cache_dir, args.workers, args.chunk_size)
    candidates["scaffold"] = cached_scaffolds(
        candidates, "broad_residual_candidates", args.cache_dir, args.workers, args.chunk_size
    )
    base_scaffold_set = set(base_scaffolds.astype(str))
    novel = candidates[~candidates.scaffold.isin(base_scaffold_set)].copy()
    group_sizes = novel.groupby("scaffold").size().to_dict()
    ordered_scaffolds = sorted(group_sizes, key=stable_rank)
    selected_scaffolds: list[str] = []
    selected_rows = 0
    for scaffold in ordered_scaffolds:
        size = int(group_sizes[scaffold])
        if selected_rows + size <= args.sealed_rows:
            selected_scaffolds.append(scaffold)
            selected_rows += size
        if selected_rows == args.sealed_rows:
            break
    if selected_rows < args.sealed_rows:
        raise RuntimeError(
            f"Could create only {selected_rows:,}/{args.sealed_rows:,} whole-scaffold sealed rows"
        )

    sealed_set = set(selected_scaffolds)
    sealed = candidates[candidates.scaffold.isin(sealed_set)].copy()
    topup = candidates[~candidates.scaffold.isin(sealed_set)].copy()
    assembled = pd.concat([base[TRAIN_COLUMNS], topup[TRAIN_COLUMNS]], ignore_index=True)
    if not assembled.iloc[: len(base)][TRAIN_COLUMNS].equals(base[TRAIN_COLUMNS]):
        raise RuntimeError("Original training prefix changed during assembly")
    if set(topup.scaffold) & set(sealed.scaffold):
        raise RuntimeError("Training and sealed scaffolds overlap")
    if set(sealed.scaffold) & base_scaffold_set:
        raise RuntimeError("Original training and sealed scaffolds overlap")

    atomic_csv(assembled, args.out_csv)
    atomic_csv(topup[TRAIN_COLUMNS], args.topup_csv)
    atomic_csv(sealed, args.sealed_csv)
    report = {
        "dataset": "phase8_original1m_plus_broad_residual_uniform",
        "base_rows": len(base),
        "broad_rows": len(broad),
        "residual_rows": len(residual),
        "candidate_input_rows": input_rows,
        "candidate_unique_disjoint_rows": len(candidates),
        "candidate_novel_scaffold_rows": len(novel),
        "topup_rows": len(topup),
        "sealed_rows": len(sealed),
        "sealed_scaffolds": len(sealed_set),
        "total_train_rows": len(assembled),
        "topup_csv": str(args.topup_csv),
        "base_prefix_preserved": True,
        "sealed_disjoint_from_base_and_topup_by_scaffold": True,
        "sampling": "uniform; no replay weighting",
        "sources": {
            "base": str(args.base_csv),
            "broad_dir": str(args.broad_dir),
            "residual": str(args.residual_csv),
        },
    }
    atomic_json(report, args.report_json)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
