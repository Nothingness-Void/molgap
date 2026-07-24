"""Validate a retrieved repair-v2 Kaggle round before it becomes a checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"bucket", "cid", "canonical_smiles", "homo", "lumo", "gap"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True)
    ap.add_argument("--manifest-name", default="repair_1m_v2_round_manifest.json")
    ap.add_argument("--train-csv", type=Path, required=True,
                    help="Rejected 1M continuation; all rows must remain disjoint from it")
    ap.add_argument("--exclude-csv", type=Path, action="append", default=[],
                    help="Previously accepted checkpoint CSVs that this round must not overlap")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    manifest_path = args.round_dir / args.manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("state") != "complete":
        errors.append(f"manifest state is {manifest.get('state')!r}, not 'complete'")

    exclusion_paths = [args.train_csv, *args.exclude_csv]
    base_cids: set[int] = set()
    base_smiles: set[str] = set()
    for exclusion_path in exclusion_paths:
        base = pd.read_csv(exclusion_path, usecols=lambda col: col in {"cid", "canonical_smiles"})
        base_cids.update(pd.to_numeric(base.get("cid", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        base_smiles.update(base.get("canonical_smiles", pd.Series(dtype=str)).dropna().astype(str))
    seen_cids: set[int] = set()
    seen_smiles: set[str] = set()
    group_summary: dict[str, dict] = {}

    for group, record in manifest.get("groups", {}).items():
        csv_path = args.round_dir / str(record.get("csv", ""))
        target = int(record.get("target_rows", 0))
        summary = {"csv": csv_path.name, "target_rows": target}
        if not csv_path.exists():
            errors.append(f"{group}: missing {csv_path.name}")
            group_summary[group] = summary
            continue
        actual_hash = sha256(csv_path)
        summary["sha256"] = actual_hash
        if actual_hash != record.get("sha256"):
            errors.append(f"{group}: SHA-256 differs from manifest")
        frame = pd.read_csv(csv_path)
        summary["rows"] = len(frame)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            errors.append(f"{group}: missing columns {missing}")
            group_summary[group] = summary
            continue
        if len(frame) != target:
            errors.append(f"{group}: rows={len(frame):,}, expected={target:,}")
        cids = set(pd.to_numeric(frame["cid"], errors="coerce").dropna().astype(int))
        smiles = set(frame["canonical_smiles"].dropna().astype(str))
        if len(cids) != len(frame) or len(smiles) != len(frame):
            errors.append(f"{group}: duplicate or missing CID/canonical SMILES within CSV")
        if cids & base_cids or smiles & base_smiles:
            errors.append(f"{group}: overlaps rejected 1M continuation")
        if cids & seen_cids or smiles & seen_smiles:
            errors.append(f"{group}: overlaps another checkpoint group")
        seen_cids.update(cids)
        seen_smiles.update(smiles)
        group_summary[group] = summary

    result = {
        "round_dir": str(args.round_dir),
        "round_index": manifest.get("round_index"),
        "exclusion_csvs": [str(path) for path in exclusion_paths],
        "valid": not errors,
        "errors": errors,
        "total_rows": len(seen_smiles),
        "groups": group_summary,
    }
    atomic_json(args.out_json, result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
