"""Validate a complete, possibly unchunked repair-v2 candidate-pool export."""
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


def normalize(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_keys(path: Path, cids: set[str], smiles: set[str]) -> None:
    for chunk in pd.read_csv(path, usecols=lambda col: col in {"cid", "canonical_smiles"},
                             dtype={"cid": "string", "canonical_smiles": "string"}, chunksize=100_000):
        cids.update(value for value in (normalize(item) for item in chunk.get("cid", [])) if value)
        smiles.update(value for value in (normalize(item) for item in chunk.get("canonical_smiles", [])) if value)


def csv_name(record: dict) -> str:
    if record.get("csv"):
        return str(record["csv"])
    report = record.get("report", {})
    if isinstance(report, dict) and report.get("out_csv"):
        return Path(str(report["out_csv"])).name
    raise ValueError("Manifest group record has no CSV path")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("complete") is not True and manifest.get("state") != "complete":
        errors.append("manifest does not report a complete task")

    excluded_cids: set[str] = set()
    excluded_smiles: set[str] = set()
    load_keys(args.train_csv, excluded_cids, excluded_smiles)
    seen_cids: set[str] = set()
    seen_smiles: set[str] = set()
    summary: dict[str, dict] = {}

    for group, record in manifest.get("groups", {}).items():
        path = args.pool_dir / csv_name(record)
        target = int(record.get("target_rows", 0))
        group_info = {"csv": path.name, "target_rows": target}
        if record.get("return_code") not in {None, 0}:
            errors.append(f"{group}: non-zero collector return code")
        if not path.exists():
            errors.append(f"{group}: missing CSV {path.name}")
            summary[group] = group_info
            continue
        group_info["sha256"] = sha256(path)
        rows = 0
        duplicate_count = 0
        excluded_overlap = 0
        examples: list[dict[str, str | None]] = []
        columns_checked = False
        for chunk in pd.read_csv(path, dtype={"cid": "string", "canonical_smiles": "string"}, chunksize=100_000):
            if not columns_checked:
                missing = sorted(REQUIRED_COLUMNS - set(chunk.columns))
                if missing:
                    errors.append(f"{group}: missing columns {missing}")
                    break
                columns_checked = True
            for row in chunk[["cid", "canonical_smiles"]].itertuples(index=False):
                cid, smiles = normalize(row.cid), normalize(row.canonical_smiles)
                duplicate = (cid is not None and cid in seen_cids) or (smiles is not None and smiles in seen_smiles)
                overlaps_exclusion = ((cid is not None and cid in excluded_cids)
                                      or (smiles is not None and smiles in excluded_smiles))
                if duplicate:
                    duplicate_count += 1
                if overlaps_exclusion:
                    excluded_overlap += 1
                if (duplicate or overlaps_exclusion) and len(examples) < 20:
                    examples.append({"cid": cid, "canonical_smiles": smiles})
                if cid is not None:
                    seen_cids.add(cid)
                if smiles is not None:
                    seen_smiles.add(smiles)
                rows += 1
        group_info.update({"rows": rows, "cross_or_within_pool_duplicates": duplicate_count,
                           "rejected_1m_overlaps": excluded_overlap, "examples": examples})
        if rows != target:
            errors.append(f"{group}: rows={rows:,}, expected={target:,}")
        if duplicate_count:
            errors.append(f"{group}: {duplicate_count:,} duplicate rows within/cross pool")
        if excluded_overlap:
            errors.append(f"{group}: {excluded_overlap:,} rows overlap rejected 1M")
        summary[group] = group_info

    result = {
        "pool_dir": str(args.pool_dir),
        "manifest": str(args.manifest),
        "valid": not errors,
        "errors": errors,
        "unique_rows": len(seen_smiles),
        "groups": summary,
    }
    atomic_json(args.out_json, result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
