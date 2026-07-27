"""Validate and reconcile a retrieved chunked candidate-pool export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"bucket", "cid", "canonical_smiles", "homo", "lumo", "gap"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_keys(paths: list[Path]) -> tuple[set[str], set[str]]:
    cids: set[str] = set()
    smiles: set[str] = set()
    for path in paths:
        for frame in pd.read_csv(
            path,
            usecols=lambda column: column in {"cid", "canonical_smiles"},
            dtype={"cid": "string", "canonical_smiles": "string"},
            chunksize=100_000,
        ):
            cids.update(value for value in (normalize(item) for item in frame.get("cid", [])) if value)
            smiles.update(
                value for value in (normalize(item) for item in frame.get("canonical_smiles", [])) if value
            )
    return cids, smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--accepted-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    if args.accepted_dir.exists():
        raise FileExistsError(f"Refusing to replace accepted directory: {args.accepted_dir}")
    args.accepted_dir.mkdir(parents=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("state") != "complete":
        errors.append(f"manifest state is {manifest.get('state')!r}, not 'complete'")

    excluded_cids, excluded_smiles = load_keys(args.exclude_csv)
    seen_cids: set[str] = set()
    seen_smiles: set[str] = set()
    accepted_manifest: dict[str, object] = {
        "source_manifest": str(args.manifest),
        "source_state": manifest.get("state"),
        "dedup_keys": ["cid", "canonical_smiles"],
        "chunks": [],
    }
    structural_rows = 0
    accepted_rows = 0
    prior_overlap_rows = 0
    within_pool_duplicate_rows = 0

    for record in sorted(manifest.get("chunks", []), key=lambda item: int(item["chunk_index"])):
        chunk_index = int(record["chunk_index"])
        csv_path = args.pool_dir / str(record["csv"])
        expected_rows = int(record.get("target_rows", 0))
        summary: dict[str, object] = {
            "chunk_index": chunk_index,
            "source_csv": csv_path.name,
            "expected_rows": expected_rows,
        }
        if not csv_path.exists():
            errors.append(f"chunk {chunk_index}: missing {csv_path.name}")
            accepted_manifest["chunks"].append(summary)  # type: ignore[union-attr]
            continue
        actual_hash = sha256(csv_path)
        summary["source_sha256"] = actual_hash
        if actual_hash != record.get("sha256"):
            errors.append(f"chunk {chunk_index}: SHA-256 differs from manifest")

        frame = pd.read_csv(csv_path, dtype={"cid": "string", "canonical_smiles": "string"})
        structural_rows += len(frame)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            errors.append(f"chunk {chunk_index}: missing columns {missing}")
            accepted_manifest["chunks"].append(summary)  # type: ignore[union-attr]
            continue
        if len(frame) != expected_rows or int(record.get("rows", -1)) != len(frame):
            errors.append(
                f"chunk {chunk_index}: rows={len(frame):,}, expected={expected_rows:,}, "
                f"manifest={record.get('rows')!r}"
            )
        targets = frame[["homo", "lumo", "gap"]].apply(pd.to_numeric, errors="coerce").to_numpy()
        nonfinite = int((~np.isfinite(targets)).any(axis=1).sum())
        if nonfinite:
            errors.append(f"chunk {chunk_index}: {nonfinite:,} rows have non-finite targets")

        keep: list[bool] = []
        chunk_prior_overlap = 0
        chunk_duplicates = 0
        for row in frame[["cid", "canonical_smiles"]].itertuples(index=False):
            cid = normalize(row.cid)
            smiles = normalize(row.canonical_smiles)
            prior_overlap = (
                (cid is not None and cid in excluded_cids)
                or (smiles is not None and smiles in excluded_smiles)
            )
            duplicate = (
                (cid is not None and cid in seen_cids)
                or (smiles is not None and smiles in seen_smiles)
            )
            accepted = not prior_overlap and not duplicate and cid is not None and smiles is not None
            keep.append(accepted)
            if prior_overlap:
                chunk_prior_overlap += 1
            elif duplicate:
                chunk_duplicates += 1
            if accepted:
                seen_cids.add(cid)
                seen_smiles.add(smiles)

        accepted = frame.loc[keep].copy()
        accepted_path = args.accepted_dir / csv_path.name
        atomic_csv(accepted_path, accepted)
        accepted_hash = sha256(accepted_path)
        prior_overlap_rows += chunk_prior_overlap
        within_pool_duplicate_rows += chunk_duplicates
        accepted_rows += len(accepted)
        summary.update(
            {
                "source_rows": len(frame),
                "prior_overlap_rows": chunk_prior_overlap,
                "within_pool_duplicate_rows": chunk_duplicates,
                "accepted_rows": len(accepted),
                "accepted_csv": accepted_path.name,
                "accepted_sha256": accepted_hash,
            }
        )
        accepted_manifest["chunks"].append(summary)  # type: ignore[union-attr]

    accepted_manifest.update(
        {
            "state": "accepted" if not errors else "invalid",
            "source_rows": structural_rows,
            "prior_overlap_rows": prior_overlap_rows,
            "within_pool_duplicate_rows": within_pool_duplicate_rows,
            "accepted_rows": accepted_rows,
            "errors": errors,
        }
    )
    accepted_manifest_path = args.accepted_dir / "accepted_manifest.json"
    atomic_json(accepted_manifest_path, accepted_manifest)
    result = {
        "valid_source": not errors,
        "source_rows": structural_rows,
        "prior_overlap_rows": prior_overlap_rows,
        "within_pool_duplicate_rows": within_pool_duplicate_rows,
        "accepted_rows": accepted_rows,
        "errors": errors,
        "accepted_manifest": str(accepted_manifest_path),
    }
    atomic_json(args.out_json, result)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
