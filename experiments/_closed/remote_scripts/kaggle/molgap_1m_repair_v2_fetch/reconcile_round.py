"""Reconcile duplicate rows produced by independently parallel bucket workers.

The source collectors already exclude prior accepted rounds. This local
acceptance step handles the remaining race: a molecule can satisfy two bucket
definitions and be emitted by two workers in the same round. It keeps the
first row in a fixed bucket priority order and makes the reduced accepted quota
explicit in the manifest before that round is published as a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


GROUP_ORDER = (
    "macro_amide",
    "very_large",
    "flexible_lowmid",
    "sp3_nonaromatic",
    "rare",
    "aromatic_large",
    "topology_elements",
    "balanced",
)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True)
    ap.add_argument("--manifest-name", default="repair_1m_v2_round_manifest.json",
                    help="Manifest filename; use repair_1m_v2_manifest.json for the legacy full-pool run")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    manifest_path = args.round_dir / args.manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen_cids: set[str] = set()
    seen_smiles: set[str] = set()
    reconciliation: dict[str, dict] = {}
    total_rows = 0

    groups = manifest.get("groups", {})
    ordered_groups = [group for group in GROUP_ORDER if group in groups]
    ordered_groups.extend(group for group in groups if group not in ordered_groups)
    for group in ordered_groups:
        record = groups[group]
        report = record.get("report", {})
        csv_name = record.get("csv") or (Path(str(report["out_csv"])).name if isinstance(report, dict) and report.get("out_csv") else None)
        if not csv_name:
            raise ValueError(f"{group}: manifest record has no CSV path")
        csv_path = args.round_dir / str(csv_name)
        frame = pd.read_csv(csv_path, dtype={"cid": "string", "canonical_smiles": "string"})
        keep = []
        dropped: list[dict[str, str | None]] = []
        for row in frame[["cid", "canonical_smiles"]].itertuples(index=False):
            cid, smiles = key(row.cid), key(row.canonical_smiles)
            duplicate = (cid is not None and cid in seen_cids) or (smiles is not None and smiles in seen_smiles)
            keep.append(not duplicate)
            if duplicate:
                dropped.append({"cid": cid, "canonical_smiles": smiles})
                continue
            if cid is not None:
                seen_cids.add(cid)
            if smiles is not None:
                seen_smiles.add(smiles)
        accepted = frame.loc[keep].copy()
        original_rows = len(frame)
        if len(accepted) != original_rows:
            atomic_csv(csv_path, accepted)
        original_target = int(record.get("target_rows", original_rows))
        record["original_target_rows"] = original_target
        record["target_rows"] = len(accepted)
        record["csv"] = csv_path.name
        record["sha256"] = sha256(csv_path)
        reconciliation[group] = {
            "original_rows": original_rows,
            "accepted_rows": len(accepted),
            "dropped_duplicate_rows": original_rows - len(accepted),
            "dropped_examples": dropped[:20],
        }
        total_rows += len(accepted)

    if "round_target_rows" in manifest:
        manifest["round_target_rows"] = total_rows
    else:
        manifest["accepted_total_rows"] = total_rows
    manifest["state"] = "complete"
    manifest["complete"] = True
    manifest["reconciliation"] = reconciliation
    atomic_json(manifest_path, manifest)
    result = {
        "round_dir": str(args.round_dir),
        "round_index": manifest.get("round_index"),
        "accepted_rows": total_rows,
        "reconciliation": reconciliation,
    }
    atomic_json(args.out_json, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
