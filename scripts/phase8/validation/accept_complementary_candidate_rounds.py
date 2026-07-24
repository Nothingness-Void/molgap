"""Strictly accept complete and explicitly recovered complementary fetch rounds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"bucket", "cid", "canonical_smiles", "homo", "lumo", "gap"}
GROUP_PRIORITY = ("high_gap", "hetero_dense", "bridged_rigid", "conjugated_da")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


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


def recovery_map(values: list[str]) -> dict[tuple[int, str], int]:
    result = {}
    for value in values:
        round_text, group, rows_text = value.split(":", 2)
        key = (int(round_text), group)
        if key in result or int(rows_text) <= 0:
            raise ValueError(f"Invalid duplicate/non-positive recovery: {value}")
        result[key] = int(rows_text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--recover-group", action="append", default=[], metavar="ROUND:GROUP:ROWS")
    parser.add_argument("--accepted-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.accepted_dir.exists():
        raise FileExistsError(f"Refusing to replace accepted directory: {args.accepted_dir}")
    args.accepted_dir.mkdir(parents=True)
    recoveries = recovery_map(args.recover_group)
    excluded_cids, excluded_smiles = load_keys(args.exclude_csv)
    seen_cids: set[str] = set()
    seen_smiles: set[str] = set()
    errors: list[str] = []
    sources: list[dict] = []

    rounds = []
    for round_dir in args.round_dir:
        manifests = list(round_dir.glob("phase8_2m_round_manifest.json"))
        if len(manifests) != 1:
            raise FileNotFoundError(f"Expected one round manifest in {round_dir}, found {manifests}")
        manifest_path = manifests[0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rounds.append((int(manifest["round_index"]), round_dir, manifest_path, manifest))

    for group in GROUP_PRIORITY:
        for round_index, round_dir, manifest_path, manifest in sorted(rounds):
            record = manifest.get("groups", {}).get(group)
            if record is None:
                errors.append(f"round {round_index}: missing group {group}")
                continue
            csv_path = round_dir / str(record["csv"])
            recovery_rows = recoveries.get((round_index, group))
            complete = int(record.get("return_code", -1)) == 0 and "report" in record
            if not complete and recovery_rows is None:
                errors.append(f"round {round_index} {group}: incomplete without explicit recovery")
                continue
            if not csv_path.is_file():
                errors.append(f"round {round_index} {group}: missing {csv_path.name}")
                continue

            source_hash = sha256(csv_path)
            if complete and source_hash != record.get("sha256"):
                errors.append(f"round {round_index} {group}: SHA-256 differs from manifest")
            frame = pd.read_csv(csv_path, dtype={"cid": "string", "canonical_smiles": "string"})
            raw_rows = len(frame)
            if complete:
                expected_rows = int(record["report"]["total_rows"])
                if raw_rows != expected_rows or expected_rows != int(record["target_rows"]):
                    errors.append(
                        f"round {round_index} {group}: rows={raw_rows}, expected={expected_rows}, "
                        f"target={record['target_rows']}"
                    )
            else:
                expected_rows = int(recovery_rows)
                if raw_rows < expected_rows:
                    errors.append(
                        f"round {round_index} {group}: only {raw_rows} rows for recovery {expected_rows}"
                    )
                frame = frame.iloc[:expected_rows].copy()

            missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
            if missing:
                errors.append(f"round {round_index} {group}: missing columns {missing}")
                continue
            targets = frame[["homo", "lumo", "gap"]].apply(pd.to_numeric, errors="coerce").to_numpy()
            nonfinite = int((~np.isfinite(targets)).any(axis=1).sum())
            mismatch = np.abs(targets[:, 2] - (targets[:, 1] - targets[:, 0]))
            algebra_failures = int((mismatch > 1e-6).sum())
            missing_keys = int(
                sum(
                    normalize(row.cid) is None or normalize(row.canonical_smiles) is None
                    for row in frame[["cid", "canonical_smiles"]].itertuples(index=False)
                )
            )
            unknown_buckets = sorted(set(frame["bucket"].dropna().astype(str)) - set(record["buckets"]))
            if nonfinite or algebra_failures or missing_keys or unknown_buckets:
                errors.append(
                    f"round {round_index} {group}: nonfinite={nonfinite}, algebra={algebra_failures}, "
                    f"missing_keys={missing_keys}, unknown_buckets={unknown_buckets}"
                )

            keep = []
            prior_overlap = 0
            duplicate = 0
            for row in frame[["cid", "canonical_smiles"]].itertuples(index=False):
                cid = normalize(row.cid)
                smiles = normalize(row.canonical_smiles)
                is_prior = cid in excluded_cids or smiles in excluded_smiles
                is_duplicate = cid in seen_cids or smiles in seen_smiles
                accepted = not is_prior and not is_duplicate and cid is not None and smiles is not None
                keep.append(accepted)
                if is_prior:
                    prior_overlap += 1
                elif is_duplicate:
                    duplicate += 1
                if accepted:
                    seen_cids.add(cid)
                    seen_smiles.add(smiles)

            accepted = frame.loc[keep].copy()
            accepted_name = f"phase8_2m_round{round_index:02d}_{group}.csv"
            accepted_path = args.accepted_dir / accepted_name
            atomic_csv(accepted_path, accepted)
            sources.append({
                "round": round_index,
                "group": group,
                "source_manifest": str(manifest_path),
                "source_state": manifest.get("state"),
                "recovered": not complete,
                "source_csv": str(csv_path),
                "source_sha256": source_hash,
                "raw_rows": raw_rows,
                "validated_rows": len(frame),
                "prior_overlap_rows": prior_overlap,
                "cross_source_duplicate_rows": duplicate,
                "accepted_rows": len(accepted),
                "accepted_csv": accepted_name,
                "accepted_sha256": sha256(accepted_path),
                "max_gap_algebra_error": float(mismatch.max(initial=0.0)),
            })

    report = {
        "state": "accepted" if not errors else "invalid",
        "rounds": [round_index for round_index, *_ in sorted(rounds)],
        "recoveries": [
            {"round": round_index, "group": group, "durable_rows": rows}
            for (round_index, group), rows in sorted(recoveries.items())
        ],
        "exclusion_csvs": [str(path) for path in args.exclude_csv],
        "excluded_cids": len(excluded_cids),
        "excluded_canonical_smiles": len(excluded_smiles),
        "validated_rows": sum(int(source["validated_rows"]) for source in sources),
        "prior_overlap_rows": sum(int(source["prior_overlap_rows"]) for source in sources),
        "cross_source_duplicate_rows": sum(int(source["cross_source_duplicate_rows"]) for source in sources),
        "accepted_rows": sum(int(source["accepted_rows"]) for source in sources),
        "sources": sources,
        "errors": errors,
    }
    atomic_json(args.accepted_dir / "accepted_manifest.json", report)
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
