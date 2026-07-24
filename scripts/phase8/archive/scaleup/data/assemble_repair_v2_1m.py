"""Byte-preservingly append a deduplicated top-up to a frozen training CSV.

The frozen expansion500K CSV is copied as raw bytes before appending the
selected repair-v2 top-up body (without its header).  This is intentionally not
a pandas concatenation: the assembled file's prefix is required to be bytewise
identical to the accepted 500K base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.utils import ensure_dirs


BASE_CSV = RAW_DIR / "phase8_expansion_500k.csv"
TOPUP_CSV = RAW_DIR / "phase8_repair_v2_selected_500k.csv"
OUT_CSV = RAW_DIR / "phase8_repair_v2_1m.csv"
REPORT_JSON = RESULTS_DIR / "phase8" / "repair_v2_1m_assembly_report.json"
TRAIN_HEADER = b"cid,mw,formula,smiles,homo,lumo,gap,canonical_smiles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, default=BASE_CSV)
    parser.add_argument("--topup-csv", type=Path, default=TOPUP_CSV)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--dataset-name", default="phase8_repair_1m_v2")
    parser.add_argument("--expected-base-rows", type=int, default=500_000)
    parser.add_argument("--expected-topup-rows", type=int, default=500_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(1024 * 1024), b""))


def header_and_body_offset(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
        offset = handle.tell()
    if header != TRAIN_HEADER:
        raise ValueError(f"Unexpected training header in {path}: {header!r}")
    return offset


def assert_prefix_equals(base: Path, assembled: Path) -> None:
    with base.open("rb") as left, assembled.open("rb") as right:
        while True:
            lhs = left.read(1024 * 1024)
            rhs = right.read(len(lhs))
            if lhs != rhs:
                raise RuntimeError("Assembled CSV does not preserve the frozen base byte prefix")
            if not lhs:
                return


def main() -> None:
    args = parse_args()
    for path in (args.base_csv, args.topup_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_rows = line_count(args.base_csv) - 1
    topup_rows = line_count(args.topup_csv) - 1
    if base_rows != args.expected_base_rows or topup_rows != args.expected_topup_rows:
        raise ValueError(
            f"Expected {args.expected_base_rows:,} + {args.expected_topup_rows:,}, "
            f"got {base_rows:,} + {topup_rows:,}"
        )
    topup_offset = header_and_body_offset(args.topup_csv)
    header_and_body_offset(args.base_csv)

    # CSV-level identity checks make concatenation failure explicit before write.
    base = pd.read_csv(args.base_csv, usecols=["cid", "canonical_smiles"], dtype="string")
    topup = pd.read_csv(args.topup_csv, usecols=["cid", "canonical_smiles"], dtype="string")
    duplicate_cid = int(topup["cid"].duplicated().sum())
    duplicate_smiles = int(topup["canonical_smiles"].duplicated().sum())
    overlap_cid = int(topup["cid"].isin(set(base["cid"].dropna())).sum())
    overlap_smiles = int(topup["canonical_smiles"].isin(set(base["canonical_smiles"].dropna())).sum())
    if duplicate_cid or duplicate_smiles:
        raise RuntimeError(
            f"Top-up contains duplicates: cid={duplicate_cid}, canonical_smiles={duplicate_smiles}"
        )
    if overlap_cid or overlap_smiles:
        raise RuntimeError(
            f"Top-up overlaps frozen base: cid={overlap_cid}, canonical_smiles={overlap_smiles}"
        )

    ensure_dirs(args.out_csv.parent, args.report_json.parent)
    temporary = args.out_csv.with_name(f".{args.out_csv.name}.tmp")
    with temporary.open("wb") as out, args.base_csv.open("rb") as base_file, args.topup_csv.open("rb") as topup_file:
        for block in iter(lambda: base_file.read(1024 * 1024), b""):
            out.write(block)
        topup_file.seek(topup_offset)
        for block in iter(lambda: topup_file.read(1024 * 1024), b""):
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, args.out_csv)
    assert_prefix_equals(args.base_csv, args.out_csv)
    assembled_rows = line_count(args.out_csv) - 1
    if assembled_rows != base_rows + topup_rows:
        raise RuntimeError(f"Expected {base_rows + topup_rows:,} rows, got {assembled_rows:,}")

    report = {
        "dataset": args.dataset_name,
        "base_csv": str(args.base_csv),
        "topup_csv": str(args.topup_csv),
        "out_csv": str(args.out_csv),
        "base_rows": base_rows,
        "topup_rows": topup_rows,
        "assembled_rows": assembled_rows,
        "base_sha256": sha256(args.base_csv),
        "topup_sha256": sha256(args.topup_csv),
        "assembled_sha256": sha256(args.out_csv),
        "base_prefix_bytewise_identical": True,
        "topup_duplicate_cid": duplicate_cid,
        "topup_duplicate_canonical_smiles": duplicate_smiles,
        "cross_overlap_cid": overlap_cid,
        "cross_overlap_canonical_smiles": overlap_smiles,
    }
    report_tmp = args.report_json.with_name(f".{args.report_json.name}.tmp")
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(report_tmp, args.report_json)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
