"""Build the durable row ledger and fixed-size repaired-2M sampling manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.data_repair import (
    build_ledger,
    compare_current_and_repaired,
    discover_sources,
    materialize_manifest,
    select_repaired_manifest,
    validate_manifest_references,
    write_manifest_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase8/repaired_2m"),
    )
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--skip-ledger",
        action="store_true",
        help="Reuse already validated ledger parts.",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Write data/raw/phase8_repaired_2m.csv after manifest acceptance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    ledger_dir = output_dir / "ledger"
    sources = discover_sources(root)
    if not args.skip_ledger:
        build_ledger(
            sources,
            ledger_dir,
            chunk_size=args.chunk_size,
            workers=args.workers,
        )
    report, selected = select_repaired_manifest(
        ledger_dir,
        output_dir / "repaired_2m_manifest.parquet",
        seed=args.seed,
    )
    comparison = compare_current_and_repaired(
        ledger_dir,
        selected,
        output_dir / "distribution_before_after.csv",
    )
    audit = write_manifest_audit(ledger_dir, selected, output_dir)
    validation = validate_manifest_references(
        ledger_dir,
        selected,
        output_dir / "validation_report.json",
        seed=args.seed,
    )
    materialized = None
    if args.materialize:
        materialized = materialize_manifest(
            selected,
            root / "data" / "raw" / "phase8_repaired_2m.csv",
            output_dir / "materialization_report.json",
        )
    print(report)
    print(audit)
    print(validation)
    if materialized is not None:
        print(materialized)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
