"""Assemble the exact 500K append and future sealed set for a 2M 2D expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.multi2d_data import (
    assemble_coverage_topup,
    parse_family_values,
    parse_source_patterns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="FAMILY=GLOB")
    parser.add_argument("--train-quota", action="append", required=True, help="FAMILY=ROWS")
    parser.add_argument("--sealed-quota", action="append", required=True, help="FAMILY=ROWS")
    parser.add_argument("--exclude-cache", type=Path, action="append", required=True)
    parser.add_argument("--existing-sealed", type=Path, action="append", default=[])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--topup-out", type=Path, required=True)
    parser.add_argument("--sealed-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--clusters", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = assemble_coverage_topup(
        base_csv=args.base_csv,
        source_patterns=parse_source_patterns(args.source),
        train_quotas=parse_family_values(args.train_quota),
        sealed_quotas=parse_family_values(args.sealed_quota),
        exclusion_cache_dirs=args.exclude_cache,
        existing_sealed_csvs=args.existing_sealed,
        scaffold_cache_dir=args.cache_dir,
        topup_out=args.topup_out,
        sealed_out=args.sealed_out,
        audit_out=args.audit_out,
        report_out=args.report_out,
        workers=args.workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
        clusters=args.clusters,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
