"""Select a leakage-safe 20K hard-region top-up for the exact 2M 2D model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.multi2d_data import parse_family_values, select_bucketed_rescue_topup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-report", type=Path, required=True)
    parser.add_argument("--source-family", action="append", required=True)
    parser.add_argument("--quota", action="append", required=True, help="BUCKET=ROWS")
    parser.add_argument("--used-csv", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-csv", type=Path, action="append", required=True)
    parser.add_argument("--scaffold-cache-dir", type=Path, required=True)
    parser.add_argument("--topup-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--clusters", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = select_bucketed_rescue_topup(
        assembly_report=args.assembly_report,
        source_families=args.source_family,
        quotas=parse_family_values(args.quota),
        used_csvs=args.used_csv,
        evaluation_csvs=args.evaluation_csv,
        scaffold_cache_dir=args.scaffold_cache_dir,
        topup_out=args.topup_out,
        audit_out=args.audit_out,
        report_out=args.report_out,
        workers=args.workers,
        seed=args.seed,
        clusters=args.clusters,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
