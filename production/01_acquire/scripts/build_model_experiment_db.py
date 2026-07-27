"""Build the normalized Phase 8 model and experiment database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from molgap.experiment_db import build_database, export_query


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO / "results" / "phase8" / "model_inventory_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIR / "model_experiments.sqlite",
    )
    parser.add_argument(
        "--hash-artifacts",
        action="store_true",
        help="SHA256 all files under models/; slower but suitable for release audits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_database(REPO, args.output, hash_artifacts=args.hash_artifacts)

    comparison_csv = args.output.parent / "unified_model_comparison.csv"
    export_query(
        args.output,
        """
        SELECT
            e.model_id,
            m.family,
            m.status,
            m.approximate_encoder_passes,
            e.scope,
            e.target,
            e.n,
            e.mae_ev,
            e.baseline_model_id,
            e.delta_mae_ev,
            e.ci95_low_ev,
            e.ci95_high_ev,
            e.source_path
        FROM comparable_external e
        JOIN models m USING(model_id)
        ORDER BY e.scope, e.target, e.mae_ev
        """,
        comparison_csv,
    )
    artifact_csv = args.output.parent / "artifact_inventory.csv"
    export_query(
        args.output,
        """
        SELECT
            a.model_id,
            m.family,
            a.path,
            a.artifact_kind,
            a.bytes,
            a.sha256
        FROM artifacts a
        LEFT JOIN models m USING(model_id)
        ORDER BY COALESCE(a.model_id, 'ZZZ'), a.path
        """,
        artifact_csv,
    )

    with sqlite3.connect(args.output) as connection:
        models = connection.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        evaluations = connection.execute(
            "SELECT COUNT(*) FROM evaluations"
        ).fetchone()[0]
        artifacts = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    print(f"database: {args.output}")
    print(f"models={models} evaluations={evaluations} artifacts={artifacts}")
    print(f"comparison: {comparison_csv}")
    print(f"artifacts: {artifact_csv}")


if __name__ == "__main__":
    main()

