from __future__ import annotations

import sqlite3
from pathlib import Path

from molgap.experiment_db import build_database


REPO = Path(__file__).resolve().parents[1]


def test_build_database_preserves_protocol_boundaries(tmp_path: Path) -> None:
    database = tmp_path / "experiments.sqlite"
    build_database(REPO, database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        external = connection.execute(
            """
            SELECT mae_ev FROM evaluations
            WHERE model_id = 'M25'
              AND protocol_id = 'fixed_external_1977_v1'
              AND scope = 'all'
              AND target = 'average'
            """
        ).fetchone()
        internal = connection.execute(
            """
            SELECT mae_ev FROM evaluations
            WHERE model_id = 'M26'
              AND protocol_id = 'internal_random_model_specific'
              AND scope = 'test'
              AND target = 'average'
            """
        ).fetchone()
        comparable_a = connection.execute(
            """
            SELECT COUNT(*) FROM comparable_external
            WHERE model_id = 'M26'
            """
        ).fetchone()[0]

    assert external == (0.1000743234815492,)
    assert internal == (0.10243107626835506,)
    assert comparable_a == 0


def test_reuse_metadata_is_explicit(tmp_path: Path) -> None:
    database = tmp_path / "experiments.sqlite"
    build_database(REPO, database)
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT approximate_encoder_passes, reuse_mode
            FROM models WHERE model_id = 'M07'
            """
        ).fetchone()
    assert row == (4.0, "teacher")
