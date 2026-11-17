from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from molgap.database import build_database_ledger, run_database_build


def _predictor(smiles, **kwargs):
    # Return rows out of order so the source-index join is tested explicitly.
    valid_idx = np.asarray([3, 1, 0, 2], dtype=int)
    predictions = np.asarray([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]], dtype=np.float32)
    experts = np.asarray(
        [
            [[10, 11, 12], [12, 13, 14], [14, 15, 16]],
            [[20, 21, 22], [20, 21, 22], [20, 21, 22]],
            [[30, 31, 32], [32, 33, 34], [34, 35, 36]],
            [[40, 41, 42], [40, 41, 42], [40, 41, 42]],
        ],
        dtype=np.float32,
    )
    assert len(smiles) == 4
    return valid_idx, predictions, experts


def _graph_builder(smiles):
    return None if smiles == "C1CC1" else object()


def _input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["a", "a-copy", "silicon", "broken", "graph-fail", "small"],
            "smiles": [
                "Clc1ccc(cc1)C(=O)Nc1ccccc1",
                "Clc1ccc(cc1)C(=O)Nc1ccccc1",
                "CC[Si](C)(C)C",
                "not_a_smiles",
                "C1CC1",
                "CCO",
            ],
        }
    )


def test_ledger_preserves_rows_and_joins_predictions_by_source_row():
    ledger, stats = build_database_ledger(
        _input_frame(),
        models={},
        id_column="name",
        graph_builder=_graph_builder,
        predictor=_predictor,
    )

    assert len(ledger) == 6
    assert stats["predicted_rows"] == 4
    assert stats["rejected_rows"] == 2
    assert ledger.loc[0, "homo"] == 30.0
    assert ledger.loc[5, "homo"] == 10.0
    assert ledger.loc[3, "rejection_reason"] == "invalid_smiles"
    assert ledger.loc[4, "rejection_reason"] == "graph_failed"
    assert bool(ledger.loc[2, "allowed_elements"]) is False
    assert bool(ledger.loc[2, "in_domain"]) is False
    assert bool(ledger.loc[0, "duplicate_identity"]) is True
    assert bool(ledger.loc[1, "duplicate_identity"]) is True
    assert stats["all_predicted_values_finite"] is True


def test_build_writes_atomic_csv_and_hash_manifest(tmp_path):
    input_csv = tmp_path / "input.csv"
    _input_frame().to_csv(input_csv, index=False)
    out_dir = tmp_path / "build"

    manifest = run_database_build(
        input_csv,
        out_dir,
        models={},
        id_column="name",
        graph_builder=_graph_builder,
        predictor=_predictor,
    )

    predictions = out_dir / "predictions.csv"
    manifest_path = out_dir / "manifest.json"
    assert predictions.is_file()
    assert manifest_path.is_file()
    assert manifest["status"] == "complete"
    assert manifest["row_counts"]["input_rows"] == 6
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert manifest["outputs"]["predictions_csv"]["sha256"] == digest
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["row_contract"]["ledger_preserves_input_rows"] is True

    with pytest.raises(FileExistsError):
        run_database_build(
            input_csv,
            out_dir,
            models={},
            id_column="name",
            graph_builder=_graph_builder,
            predictor=_predictor,
        )
