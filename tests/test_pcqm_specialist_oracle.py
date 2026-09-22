from __future__ import annotations

import numpy as np

from molgap.pcqm_specialist_oracle import _fold_ids, _greedy_oracle, _safe_auc


def test_greedy_oracle_is_monotonic() -> None:
    target = np.asarray([0.0, 0.0, 0.0, 0.0])
    predictions = {
        "neural_atom_k1_v4": np.asarray([1.0, 1.0, 1.0, 1.0]),
        "left": np.asarray([0.0, 0.0, 2.0, 2.0]),
        "right": np.asarray([2.0, 2.0, 0.0, 0.0]),
    }
    curve = _greedy_oracle(target, predictions, list(predictions))
    maes = [row["oracle_mae_eV"] for row in curve]
    assert maes == sorted(maes, reverse=True)
    assert maes[-1] == 0.0


def test_auc_selects_signal_orientation() -> None:
    labels = np.asarray([False, False, True, True])
    signal = np.asarray([4.0, 3.0, 2.0, 1.0])
    result = _safe_auc(labels, signal)
    assert result["orientation"] == "low"
    assert result["average_precision"] == 1.0


def test_fold_ids_are_stable_and_bounded() -> None:
    source = np.arange(100_000, 150_000, dtype=np.int64)
    first = _fold_ids(source)
    second = _fold_ids(source.copy())
    assert np.array_equal(first, second)
    assert set(first) == set(range(5))

