import pytest
import torch

from experiments.pcqm_gptrans_100k_transfer_control.analyze_pair import paired_metrics


def _record(indices, predictions):
    return {
        "source_idx": torch.tensor(indices, dtype=torch.long),
        "target_eV": torch.tensor([1.0, 2.0, 3.0]),
        "prediction_eV": torch.tensor(predictions),
    }


def test_paired_metrics_uses_same_rows():
    reference = _record([100000, 100001, 100002], [2.0, 2.0, 3.0])
    candidate = _record([100000, 100001, 100002], [1.0, 2.0, 3.0])
    result = paired_metrics(reference, candidate, 100000, 100003)
    assert result["reference_mae_eV"] == pytest.approx(1 / 3)
    assert result["candidate_mae_eV"] == 0.0
    assert result["paired_gain_eV"] == pytest.approx(1 / 3)


def test_paired_metrics_rejects_wrong_row_order():
    reference = _record([100000, 100001, 100002], [2.0, 2.0, 3.0])
    candidate = _record([100001, 100000, 100002], [1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="source-index order"):
        paired_metrics(reference, candidate, 100000, 100003)
