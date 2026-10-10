from copy import deepcopy

import pytest
import torch

from molgap.k1_precision_diagnostic import matmul_tf32, prediction_drift


@pytest.mark.parametrize("enabled", [False, True])
def test_tf32_restores_flags_on_exception(enabled):
    original = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError):
        with matmul_tf32(enabled):
            assert torch.backends.cuda.matmul.allow_tf32 is enabled
            assert torch.backends.cudnn.allow_tf32 == original[2]
            raise RuntimeError("fixture")
    assert original == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                        torch.backends.cudnn.allow_tf32)


def fixture():
    return dict(source_idx=torch.arange(4), target_eV=torch.ones(4), prediction_eV=torch.ones(4))


def test_prediction_drift_units_and_bound():
    reference, candidate = fixture(), fixture()
    candidate["prediction_eV"] += torch.tensor([0., .1, -.1, .2])
    report = prediction_drift(reference, candidate)
    assert report["mean_abs_prediction_delta_eV"] == pytest.approx(.1)
    assert report["train_sample_mae_delta_eV"] == pytest.approx(.1)
    assert not report["generalization_accuracy_qualified"]
    assert prediction_drift(reference, reference)["exact"]


@pytest.mark.parametrize("field", ["source_idx", "target_eV"])
def test_drift_rejects_misalignment(field):
    candidate = fixture()
    candidate[field][0] = 10
    with pytest.raises(ValueError):
        prediction_drift(fixture(), candidate)


def test_drift_rejects_nonfinite():
    candidate = fixture()
    candidate["prediction_eV"][0] = float("nan")
    with pytest.raises(ValueError):
        prediction_drift(fixture(), candidate)


def test_tf32_must_be_explicit():
    with pytest.raises(ValueError):
        with matmul_tf32(1):
            pass
