from copy import deepcopy

import pytest
import torch

from molgap.k1_precision_diagnostic import matmul_tf32, prediction_drift, qualify_precision_report


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


def report_fixture():
    cases = {}
    for name in ("baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", "cpu_layout", "tf32", "tf32_layout"):
        changed = name.startswith("tf32")
        cases[name] = dict(status="numerical_mismatch" if changed else "qualified", equivalence=dict(exact=not changed),
            timings=[dict(seconds_per_step=.9 if changed else 1.) for _ in range(4)],
            median_seconds_per_step=.9 if changed else 1.)
    return dict(status="complete", cases=cases)


def test_precision_acceptance_keeps_tf32_contract_separate():
    qualified = qualify_precision_report(report_fixture())
    assert qualified["tf32_incremental_time_reduction"] == pytest.approx(.1)
    assert not qualified["strict_fp32_admission"]


@pytest.mark.parametrize("failure", ["failed", "missing_window", "bad_timing", "nonexact_control", "missing_case"])
def test_complete_top_level_never_masks_case_failure(failure):
    report = report_fixture()
    if failure == "failed":
        report["cases"]["tf32"]["status"] = "failed_or_unsupported"
    elif failure == "missing_window":
        report["cases"]["tf32"]["timings"].pop()
    elif failure == "bad_timing":
        report["cases"]["tf32"]["timings"][0]["seconds_per_step"] = float("nan")
    elif failure == "nonexact_control":
        report["cases"]["cpu_layout"]["equivalence"]["exact"] = False
    else:
        del report["cases"]["tf32"]
    with pytest.raises(ValueError):
        qualify_precision_report(report)
