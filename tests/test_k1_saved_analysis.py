"""Synthetic-only checks for saved K1 artifact arithmetic."""

from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock

import pytest
import torch

from molgap.k1_saved_analysis import ARMS, bn_rows, trace_pair
from molgap import saved_diagnostic_closure as closure


def trace_inputs():
    traces, manifests, accepted = {}, {}, {}
    for side, arm in enumerate(ARMS):
        dev = ([0.3, 0.1, 0.2], [0.4, 0.15, 0.18])[side]
        epochs = []
        for epoch, score in enumerate(dev):
            epochs.append(dict(
                epoch=epoch, global_step=(epoch + 1) * 2,
                sample_presentations=(epoch + 1) * 8,
                train_mae_eV=0.5 - epoch * 0.1,
                development_mae_eV=score, lr=0.01 / (epoch + 1),
                seconds=10.0 + side,
                objective_components=dict(supervised_l1=0.2, disagreement=0.1,
                                          combined_loss=0.2 + side * 0.01),
            ))
        traces[arm] = dict(epochs=epochs)
        manifests[arm] = dict(
            contract=dict(arm=arm, loss_fingerprint=str(side),
                          binding_identity=dict(source_config_identity="synthetic-config-" + str(side),
                                                spec_identity="synthetic-spec",
                                                trajectory_id=arm), epochs=60,
                          steps_per_epoch=2, physical_batch_per_device=4,
                          seed=42, precision="fp32"),
            source_sha256="synthetic-source", next_epoch=3,
            best_development_mae_eV=min(dev),
        )
        accepted[arm] = dict(stage_mechanical_pass=True,
                             optimizer_steps=6, sample_presentations=24)
    inspection = dict(mechanical_stage_pass=True, disposition="ACTIVE_PARTIAL_STAGE",
                      source_sha256="synthetic-source", spec_identity="synthetic-spec", arms=accepted,
                      budget=dict(training_allocated_T4_hours=1.0))
    return traces, manifests, inspection


def prediction_inputs(n=100, start=20, ties=False):
    target = torch.zeros(n, dtype=torch.float64)
    residuals = {
        "selected_raw": [4.0, -1.0, 2.0, -2.0],
        "selected_clean": [2.0, -3.0, 2.0, -1.0],
        "final_raw": [5.0, -2.0, 1.0, -4.0],
        "final_clean": [3.0, -1.0, 2.0, -2.0],
    }
    predictions = {}
    for name, values in residuals.items():
        pred = torch.zeros(n, dtype=torch.float64) if ties else torch.tensor(
            (values * ((n + 3) // 4))[:n], dtype=torch.float64)
        predictions[name] = dict(source_idx=torch.arange(start, start + n),
                                 target_eV=target.clone(), prediction_eV=pred)
    return predictions, (start, start + n)


def test_trace_matching_axes_and_objectives():
    traces, manifests, inspection = trace_inputs()
    assert (manifests[ARMS[0]]["contract"]["binding_identity"]["source_config_identity"]
            != manifests[ARMS[1]]["contract"]["binding_identity"]["source_config_identity"])
    report = trace_pair(traces, manifests, inspection)
    assert report["completed_epochs"] == 3
    assert report["contract_epochs"] == 60
    assert [r["optimizer_steps"] for r in report["curve"]] == [2, 4, 6]
    assert [r["sample_presentations"] for r in report["curve"]] == [8, 16, 24]
    assert [r["learning_rate"] for r in report["curve"]] == pytest.approx([.01, .005, .01 / 3])
    assert report["curve"][0]["consistency_penalty_to_supervised_ratio"] == pytest.approx(.05)
    assert report["epoch_window_seconds"] == dict(zip(ARMS, [30.0, 33.0]))


def test_trace_selected_is_not_last():
    report = trace_pair(*trace_inputs())
    assert report["selected_stage_difference_meV"] == pytest.approx(50)
    assert report["same_step_last_difference_meV"] == pytest.approx(-20)
    assert report["curve"][-1]["best_so_far_consistency_minus_control_meV"] == pytest.approx(50)
    assert report["consistency_better_epochs"] == 1


def test_trace_rejects_axis_mismatch():
    for field, bad in (("epoch", 3), ("global_step", 5), ("sample_presentations", 17)):
        traces, manifests, inspection = trace_inputs()
        traces[ARMS[1]]["epochs"][1][field] = bad
        with pytest.raises(ValueError, match="exposure"):
            trace_pair(traces, manifests, inspection)


def test_trace_rejects_lr_mismatch():
    traces, manifests, inspection = trace_inputs()
    traces[ARMS[1]]["epochs"][0]["lr"] = .02
    with pytest.raises(ValueError, match="learning rates"):
        trace_pair(traces, manifests, inspection)


def test_trace_rejects_nonfinite_metrics_and_objectives():
    for field in ("train_mae_eV", "development_mae_eV", "lr", "seconds",
                  "supervised_l1", "disagreement", "combined_loss"):
        for bad in (float("nan"), float("inf")):
            traces, manifests, inspection = trace_inputs()
            row = traces[ARMS[1]]["epochs"][0]
            container = row if field in row else row["objective_components"]
            container[field] = bad
            with pytest.raises(ValueError, match="Invalid trace metric"):
                trace_pair(traces, manifests, inspection)


def test_trace_rejects_incorrect_objective_arithmetic():
    for arm in ARMS:
        traces, manifests, inspection = trace_inputs()
        traces[arm]["epochs"][0]["objective_components"]["combined_loss"] += .02
        with pytest.raises(ValueError, match="Objective arithmetic"):
            trace_pair(traces, manifests, inspection)


def test_trace_rejects_contract_mismatch():
    for field, bad in (("seed", 7), ("precision", "bf16"), ("epochs", 30)):
        traces, manifests, inspection = trace_inputs()
        manifests[ARMS[1]]["contract"][field] = bad
        with pytest.raises(ValueError, match="contracts differ"):
            trace_pair(traces, manifests, inspection)
    traces, manifests, inspection = trace_inputs()
    manifests[ARMS[1]]["contract"]["binding_identity"]["spec_identity"] = "different"
    with pytest.raises(ValueError, match="binding identity"):
        trace_pair(traces, manifests, inspection)
    traces, manifests, inspection = trace_inputs()
    inspection["spec_identity"] = "different-accepted-spec"
    with pytest.raises(ValueError, match="Spec identity"):
        trace_pair(traces, manifests, inspection)


def test_trace_rejects_unaccepted_or_inconsistent_endpoints():
    for mode in ("acceptance", "source", "length", "cursor", "best", "counters"):
        traces, manifests, inspection = trace_inputs()
        if mode == "acceptance":
            inspection["mechanical_stage_pass"] = False
        elif mode == "source":
            manifests[ARMS[1]]["source_sha256"] = "different"
        elif mode == "length":
            traces[ARMS[1]]["epochs"].pop()
        elif mode == "cursor":
            manifests[ARMS[1]]["next_epoch"] = 4
        elif mode == "best":
            manifests[ARMS[1]]["best_development_mae_eV"] = .01
        else:
            inspection["arms"][ARMS[1]]["optimizer_steps"] = 7
        with pytest.raises(ValueError):
            trace_pair(traces, manifests, inspection)


def test_bn_rejects_membership_order_and_index_dtype():
    for mode in ("order", "membership", "dtype"):
        predictions, bounds = prediction_inputs()
        ids = predictions["final_clean"]["source_idx"]
        if mode == "order":
            ids = ids.flip(0)
        elif mode == "membership":
            ids = ids + 1
        else:
            ids = ids.double()
        predictions["final_clean"]["source_idx"] = ids
        with pytest.raises(ValueError, match="membership/order"):
            bn_rows(predictions, bounds)


def test_bn_rejects_target_mismatch():
    predictions, bounds = prediction_inputs()
    predictions["final_clean"]["target_eV"][0] = .1
    with pytest.raises(ValueError, match="Paired targets"):
        bn_rows(predictions, bounds)


def test_bn_rejects_nonfinite_or_wrong_length():
    for field in ("target_eV", "prediction_eV"):
        for bad in (float("nan"), float("inf"), "short"):
            predictions, bounds = prediction_inputs()
            if bad == "short":
                predictions["final_clean"][field] = predictions["final_clean"][field][:-1]
            else:
                predictions["final_clean"][field][0] = bad
            with pytest.raises(ValueError, match="length/finiteness"):
                bn_rows(predictions, bounds)


def test_bn_signed_decomposition_and_means():
    predictions, bounds = prediction_inputs()
    before = deepcopy(predictions)
    report, rows = bn_rows(predictions, bounds)
    expected = {
        "bn49_gain": [2., -2., 0., 1.],
        "bn60_gain": [2., 1., -1., 2.],
        "clean_late_deterioration": [1., -2., 0., 1.],
        "raw_late_deterioration": [1., 1., -1., 2.],
    }
    for name, pattern in expected.items():
        delta = torch.tensor(pattern * 25, dtype=torch.float64)
        torch.testing.assert_close(rows[name + "_eV"], delta)
        assert report[name]["mean_meV"] == pytest.approx(delta.mean().item() * 1000)
    torch.testing.assert_close(rows["raw_late_deterioration_eV"],
                               rows["clean_late_deterioration_eV"]
                               + rows["bn60_gain_eV"] - rows["bn49_gain_eV"])
    assert report["bn49_gain"]["positive_rows_fraction"] == .5
    assert report["bn49_gain"]["negative_rows_fraction"] == .25
    assert report["bn49_gain"]["equal_rows_fraction"] == .25
    assert report["signed_residual_mean_eV"]["selected_raw"] == pytest.approx(.75)
    for name in predictions:
        for field in predictions[name]:
            torch.testing.assert_close(predictions[name][field], before[name][field])


def test_bn_ties_small_bounds_and_legacy_keys():
    predictions, bounds = prediction_inputs(n=4, start=7, ties=True)
    for value in predictions.values():
        value["target"] = value.pop("target_eV")
        value["prediction"] = value.pop("prediction_eV")
    report, rows = bn_rows(predictions, bounds)
    assert report["rows"] == 4
    assert rows["source_idx"].tolist() == [7, 8, 9, 10]
    assert report["bn49_gain"]["mean_meV"] == 0
    assert report["bn49_gain"]["equal_rows_fraction"] == 1
    assert report["bn49_gain_vs_clean_late_pearson"] is None
    assert report["clean_late_worsened_given_bn49_improved_fraction"] is None
    assert all(tail["net_gain_share"] is None for tail in report["baseline_error_tails"])


def test_bn_stratified_contributions_sum_to_total():
    predictions, bounds = prediction_inputs()
    report, _ = bn_rows(predictions, bounds)
    for name in ("baseline_error_strata", "target_quartiles", "bn49_gain_quartiles"):
        groups = report[name]
        assert sum(g["rows"] for g in groups) == report["rows"]
        assert sum(g["bn49_net_contribution_meV"] for g in groups) == pytest.approx(
            report["bn49_gain"]["mean_meV"])
        for metric in ("bn60_gain", "clean_late_deterioration", "raw_late_deterioration"):
            weighted = sum(g[metric + "_meV"] * g["rows"] for g in groups) / report["rows"]
            assert weighted == pytest.approx(report[metric]["mean_meV"])


def test_bn_shared_term_covariance_decomposition():
    predictions, bounds = prediction_inputs()
    report, rows = bn_rows(predictions, bounds)
    covariance = report["shared_term_covariance_eV_squared"]
    gain = rows["bn49_gain_eV"]
    late = rows["clean_late_deterioration_eV"]
    observed = ((gain - gain.mean()) * (late - late.mean())).mean().item()
    assert covariance["observed"] == pytest.approx(observed)
    assert sum(covariance["terms"].values()) == pytest.approx(observed)


def adapter_fixture(tmp_path, monkeypatch, snapshots=None):
    experiment = tmp_path / "experiments" / "synthetic" / "attempt_002"
    (experiment / "rml").mkdir(parents=True)
    (experiment / "source_snapshot").mkdir()
    (experiment / "inputs.json").write_text(
        json.dumps(dict(snapshot_files=snapshots or {})), encoding="utf-8")
    (experiment / "rml" / "trajectory.json").write_text("{}", encoding="utf-8")
    finalize = Mock(side_effect=AssertionError("Unexpected finalization"))
    atomic_json = Mock(side_effect=AssertionError("Unexpected canonical write"))
    git = Mock(side_effect=AssertionError("Unexpected git call"))
    monkeypatch.setattr(closure, "finalize", finalize)
    monkeypatch.setattr(closure, "atomic_json", atomic_json)
    monkeypatch.setattr(closure.subprocess, "check_output", git)
    return experiment, finalize, atomic_json, git


def test_adapter_rejects_stale_frozen_snapshot_before_writes(tmp_path, monkeypatch):
    original = b'{"metadata": "frozen synthetic value"}'
    experiment, finalize, atomic_json, git = adapter_fixture(
        tmp_path, monkeypatch,
        {"trace.json": hashlib.sha256(original).hexdigest()})
    snapshot = experiment / "source_snapshot" / "trace.json"
    snapshot.write_bytes(b'{"metadata": "changed synthetic value"}')
    before = snapshot.read_bytes()
    with pytest.raises(ValueError, match="Frozen metadata hash differs"):
        closure.close_saved_diagnostic(tmp_path, experiment)
    assert snapshot.read_bytes() == before
    finalize.assert_not_called()
    atomic_json.assert_not_called()
    git.assert_not_called()


def test_adapter_rejects_failure_and_completed_result_before_writes(tmp_path, monkeypatch):
    experiment, finalize, atomic_json, git = adapter_fixture(tmp_path, monkeypatch)
    (experiment / "results").mkdir()
    failure = experiment / "failure.json"
    result = experiment / "results" / "result.json"
    failure.write_text('{"status": "synthetic failure"}', encoding="utf-8")
    result.write_text('{"status": "complete"}', encoding="utf-8")
    before = {path: path.read_bytes() for path in (failure, result)}
    with pytest.raises(ValueError, match="cannot coexist"):
        closure.close_saved_diagnostic(tmp_path, experiment)
    assert {path: path.read_bytes() for path in before} == before
    finalize.assert_not_called()
    atomic_json.assert_not_called()
    git.assert_not_called()


def test_adapter_finalized_retry_delegates_without_overwriting(tmp_path, monkeypatch):
    experiment, finalize, atomic_json, git = adapter_fixture(tmp_path, monkeypatch)
    finalized = experiment / "rml" / "rml_finalized"
    finalized.mkdir()
    (finalized / "finalization.json").write_text("{}", encoding="utf-8")
    for name in ("acceptance.json", "terminal.json"):
        (experiment / name).write_text('{"synthetic": "immutable"}', encoding="utf-8")
    # Invalid inputs prove the retry guard bypasses reconstruction entirely.
    (experiment / "inputs.json").write_text("not JSON", encoding="utf-8")
    before = {path: path.read_bytes() for path in experiment.rglob("*") if path.is_file()}
    receipt = dict(status="ALREADY_FINALIZED", synthetic=True)
    finalize.side_effect = None
    finalize.return_value = receipt
    assert closure.close_saved_diagnostic(tmp_path, experiment) is receipt
    rel = experiment.relative_to(tmp_path).as_posix()
    finalize.assert_called_once_with(tmp_path.resolve(), f"{rel}/rml", f"{rel}/terminal.json")
    atomic_json.assert_not_called()
    git.assert_not_called()
    assert {path: path.read_bytes() for path in experiment.rglob("*") if path.is_file()} == before
