import copy

import pytest
import torch

from molgap.edge_state_training_core import EpochPermutationBatchSampler
from molgap.k1_single_ema_acceptance import (
    _inspect_resume, compare_predictions, export_canonical_traces, inspect_predictions, inspect_trace,
)
from molgap.pcqm_500k_v4_evidence import schedule
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.research_memory.trace import load_canonical_trace


def prediction():
    return {"source_idx": torch.arange(10, 14), "target_eV": torch.ones(4),
            "prediction_eV": torch.full((4,), 1.25), "mae_eV": 0.25}


def trace():
    steps = [{"step": i + 1, "epoch": i // 2, "next_batch": i % 2 + 1,
              "samples": (i + 1) * 2, "loss": 0.5, "seconds": 0.1,
              "allocation_seconds": i + 1.0} for i in range(3)]
    epochs = [{"epoch": 0, "steps": 2, "samples": 4, "lr": schedule(0),
               "train_normalized_l1": 0.5, "raw_mae_eV": 0.2, "calibrated_mae_eV": 0.1,
               "round_seconds": 2.0, "allocation_seconds": 2.5,
               "order_sha256": EpochPermutationBatchSampler(4, 2, seed=42, epoch=0).order_sha256}]
    state = {"epoch": 1, "offset": 1, "steps": 3, "samples": 6,
             "trace": epochs, "step_trace": steps, "train_loss_sum": 0.5,
             "sampler": EpochPermutationBatchSampler(4, 2, seed=42, epoch=1).state_for(1)}
    return state, {"epochs": copy.deepcopy(epochs), "steps": copy.deepcopy(steps)}


def test_predictions_finite_aligned():
    assert inspect_predictions(prediction(), start=10, rows=4)["mae_eV"] == 0.25


@pytest.mark.parametrize("field", ["source_idx", "target_eV", "prediction_eV", "mae_eV"])
def test_prediction_corruption_rejected(field):
    payload = prediction()
    if field == "source_idx":
        payload[field] = payload[field].flip(0)
    elif field == "mae_eV":
        payload[field] = 0.3
    else:
        payload[field][0] = float("nan")
    with pytest.raises(ValueError):
        inspect_predictions(payload, start=10, rows=4)


def test_paired_bootstrap_scope():
    ref, candidate = prediction(), prediction()
    candidate["prediction_eV"].fill_(1.125)
    result = compare_predictions(ref, candidate, n_bootstrap=20)
    assert result["reference_minus_candidate_eV"] == 0.125
    assert result["candidate_minus_reference_bootstrap"]["ci95"] == [-0.125, -0.125]
    assert "not training-seed variance" in result["uncertainty_scope"]


def test_pair_targets_rejected():
    ref, candidate = prediction(), prediction()
    candidate["target_eV"][0] += 1
    with pytest.raises(ValueError, match="targets"):
        compare_predictions(ref, candidate)


def test_trace_partial_exposure():
    state, retained = trace()
    result = inspect_trace(state, retained, train_rows=4, batch_size=2)
    assert (result["completed_epochs"], result["partial_batches"], result["samples"]) == (1, 1, 6)


@pytest.mark.parametrize("corruption", ["step", "lr", "sampler", "loss", "count", "accumulator"])
def test_trace_corruption_rejected(corruption):
    state, retained = trace()
    if corruption == "step":
        retained["steps"][1]["samples"] = 3
        state["step_trace"] = retained["steps"]
    elif corruption == "lr":
        retained["epochs"][0]["lr"] = 0.001
        state["trace"] = retained["epochs"]
    elif corruption == "sampler":
        state["sampler"]["order_sha256"] = "0" * 64
    elif corruption == "loss":
        retained["steps"][0]["loss"] = float("inf")
        state["step_trace"] = retained["steps"]
    elif corruption == "count":
        state["samples"] = 100
    else:
        state["train_loss_sum"] = 0.0
    with pytest.raises(ValueError):
        inspect_trace(state, retained, train_rows=4, batch_size=2)


def resume_state():
    return {"model": {"weight": torch.ones(1)}, "ema": None, "epoch": 1, "offset": 0,
            "steps": 2, "rng": {"python": (3, (1,), None), "numpy": ("MT19937",),
                "torch": torch.ones(2, dtype=torch.uint8), "cuda": [torch.ones(2, dtype=torch.uint8)]},
            "optimizer": {"state": {0: {"step": torch.tensor(2.), "exp_avg": torch.ones(1),
                "exp_avg_sq": torch.ones(1)}}, "param_groups": [{"params": [0], "lr": schedule(0),
                "weight_decay": 1e-5, "betas": (0.9, 0.999), "eps": 1e-8,
                "foreach": False, "fused": False, "amsgrad": False}]}}


def test_resume_state_inspection_only():
    assert "no local runtime replay" in _inspect_resume(resume_state(), "reference")["resume_contents"]


@pytest.mark.parametrize("field", ["rng", "optimizer", "model", "ema"])
def test_resume_corruption_rejected(field):
    state = resume_state()
    if field == "rng":
        state["rng"]["cuda"] = []
    elif field == "optimizer":
        state["optimizer"]["state"][0]["step"] = torch.tensor(3.)
    elif field == "model":
        state["model"]["weight"][0] = float("nan")
    else:
        state["ema"] = {"weight": torch.ones(1)}
    with pytest.raises((ValueError, RuntimeError)):
        _inspect_resume(state, "reference")


def test_canonical_trace_export_idempotent_and_tamper_rejected(tmp_path):
    attempt, owner, output = tmp_path / "attempt", tmp_path / "owner", tmp_path / "accepted"
    _, retained = trace()
    raw = {arm: retained for arm in ("reference", "ema999")}
    atomic_json(attempt / "worker/trace.json", raw)
    atomic_json(attempt / "worker/terminal.json", {"elapsed_allocation_seconds": 4.0})
    atomic_json(owner / "rml/trajectory.json", {"trajectory_id": "synthetic-k1"})
    acceptance = {"run_id": "synthetic-k1", "verified_local_artifact_sha256": {
        "trace.json": sha256_file(attempt / "worker/trace.json"), "last.pt": "0" * 64},
        "exposure": {arm: {"steps": 3, "samples": 6, "completed_epochs": 1, "partial_batches": 1}
                     for arm in raw}}
    first = export_canonical_traces(attempt, owner, output, acceptance)
    assert first == export_canonical_traces(attempt, owner, output, acceptance)
    path = output / "ema999_trace.json"
    canonical = load_canonical_trace(path)
    assert canonical["observations"][0]["ema_dev_metric"] == 0.1
    assert canonical["observations"][0]["live_dev_metric"] is None
    canonical["observations"][0]["ema_dev_metric"] = 0.2
    atomic_json(path, canonical)
    with pytest.raises(ValueError, match="differs from accepted"):
        export_canonical_traces(attempt, owner, output, acceptance)
