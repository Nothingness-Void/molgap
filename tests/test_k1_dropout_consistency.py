"""Synthetic CPU checks for K1's registered two-pass dropout objective."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from molgap import experiment_execution, k1_dropout_consistency, k1_screen_training
from molgap.experiment_spec import (
    ExperimentSpec, FAMILIES, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL,
)
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import capture_rng_state, restore_rng_state
from molgap.v4_runtime import normalized_source_sha256


@pytest.mark.parametrize("mode,weight", [
    ("dropout_mean2", 0.0),
    ("dropout_consistency2", 0.1),
])
def test_objective_matches_independent_loss_and_both_prediction_gradients(mode, weight):
    first = torch.tensor([-1.1, 0.2, 3.5], requires_grad=True)
    second = torch.tensor([0.4, -0.9, 1.2], requires_grad=True)
    target = torch.tensor([0.6, 1.0, 2.6])

    loss, disagreement = k1_dropout_consistency.objective(
        first, second, target, mode=mode,
    )
    expected_disagreement = (first - second).square().mean()
    expected_loss = 0.5 * (
        functional.l1_loss(first, target) + functional.l1_loss(second, target)
    ) + weight * expected_disagreement
    actual_gradients = torch.autograd.grad(loss, (first, second))

    count = first.numel()
    difference = first.detach() - second.detach()
    expected_first_gradient = (
        (first.detach() - target).sign() / (2 * count)
        + 2 * weight * difference / count
    )
    expected_second_gradient = (
        (second.detach() - target).sign() / (2 * count)
        - 2 * weight * difference / count
    )
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(disagreement, expected_disagreement)
    torch.testing.assert_close(actual_gradients[0], expected_first_gradient)
    torch.testing.assert_close(actual_gradients[1], expected_second_gradient)


def test_zero_weight_is_mean_of_two_l1_losses_not_l1_of_averaged_prediction():
    first = torch.tensor([0.0, 2.0])
    second = torch.tensor([2.0, 4.0])
    target = torch.tensor([1.0, 3.0])

    loss, disagreement = k1_dropout_consistency.objective(
        first, second, target, mode="dropout_mean2",
    )

    expected = 0.5 * (
        functional.l1_loss(first, target) + functional.l1_loss(second, target)
    )
    averaged_prediction_loss = functional.l1_loss((first + second) / 2, target)
    torch.testing.assert_close(loss, expected)
    assert loss.item() == pytest.approx(1.0)
    assert averaged_prediction_loss.item() == pytest.approx(0.0)
    assert not torch.equal(loss, averaged_prediction_loss)
    torch.testing.assert_close(disagreement, (first - second).square().mean())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_objective_rejects_nonfinite_result(bad_value):
    first = torch.tensor([bad_value, 1.0])
    second = torch.tensor([0.0, 2.0])
    target = torch.tensor([0.5, 1.5])

    with pytest.raises(ValueError, match="Nonfinite two-pass objective"):
        k1_dropout_consistency.objective(
            first, second, target, mode="dropout_consistency2",
        )


class TinyDropoutRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features):
        return self.dropout(self.linear(features)).view(-1)


def _synthetic_batch():
    return SimpleNamespace(
        x=torch.tensor([
            [1.0, 0.5, -0.5, 0.25],
            [-0.5, 0.25, 1.0, -1.0],
            [0.75, -0.25, 0.5, 1.5],
            [1.5, 0.25, -0.75, 0.5],
        ]),
        y=torch.tensor([0.2, -0.1, 0.7, 0.4]),
    )


def _install_tiny_forward(monkeypatch):
    observed = []

    def forward(model, batch):
        prediction = model(batch.x)
        observed.append(prediction.detach().clone())
        return prediction

    monkeypatch.setattr(k1_screen_training, "_forward", forward)
    return observed


def test_optimizer_step_updates_once_from_two_independent_dropout_forwards(monkeypatch):
    observed = _install_tiny_forward(monkeypatch)
    torch.manual_seed(812)
    model = TinyDropoutRegressor().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    batch = _synthetic_batch()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    loss, absolute_error, rows = k1_screen_training._optimizer_step(
        model, optimizer, batch, mean=0.0, std=1.0, mode="dropout_consistency2",
    )

    assert len(observed) == 2
    assert not torch.equal(observed[0], observed[1])
    target = batch.y.view(-1)
    expected_loss, _ = k1_dropout_consistency.objective(
        observed[0], observed[1], target, mode="dropout_consistency2",
    )
    expected_absolute = 0.5 * (
        (observed[0] - target).abs() + (observed[1] - target).abs()
    ).sum()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(absolute_error, expected_absolute)
    assert rows == target.numel()
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert {int(state["step"].item()) for state in optimizer.state.values()} == {1}


def test_model_adamw_and_rng_resume_reproduce_the_exact_next_step(monkeypatch):
    _install_tiny_forward(monkeypatch)
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    model = TinyDropoutRegressor().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-5)
    batch = _synthetic_batch()

    k1_screen_training._optimizer_step(
        model, optimizer, batch, mean=0.0, std=1.0, mode="dropout_consistency2",
    )
    snapshot = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": copy.deepcopy(capture_rng_state()),
    }
    uninterrupted_loss, _, _ = k1_screen_training._optimizer_step(
        model, optimizer, batch, mean=0.0, std=1.0, mode="dropout_consistency2",
    )
    uninterrupted_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    model.load_state_dict(snapshot["model"], strict=True)
    optimizer.load_state_dict(snapshot["optimizer"])
    restore_rng_state(snapshot["rng"])
    resumed_loss, _, _ = k1_screen_training._optimizer_step(
        model, optimizer, batch, mean=0.0, std=1.0, mode="dropout_consistency2",
    )

    torch.testing.assert_close(resumed_loss, uninterrupted_loss, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, uninterrupted_state[name], rtol=0, atol=0)
    assert {int(state["step"].item()) for state in optimizer.state.values()} == {2}


def _minimal_k1_spec(mode: str, addon_name: str):
    contract = FAMILIES[("neural_atom_k1", "2")]
    config = k1_dropout_consistency.configuration(mode)
    addon = {
        "name": addon_name,
        "version": "1",
        "config": config,
        "source_sha256": normalized_source_sha256(Path(k1_dropout_consistency.__file__)),
    }
    arm = {
        "arm_id": "synthetic-candidate",
        "scientific_role": "candidate",
        "family": {"name": "neural_atom_k1", "version": "2"},
        "base": {"name": "synthetic-base", "version": "1", "sha256": "a" * 64},
        "initialization": {
            "kind": "frozen_state",
            "seed": k1_screen_training.SEED,
            "state_sha256": k1_screen_training.INITIAL_STATE_SHA256,
        },
        "data": {
            "dataset": {"name": "pcqm4mv2", "version": "1", "sha256": "a" * 64},
            "split": {"name": "synthetic-split", "version": "1", "sha256": "a" * 64},
            "roles": [
                {"role": role, "membership_sha256": "a" * 64,
                 "row_order_sha256": "b" * 64, "usage_sha256": "c" * 64}
                for role in contract.roles
            ],
            "feature_schema": contract.feature_schema,
            "feature_sha256": "a" * 64,
            "target": "pcqm4mv2-gap-eV-direct",
        },
        "training": {
            "recipe": {"name": contract.recipe, "version": "1", "sha256": "d" * 64},
            "overrides": {},
            "objective": {
                "name": "normalized-gap-l1", "version": "1",
                "sha256": k1_dropout_consistency.objective_identity(mode),
            },
            "sampler": {
                "name": contract.sampler, "version": "1",
                "sha256": k1_screen_training.ROW_ORDER_FINGERPRINT,
            },
            "transform": {
                "name": contract.transform, "version": "1", "sha256": "e" * 64,
            },
        },
        "addons": [addon],
        "addon_semantics": "ordered",
    }
    payload = {
        "schema_version": SCHEMA_VERSION_V2,
        "experiment_id": "synthetic-k1-dropout",
        "logical_run_id": "synthetic-k1-dropout-run",
        "arms": [arm],
        "platform": {
            "name": "local", "accelerator": "synthetic-cpu", "device_count": 1,
            "cpu_cores": 2, "memory_gib": 8,
            "atomic_checkpoints": True, "retrievable_chunks": True,
        },
        "prospective": {"arms": [{
            "arm_id": "synthetic-candidate", "trajectory_id": "synthetic-trajectory",
            "plan_spec_ref": "plans/synthetic-candidate.json",
            "plan_spec_sha256": "a" * 64,
            "output": "experiments/synthetic-k1-dropout",
        }]},
        "evidence": {
            "policy": {"name": "molgap-v5", "version": "1", "sha256": "a" * 64},
            "required_artifacts": [
                "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact",
            ],
        },
        "terminal_protocol": TERMINAL_PROTOCOL,
    }
    return ExperimentSpec(payload), payload


@pytest.mark.parametrize("addon_name,mode", [
    ("k1_dropout_mean2", "dropout_mean2"),
    ("k1_dropout_consistency2", "dropout_consistency2"),
])
def test_static_recipe_binds_addon_identity_and_rejects_tampering(addon_name, mode):
    recipe = experiment_execution.build_family_recipe(
        ("neural_atom_k1", "2"), addon=addon_name,
        source_idx_sha256="2" * 64, target_sha256="3" * 64,
    )
    adapter = experiment_execution.TRAINING_ADAPTERS[("neural_atom_k1", "2")]
    spec, payload = _minimal_k1_spec(mode, addon_name)
    arm = spec.to_dict()["arms"][0]
    assert adapter.mode(arm) == mode
    assert recipe["dropout_objective"] == k1_dropout_consistency.configuration(mode)
    assert recipe["loss_identity"] == k1_dropout_consistency.objective_identity(mode)
    assert recipe["gradient_loss_row_evaluations"] == 2 * k1_screen_training.SAMPLE_EXPOSURE
    assert k1_screen_training.validate_screen_recipe(
        spec, "synthetic-candidate", recipe,
    )["recipe"] == "frozen_recipe_verified"

    altered_recipe = copy.deepcopy(recipe)
    altered_recipe["gradient_loss_row_evaluations"] -= 1
    with pytest.raises(ValueError, match="two-pass objective/exposure"):
        k1_screen_training.validate_screen_recipe(
            spec, "synthetic-candidate", altered_recipe,
        )

    altered_payload = copy.deepcopy(payload)
    altered_payload["arms"][0]["addons"][0]["source_sha256"] = "f" * 64
    altered_spec = ExperimentSpec(altered_payload)
    with pytest.raises(ValueError, match="executable two-pass objective"):
        k1_screen_training.validate_screen_recipe(
            altered_spec, "synthetic-candidate", recipe,
        )


def _write_dropout_preflight(directory: Path, *, overhead=0.75, dropout_signal=True):
    provenance = {
        "runtime_fingerprint": "1" * 64,
        "mode": "dropout_consistency2",
        "context": {"arm_id": "synthetic-candidate"},
    }
    architecture = {
        "accepted": True,
        "mode": "dropout_consistency2",
        "repeatability": {"accepted": True},
        "resume_roundtrip": {"accepted": True},
        "zero_initialization_delta": 0.0,
        "maximum_overhead_fraction": 2.0,
        "synchronized_step_overhead_fraction": overhead,
        "selected_state_roundtrip_delta": 0.0,
        "dropout_signal": {"accepted": dropout_signal},
    }
    saved = {
        "status": "accepted",
        "calibration_checks_passed": True,
        "runtime_fingerprint": provenance["runtime_fingerprint"],
        "provenance_sha256": canonical_fingerprint(provenance),
        "architecture_sha256": canonical_fingerprint(architecture),
    }
    values = {
        "runtime_provenance": provenance,
        "runtime_certificate": saved,
        "architecture_preflight": architecture,
        "runtime_manifest": {"runtime_fingerprint": provenance["runtime_fingerprint"]},
    }
    for name, value in values.items():
        (directory / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    return provenance


@pytest.mark.parametrize(
    "overhead,dropout_signal,expected_error",
    [
        (0.75, False, "two-pass preflight is incomplete"),
        (2.01, True, "calibration evidence differs from the gate"),
    ],
)
def test_two_pass_preflight_fails_closed_on_signal_or_cost_gate(
    tmp_path, overhead, dropout_signal, expected_error,
):
    provenance = _write_dropout_preflight(
        tmp_path, overhead=overhead, dropout_signal=dropout_signal,
    )

    with pytest.raises(ValueError, match=expected_error):
        k1_screen_training.validate_runtime_preflight(tmp_path, provenance)


def test_training_only_preflight_skips_development_shards_but_checks_manifest_identity(
    monkeypatch, tmp_path,
):
    train_item = {
        "role": "train", "file": "train-shard.pt", "sha256": "a" * 64, "rows": 1,
    }
    development_item = {
        "role": "development", "file": "development-shard.pt",
        "sha256": "b" * 64, "rows": 1,
    }
    manifest = {"geometry_shards": [train_item, development_item]}
    aggregate = hashlib.sha256()
    for item in manifest["geometry_shards"]:
        aggregate.update(
            f"{item['role']}\tstore/geometry/{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    monkeypatch.setattr(k1_screen_training, "TRAIN_ROWS", 1)
    monkeypatch.setattr(k1_screen_training, "DEVELOPMENT_ROWS", 1)
    monkeypatch.setattr(k1_screen_training, "FIXED_GEOMETRY_SHA256", aggregate.hexdigest())

    hashed, opened = [], []

    def fake_hash(path):
        name = Path(path).name
        hashed.append(name)
        if name == development_item["file"]:
            raise AssertionError("training-only preflight hashed a development shard")
        return train_item["sha256"]

    def fake_load(path):
        name = Path(path).name
        opened.append(name)
        if name == development_item["file"]:
            raise AssertionError("training-only preflight opened a development shard")
        payload = torch.utils.data.TensorDataset(torch.tensor([0]))
        payload._data = SimpleNamespace(source_idx=torch.tensor([0]))
        return payload

    monkeypatch.setattr(k1_screen_training, "sha256_file", fake_hash)
    monkeypatch.setattr(
        k1_screen_training._PackedGraphDatasetFactory, "load", staticmethod(fake_load),
    )

    roles = k1_screen_training.load_roles(tmp_path, manifest, training_only=True)

    assert set(roles) == {"train"}
    assert len(roles["train"]) == 1
    assert hashed == [train_item["file"]]
    assert opened == [train_item["file"]]

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["geometry_shards"][1]["sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="Fixed aggregate recomputation changed"):
        k1_screen_training.load_roles(tmp_path, changed_manifest, training_only=True)
    assert development_item["file"] not in hashed
    assert development_item["file"] not in opened
