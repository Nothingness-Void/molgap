"""Synthetic checks only; no molecular cache, role or pretrained artifact read."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from molgap import k1_pretrained_combo as combo, k1_screen_training as trainer
from molgap.experiment_execution import build_family_recipe, TRAINING_ADAPTERS
from molgap.v4_runtime import state_dict_sha256


def config(mode=combo.MODES[0]):
    return {"initialization_sha256": "1" * 64, "pretrained_source_sha256": "2" * 64,
            "head_reset_sha256": "3" * 64, "consistency_weight": 0.1,
            "teacher_cache": None if mode == combo.MODES[0] else {
                "weight": 1.0, "teacher_identity": "4" * 64, "cache_manifest_sha256": "5" * 64}}


@pytest.mark.parametrize("mode", combo.MODES)
def test_combined_objective_gradients_and_teacher_detach(mode):
    first = torch.tensor([-0.3, 0.8], requires_grad=True)
    second = torch.tensor([0.1, 0.4], requires_grad=True)
    target = torch.tensor([0.0, 0.6])
    teacher = torch.tensor([0.2, 0.2], requires_grad=True) if mode == combo.MODES[1] else None
    loss, parts = combo.objective(first, second, target, mode=mode, teacher=teacher)
    expected = 0.5 * ((first - target).abs().mean() + (second - target).abs().mean())
    expected += 0.1 * (first - second).square().mean()
    if teacher is not None:
        expected += ((first + second) / 2 - teacher.detach()).square().mean()
    torch.testing.assert_close(loss, expected)
    actual = torch.autograd.grad(loss, (first, second), retain_graph=True)
    wanted = torch.autograd.grad(expected, (first, second))
    for a, b in zip(actual, wanted):
        torch.testing.assert_close(a, b)
    loss.backward()
    if teacher is not None:
        assert teacher.grad is None
    assert (parts["teacher_mse"] is None) == (mode == combo.MODES[0])


def test_mean_teacher_mse_does_not_add_per_pass_disagreement_penalty():
    first, second, target, teacher = map(torch.tensor, ([0.0, 1.0], [2.0, 3.0], [1.0, 2.0], [1.0, 2.0]))
    _, parts = combo.objective(first, second, target, mode=combo.MODES[1], teacher=teacher)
    assert parts["teacher_mse"].item() == 0
    per_pass = 0.5 * ((first - teacher).square().mean() + (second - teacher).square().mean())
    torch.testing.assert_close(per_pass, parts["teacher_mse"] + 0.25 * parts["disagreement"])


@pytest.mark.parametrize("mode", combo.MODES)
def test_registered_recipe_binds_new_init_and_preserves_old_pins(mode):
    addon = next(k for k, v in combo.ADDON_MODES.items() if v == mode)
    recipe = build_family_recipe(("neural_atom_k1", "2"), addon=addon,
        source_idx_sha256="a" * 64, target_sha256="b" * 64, addon_config=config(mode))
    trainer.validate_recipe(recipe, mode=mode)
    assert recipe["initialization_sha256"] == "1" * 64
    assert recipe["loss_identity"] == combo.objective_identity(mode, config(mode))
    assert recipe["gradient_loss_row_evaluations"] == 2 * trainer.SAMPLE_EXPOSURE
    adapter = TRAINING_ADAPTERS[("neural_atom_k1", "2")]
    assert adapter.mode({"addons": [{"name": addon, "version": "1"}]}) == mode
    for old in ("reference", "ssma"):
        old_recipe = trainer.build_screen_recipe(old, source_idx_sha256="a" * 64, target_sha256="b" * 64)
        assert old_recipe["initialization_sha256"] == trainer.INITIAL_STATE_SHA256
        old_recipe["initialization_sha256"] = "1" * 64
        with pytest.raises(ValueError, match="initialization"):
            trainer.validate_recipe(old_recipe, mode=old)


def test_initialization_requires_actual_head_and_backbone_digest():
    state = {"head.weight": torch.tensor([[0.3]]), "backbone.weight": torch.tensor([[0.2]])}
    settings = config()
    settings["initialization_sha256"] = state_dict_sha256(state)
    settings["head_reset_sha256"] = state_dict_sha256({"head.weight": state["head.weight"]})
    combo.validate_initial_state(state, settings)
    changed = copy.deepcopy(settings)
    changed["head_reset_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="reset-head"):
        combo.validate_initial_state(state, changed)


@pytest.mark.parametrize("field,value", [("consistency_weight", 0.2), ("pretrained_source_sha256", None)])
def test_config_rejects_drift(field, value):
    settings = config()
    settings[field] = value
    with pytest.raises(ValueError):
        combo.validate_config(settings, combo.MODES[0])


def test_teacher_not_consumed_by_control():
    with pytest.raises(ValueError, match="Arm A"):
        combo.validate_config(config(combo.MODES[1]), combo.MODES[0])


def test_cached_teacher_cpu_join_and_rejects_development_rows(tmp_path, monkeypatch):
    from molgap import k1_teacher_cache as owner
    from molgap.training_reproducibility import sha256_file
    monkeypatch.setattr(owner, "TRAIN_ROWS", 3)
    payload_path = tmp_path / "teacher_predictions.pt"
    torch.save({"source_idx": torch.arange(3), "prediction_eV": torch.tensor([1., 2., 3.])}, payload_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"format": owner.FORMAT, "teacher_identity": "4" * 64,
        "role": "internal_training", "source_idx_range": [0, 3],
        "dataset_manifest_sha256": owner.FIXED_MANIFEST_SHA256,
        "path": payload_path.name, "sha256": sha256_file(payload_path)}), encoding="utf-8")
    cache = owner.K1TeacherCache(tmp_path, manifest_sha256=sha256_file(manifest_path), teacher_identity="4" * 64)
    batch = SimpleNamespace(source_idx=torch.tensor([2, 0]))
    cache.attach(batch)
    torch.testing.assert_close(batch.teacher_eV, torch.tensor([3., 1.]))
    with pytest.raises(ValueError, match="non-training"):
        cache.attach(SimpleNamespace(source_idx=torch.tensor([3])))
    with pytest.raises(ValueError, match="duplicate"):
        cache.attach(SimpleNamespace(source_idx=torch.tensor([0, 0])))


def test_two_pass_preflight_gate_requires_signal_and_keeps_old_limit(tmp_path):
    from molgap.screen_policy import canonical_fingerprint
    provenance = {"mode": combo.MODES[0], "runtime_fingerprint": "1" * 64}
    architecture = {"accepted": True, "repeatability": {"accepted": True},
        "resume_roundtrip": {"accepted": True}, "zero_initialization_delta": 0.0,
        "selected_state_roundtrip_delta": 0.0, "dropout_signal": {"accepted": True},
        "maximum_overhead_fraction": 2.0, "synchronized_step_overhead_fraction": 1.4}
    def write():
        certificate = {"status": "accepted", "calibration_checks_passed": True,
            "runtime_fingerprint": "1" * 64, "provenance_sha256": canonical_fingerprint(provenance),
            "architecture_sha256": canonical_fingerprint(architecture)}
        for name, value in (("runtime_provenance", provenance), ("runtime_certificate", certificate),
                            ("architecture_preflight", architecture),
                            ("runtime_manifest", {"runtime_fingerprint": "1" * 64})):
            (tmp_path / (name + ".json")).write_text(json.dumps(value), encoding="utf-8")
    write()
    trainer.validate_runtime_preflight(tmp_path, provenance)
    architecture["dropout_signal"]["accepted"] = False
    write()
    with pytest.raises(ValueError, match="signal"):
        trainer.validate_runtime_preflight(tmp_path, provenance)
    architecture["dropout_signal"]["accepted"] = True
    provenance["mode"] = "reference"
    write()
    with pytest.raises(ValueError, match="gate"):
        trainer.validate_runtime_preflight(tmp_path, provenance)


@pytest.mark.parametrize("mode", combo.MODES)
def test_optimizer_single_update_and_rng_resume(monkeypatch, mode):
    from molgap.training_reproducibility import capture_rng_state, restore_rng_state
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 1)
            self.dropout = nn.Dropout(0.4)
        def forward(self, x):
            return self.dropout(self.linear(x)).view(-1)
    calls = []
    def forward(model, batch):
        calls.append(1)
        return model(batch.x)
    monkeypatch.setattr(trainer, "_forward", forward)
    torch.manual_seed(42)
    model = Tiny().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    batch = SimpleNamespace(x=torch.tensor([[1., 0., 0.5], [0., 0.2, 1.]]),
                            y=torch.tensor([0.1, 0.2]), teacher_eV=torch.tensor([0.2, 0.3]))
    combo.optimizer_step(model, optimizer, batch, 0., 1., mode=mode)
    assert len(calls) == 2
    assert {int(v["step"]) for v in optimizer.state.values()} == {1}
    snapshot = copy.deepcopy((model.state_dict(), optimizer.state_dict(), capture_rng_state()))
    loss, _, _ = combo.optimizer_step(model, optimizer, batch, 0., 1., mode=mode)
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(snapshot[0]); optimizer.load_state_dict(snapshot[1]); restore_rng_state(snapshot[2])
    resumed, _, _ = combo.optimizer_step(model, optimizer, batch, 0., 1., mode=mode)
    torch.testing.assert_close(loss, resumed, rtol=0, atol=0)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
    assert set(model._combo_components) == {"supervised_l1", "disagreement", "teacher_mse", "combined_loss"}
