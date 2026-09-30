import math
from types import SimpleNamespace

import pytest
import torch

from molgap.gptrans_objective import (
    GPTransObjective, GPTransObjectiveConfig, combine_losses,
    export_gap_state_dict, validate_objective_checkpoint,
)


@pytest.mark.parametrize("params", [
    {"descriptor_weight": -1}, {"fingerprint_weight": math.nan},
    {"descriptor_weight": math.inf}, {"descriptor_weight": True},
    {"auxiliary_hidden_dim": 0}, {"auxiliary_hidden_dim": 1.5},
    {"auxiliary_seed": -1}, {"auxiliary_seed": True}, {"typo": 2},
    {"schema": "wrong"},
])
def test_invalid_config_rejected(params):
    with pytest.raises(ValueError):
        GPTransObjectiveConfig.from_dict(params)


def test_config_roundtrip_and_identity_changes():
    config = GPTransObjectiveConfig(descriptor_weight=.1, fingerprint_weight=.1)
    assert GPTransObjectiveConfig.from_dict(config.to_dict()) == config
    assert not GPTransObjectiveConfig().enabled
    for kwargs in ({"auxiliary_hidden_dim": 64}, {"descriptor_weight": .2}, {"fingerprint_weight": .2}, {"auxiliary_seed": 4}):
        values = config.to_dict() | kwargs
        assert GPTransObjectiveConfig.from_dict(values).identity != config.identity


def test_default_loss_and_gradient_are_exact_l1():
    x = torch.tensor([1., -2.], requires_grad=True)
    target = torch.zeros_like(x)
    loss, metrics = combine_losses(x, target, GPTransObjectiveConfig())
    loss.backward()
    assert loss.item() == 1.5
    torch.testing.assert_close(x.grad, torch.tensor([.5, -.5]), rtol=0, atol=0)
    assert set(metrics) == {"gap_l1_normalized", "total_loss"}


def test_auxiliary_gradients_and_gap_metric_separation():
    x = torch.tensor([1., -2.], requires_grad=True)
    auxiliary = torch.zeros(2, 712, requires_grad=True)
    config = GPTransObjectiveConfig(descriptor_weight=.1, fingerprint_weight=.2)
    loss, metrics = combine_losses(x, torch.zeros(2), config, auxiliary,
                                  torch.ones(2, 200), torch.ones(2, 512))
    assert metrics["gap_l1_normalized"].item() == 1.5
    assert loss.item() == pytest.approx(1.5 + .1 + .2 * math.log(2))
    loss.backward()
    assert torch.all(auxiliary.grad[:, :200] != 0)
    assert torch.all(auxiliary.grad[:, 200:] != 0)
    assert all(not value.requires_grad for value in metrics.values())


def _masked_loss_inputs():
    prediction = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    target = torch.zeros(3)
    auxiliary = torch.zeros(3, 712, requires_grad=True)
    descriptor_target = torch.ones(3, 200)
    descriptor_target[1] = float("nan")  # Storage placeholder excluded by the mask.
    fingerprint_target = torch.zeros(3, 512)
    fingerprint_target[1, 0] = 1.0
    valid = torch.ones(3, 200, dtype=torch.bool)
    valid[1] = False
    return prediction, target, auxiliary, descriptor_target, fingerprint_target, valid


def test_descriptor_mask_excludes_only_descriptors_and_gap_keeps_every_row():
    prediction, target, auxiliary, descriptors, fingerprints, valid = _masked_loss_inputs()
    config = GPTransObjectiveConfig(descriptor_weight=0.5, fingerprint_weight=0.2)

    loss, metrics = combine_losses(
        prediction, target, config, auxiliary, descriptors, fingerprints,
        descriptor_valid_mask=valid,
    )

    assert metrics["gap_l1_normalized"].item() == 2.0
    assert metrics["descriptor_mse"].item() == 1.0
    assert metrics["fingerprint_bce"].item() == pytest.approx(math.log(2.0))
    assert loss.item() == pytest.approx(2.0 + 0.5 + 0.2 * math.log(2.0))
    loss.backward()
    assert torch.all(prediction.grad != 0)  # Gap remains supervised on the masked row.
    assert torch.all(auxiliary.grad[1, :200] == 0)
    assert torch.all(auxiliary.grad[[0, 2], :200] != 0)
    assert torch.all(auxiliary.grad[:, 200:] != 0)  # Fingerprint supervision is unchanged.


def test_all_false_descriptor_mask_returns_connected_zero():
    prediction = torch.tensor([1.0, -1.0], requires_grad=True)
    auxiliary = torch.zeros(2, 712, requires_grad=True)
    descriptor_target = torch.full((2, 200), float("nan"))
    valid = torch.zeros(2, 200, dtype=torch.bool)

    loss, metrics = combine_losses(
        prediction, torch.zeros_like(prediction),
        GPTransObjectiveConfig(descriptor_weight=0.5),
        auxiliary, descriptor_target, descriptor_valid_mask=valid,
    )

    assert metrics["descriptor_mse"].item() == 0.0
    assert loss.item() == 1.0  # The independent Gap term remains present.
    loss.backward()
    assert prediction.grad is not None and torch.all(prediction.grad != 0)
    assert auxiliary.grad is not None
    assert torch.all(auxiliary.grad[:, :200] == 0)


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.tensor([True, False], dtype=torch.bool),
        torch.ones(3, 200, dtype=torch.uint8),
        torch.ones(3, 200, dtype=torch.bool, device="meta"),
    ],
    ids=["wrong-shape", "non-bool", "wrong-device"],
)
def test_descriptor_mask_shape_dtype_and_device_are_validated(bad_mask):
    with pytest.raises(ValueError):
        combine_losses(
            torch.ones(3), torch.zeros(3),
            GPTransObjectiveConfig(descriptor_weight=0.1),
            torch.zeros(3, 712), torch.ones(3, 200),
            descriptor_valid_mask=bad_mask,
        )


def test_unmasked_descriptor_nan_is_rejected():
    descriptor_target = torch.ones(3, 200)
    descriptor_target[0, 0] = float("nan")
    valid = torch.ones(3, 200, dtype=torch.bool)
    valid[1:] = False

    with pytest.raises(ValueError):
        combine_losses(
            torch.ones(3), torch.zeros(3),
            GPTransObjectiveConfig(descriptor_weight=0.1),
            torch.zeros(3, 712), descriptor_target,
            descriptor_valid_mask=valid,
        )


def test_none_mask_preserves_legacy_descriptor_objective():
    config = GPTransObjectiveConfig(descriptor_weight=0.1)
    values = (
        torch.tensor([1.0, -2.0]), torch.zeros(2), config,
        torch.zeros(2, 712), torch.ones(2, 200),
    )
    legacy_loss, legacy_metrics = combine_losses(*values)
    explicit_loss, explicit_metrics = combine_losses(
        *values, descriptor_valid_mask=None
    )

    torch.testing.assert_close(legacy_loss, explicit_loss, rtol=0, atol=0)
    assert legacy_metrics.keys() == explicit_metrics.keys()
    for name in legacy_metrics:
        torch.testing.assert_close(legacy_metrics[name], explicit_metrics[name], rtol=0, atol=0)


def test_gptrans_objective_reads_batch_descriptor_mask(monkeypatch):
    from molgap import pcqm_gptrans_v4 as core

    count = 3
    prediction = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    auxiliary = torch.zeros(count, 712, requires_grad=True)
    descriptor_target = torch.ones(count, 200)
    descriptor_target[1] = float("nan")
    valid = torch.ones(count, 200, dtype=torch.bool)
    valid[1] = False

    class Readout:
        def register_forward_pre_hook(self, callback):
            self.callback = callback
            return SimpleNamespace(remove=lambda: None)

    class Model:
        def __init__(self):
            self.readout = Readout()

        def _chemical_aux_head(self, features):
            assert features.shape == (count, 4)
            return auxiliary

    model = Model()
    objective = object.__new__(GPTransObjective)
    objective.model = model
    objective.config = GPTransObjectiveConfig(descriptor_weight=0.5)

    def synthetic_forward(observed_model, _batch):
        observed_model.readout.callback(observed_model.readout, (torch.zeros(count, 4),))
        return prediction

    monkeypatch.setattr(core, "_forward", synthetic_forward)
    batch = SimpleNamespace(
        y=torch.zeros(count),
        chemical_descriptors=descriptor_target,
        chemical_fingerprint=None,
        chemical_descriptor_valid_mask=valid,
    )

    loss, metrics = objective.loss(model, batch, mean=0.0, std=1.0)

    assert metrics["gap_l1_normalized"].item() == 2.0
    loss.backward()
    assert torch.all(prediction.grad != 0)
    assert torch.all(auxiliary.grad[1, :200] == 0)
    assert torch.all(auxiliary.grad[[0, 2], :200] != 0)


@pytest.mark.parametrize("bad_target", [torch.ones(2, 511), torch.full((2, 512), float('nan')), torch.full((2, 512), .5)])
def test_bad_auxiliary_labels_rejected(bad_target):
    with pytest.raises(ValueError):
        combine_losses(torch.ones(2), torch.ones(2), GPTransObjectiveConfig(fingerprint_weight=.1),
                       torch.zeros(2, 712), fingerprint_target=bad_target)


def test_resume_identity_fail_closed_and_legacy_baseline():
    config = GPTransObjectiveConfig(descriptor_weight=.1)
    validate_objective_checkpoint({}, GPTransObjectiveConfig())
    checkpoint = {"training_objective": config.to_dict(), "training_objective_sha256": config.identity}
    validate_objective_checkpoint(checkpoint, config)
    for data, expected in [({}, config), (checkpoint, GPTransObjectiveConfig()),
                           ({**checkpoint, "training_objective_sha256": "wrong"}, config),
                           ({"model": {"_chemical_aux_head.0.weight": None}}, GPTransObjectiveConfig())]:
        with pytest.raises(RuntimeError):
            validate_objective_checkpoint(data, expected)


def test_export_retains_only_backbone_keys():
    state = {"readout.0.weight": object(), "_chemical_aux_head.0.weight": object()}
    exported = export_gap_state_dict(state)
    assert list(exported) == ["readout.0.weight"]
    assert exported["readout.0.weight"] is state["readout.0.weight"]


def test_optimizer_step_uses_configured_loss_and_keeps_gap_metric(monkeypatch):
    from molgap import pcqm_gptrans_v4 as core
    # Scalar tensor fixture verifies the existing optimizer wheel without a model forward.
    parameter = torch.tensor(2., requires_grad=True)
    class Model:
        def parameters(self): return [parameter]
    model = Model()
    class Objective:
        def loss(self, m, b, mean, std):
            assert m is model
            return parameter.square(), {"gap_l1_normalized": parameter.detach().clone(), "total_loss": parameter.detach().square()}
    class EMA:
        def update(self, m): assert m is model
    optimizer = torch.optim.SGD([parameter], lr=.01)
    monkeypatch.setattr(core, "_forward", lambda *args: pytest.fail("unexpected frozen forward"))
    metrics = {}
    gap = core._optimizer_step(model, optimizer, EMA(), None, 0, 1, check_finite=True,
                               objective=Objective(), loss_metrics=metrics)
    assert gap.item() == 2 and metrics["total_loss"].item() == 4
    assert parameter.item() < 2


def test_checkpoint_writer_binds_config_and_distinguishes_loss(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from molgap import pcqm_gptrans_v4 as core
    from molgap.gptrans_objective import GPTransObjective
    component = SimpleNamespace(state_dict=lambda: {})
    model = SimpleNamespace(state_dict=lambda: {"_chemical_aux_head.weight": torch.zeros(1)})
    # No model construction: exercise serialization with a metadata-only adapter.
    objective = object.__new__(GPTransObjective)
    objective.model = model
    objective.config = GPTransObjectiveConfig(descriptor_weight=.2)
    captured = {}
    monkeypatch.setattr(core, "atomic_torch_save", lambda path, value: captured.update(value))
    monkeypatch.setattr(core, "capture_rng_state", lambda: {})
    kwargs = dict(epoch=0, model=model, optimizer=component, scheduler=component,
                  ema=component, trace=[], best=1., best_epoch=0, target_stats={},
                  runtime_certificate_id="fixture", source_archive_sha256="fixture")
    core._save_checkpoint(tmp_path / "state.pt", **kwargs, objective=objective)
    validate_objective_checkpoint(captured, objective.config)
    assert captured["scientific_fields"]["loss_fingerprint"] != core._scientific_fields()["loss_fingerprint"]
    objective.model = object()
    with pytest.raises(RuntimeError):
        core._save_checkpoint(tmp_path / "state.pt", **kwargs, objective=objective)
