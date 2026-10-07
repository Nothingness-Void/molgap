import pytest
import torch

from molgap import k1_bn_calibration as inference


class _Batch(dict):
    def __init__(self, source_idx, x, **fields):
        super().__init__(x=x, **fields)
        self.source_idx = torch.as_tensor(source_idx, dtype=torch.long)

    def to(self, device):
        for name, value in tuple(self.items()):
            if torch.is_tensor(value):
                self[name] = value.to(device)
        self.source_idx = self.source_idx.to(device)
        return self


class _MinimalFakeGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2, momentum=0.37)
        self.dropout = torch.nn.Dropout(p=0.9)
        self.head = torch.nn.Linear(2, 1)
        self.forward_modes = []
        with torch.no_grad():
            self.bn.running_mean.copy_(torch.tensor([9.0, -4.0]))
            self.bn.running_var.copy_(torch.tensor([7.0, 11.0]))
            self.bn.num_batches_tracked.fill_(9)
            self.head.weight.copy_(torch.tensor([[0.5, -0.25]]))
            self.head.bias.fill_(0.1)

    def forward(self, batch):
        self.forward_modes.append((self.bn.training, self.dropout.training))
        value = self.dropout(self.bn(batch["x"]))
        return self.head(value).view(-1)


def _frozen_model():
    model = _MinimalFakeGraphModel().eval().requires_grad_(False)
    return model


def _snapshot(values):
    return {name: value.detach().clone() for name, value in values.items()}


def _assert_tensor_mapping_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def _parameter_snapshot(model):
    return _snapshot(dict(model.named_parameters()))


def _calibration_batches():
    return [
        _Batch([0, 1], torch.tensor([[-2.0, 0.0], [2.0, 2.0]])),
        _Batch([2, 3], torch.tensor([[8.0, 4.0], [12.0, 8.0]])),
    ]


def test_recalibration_updates_only_bn_buffers_and_restores_state(monkeypatch):
    model = _frozen_model()
    monkeypatch.setattr(inference, "_forward", lambda model, batch: model(batch))
    original_buffers = _snapshot(dict(model.named_buffers()))
    original_parameters = _parameter_snapshot(model)
    assert all(parameter.grad is None for parameter in model.parameters())

    with inference.recalibrated_batch_norm(
        model,
        _calibration_batches(),
        source_idx=[0, 1, 2, 3],
        source_bounds=(0, 4),
        device="cpu",
        deadline=float("inf"),
    ) as report:
        assert model.training is False
        assert model.bn.training is False
        assert model.dropout.training is False
        assert model.bn.momentum is None
        assert torch.equal(model.bn.running_mean, torch.tensor([5.0, 3.5]))
        assert torch.equal(model.bn.running_var, torch.tensor([8.0, 5.0]))
        assert int(model.bn.num_batches_tracked) == 2
        assert all(dropout_training is False for _, dropout_training in model.forward_modes)
        assert all(batch_norm_training is True for batch_norm_training, _ in model.forward_modes)
        _assert_tensor_mapping_equal(dict(model.named_parameters()), original_parameters)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert report["rows"] == 4
        assert report["batches"] == 2
        assert report["parameters_unchanged"] is True
        assert report["non_bn_buffers_unchanged"] is True
        assert report["dropout_disabled"] is True

    _assert_tensor_mapping_equal(dict(model.named_buffers()), original_buffers)
    _assert_tensor_mapping_equal(dict(model.named_parameters()), original_parameters)
    assert model.training is False
    assert all(not module.training for module in model.modules())
    assert model.bn.momentum == pytest.approx(0.37)
    assert report["buffers_restored"] is True


def test_recalibration_restores_buffers_and_modes_after_forward_exception(monkeypatch):
    model = _frozen_model()
    original_buffers = _snapshot(dict(model.named_buffers()))
    original_parameters = _parameter_snapshot(model)

    def interrupted_forward(model, batch):
        model(batch)
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(inference, "_forward", interrupted_forward)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with inference.recalibrated_batch_norm(
            model,
            _calibration_batches(),
            source_idx=[0, 1, 2, 3],
            source_bounds=(0, 4),
            device="cpu",
            deadline=float("inf"),
        ):
            pytest.fail("an interrupted forward must not yield")

    _assert_tensor_mapping_equal(dict(model.named_buffers()), original_buffers)
    _assert_tensor_mapping_equal(dict(model.named_parameters()), original_parameters)
    assert model.training is False
    assert all(not module.training for module in model.modules())
    assert model.bn.momentum == pytest.approx(0.37)
    assert model.forward_modes == [(True, False)]


@pytest.mark.parametrize(
    ("expected", "observed", "bounds", "message"),
    [
        ([4], [4], (0, 4), "membership"),
        ([1], [0], (0, 4), "order or membership"),
    ],
    ids=("development-row", "unbound-row"),
)
def test_development_or_unbound_rows_are_rejected_before_forward(
    monkeypatch, expected, observed, bounds, message
):
    model = _frozen_model()
    calls = []
    monkeypatch.setattr(
        inference, "_forward", lambda model, batch: calls.append(batch)
    )

    with pytest.raises(ValueError, match=message):
        with inference.recalibrated_batch_norm(
            model,
            [_Batch(observed, torch.ones(len(observed), 2))],
            source_idx=expected,
            source_bounds=bounds,
            device="cpu",
            deadline=float("inf"),
        ):
            pytest.fail("invalid calibration membership must not yield")

    assert calls == []
    assert model.forward_modes == []


def test_geometry_field_is_rejected_before_forward(monkeypatch):
    model = _frozen_model()
    calls = []
    monkeypatch.setattr(
        inference, "_forward", lambda model, batch: calls.append(batch)
    )

    with pytest.raises(ValueError, match="Geometry reached"):
        with inference.recalibrated_batch_norm(
            model,
            [_Batch([0], torch.ones(1, 2), pos=torch.zeros(1, 3))],
            source_idx=[0],
            source_bounds=(0, 4),
            device="cpu",
            deadline=float("inf"),
        ):
            pytest.fail("geometry must be rejected before yielding")

    assert calls == []
    assert model.forward_modes == []


def test_expired_deadline_rejects_before_forward_and_restores_buffers(monkeypatch):
    model = _frozen_model()
    original_buffers = _snapshot(dict(model.named_buffers()))
    calls = []
    monkeypatch.setattr(
        inference, "_forward", lambda model, batch: calls.append(batch)
    )

    with pytest.raises(TimeoutError, match="wall budget exhausted"):
        with inference.recalibrated_batch_norm(
            model,
            _calibration_batches(),
            source_idx=[0, 1, 2, 3],
            source_bounds=(0, 4),
            device="cpu",
            deadline=float("-inf"),
        ):
            pytest.fail("expired calibration must not yield")

    assert calls == []
    _assert_tensor_mapping_equal(dict(model.named_buffers()), original_buffers)
    assert all(not module.training for module in model.modules())
    assert model.bn.momentum == pytest.approx(0.37)


def test_incomplete_membership_raises_and_restores_buffers(monkeypatch):
    model = _frozen_model()
    original_buffers = _snapshot(dict(model.named_buffers()))
    monkeypatch.setattr(inference, "_forward", lambda model, batch: model(batch))

    with pytest.raises(ValueError, match="Incomplete calibration membership"):
        with inference.recalibrated_batch_norm(
            model,
            [_Batch([0, 1], torch.tensor([[-2.0, 0.0], [2.0, 2.0]]))],
            source_idx=[0, 1, 2, 3],
            source_bounds=(0, 4),
            device="cpu",
            deadline=float("inf"),
        ):
            pytest.fail("incomplete calibration must not yield")

    assert model.forward_modes == [(True, False)]
    _assert_tensor_mapping_equal(dict(model.named_buffers()), original_buffers)
    assert all(not module.training for module in model.modules())
    assert model.bn.momentum == pytest.approx(0.37)
