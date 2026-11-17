"""Tiny CPU proofs for opt-in native calibration, never scientific qualification."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molgap.family_training_state import CalibrationPolicy, FamilyTrainingState, source_order_sha256
from molgap.gptrans_execution_hotpath import BoundForeachEMA
from molgap.training_reproducibility import capture_rng_state


def factory():
    return torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2),
                               torch.nn.Dropout(.8), torch.nn.Linear(2, 1))


def make_case():
    live = factory()
    optimizer = torch.optim.AdamW(live.parameters(), lr=.01)
    live.train()
    optimizer.zero_grad(set_to_none=True)
    live(torch.arange(8, dtype=torch.float32).reshape(4, 2)).sum().backward()
    optimizer.step()
    raw = BoundForeachEMA(live, .999)
    batches = [SimpleNamespace(x=torch.arange(8, dtype=torch.float32).reshape(4, 2) + i * 10,
                              source_idx=torch.arange(i * 4, (i + 1) * 4), num_graphs=4)
               for i in range(2)]
    policy = CalibrationPolicy("1" * 64, "2" * 64, source_order_sha256(torch.arange(8)), 2, 4)
    return FamilyTrainingState(live, raw, clean_factory=factory), optimizer, batches, policy


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def calibrate(owner, batches, policy, *, forward=None):
    return owner.calibrate(policy, batches, forward=forward or (lambda model, batch: model(batch.x).reshape(-1)),
                           source_indices=lambda batch: batch.source_idx)


def test_native_calibration_changes_only_clean_bn_and_persists_across_return_save_load(tmp_path):
    owner, optimizer, batches, policy = make_case()
    live, raw, optim = deepcopy(owner.live_model.state_dict()), deepcopy(owner.raw_ema.state_dict()), deepcopy(optimizer.state_dict())
    rng = capture_rng_state()
    modes = [module.training for module in owner.live_model.modules()]
    seen = []
    def forward(model, batch):
        seen.append((model[1].training, model[2].training, model[1].momentum))
        return model(batch.x).reshape(-1)
    record = calibrate(owner, batches, policy, forward=forward)
    assert seen == [(True, False, None)] * 2
    assert owner.clean_status == "ready" and record["policy_identity"] == policy.identity
    equal(owner.live_model.state_dict(), live)
    equal(owner.raw_ema.state_dict(), raw)
    equal(optimizer.state_dict(), optim)
    equal(capture_rng_state(), rng)
    assert [module.training for module in owner.live_model.modules()] == modes
    clean = owner.clean_model.state_dict()
    changed = {key for key in raw if not torch.equal(raw[key], clean[key])}
    assert changed == {"1.running_mean", "1.running_var", "1.num_batches_tracked"}
    assert clean["1.num_batches_tracked"].item() == 2
    assert owner.clean_model[1].momentum == .1 and owner.clean_model[1].training
    prediction = owner.evaluate("clean-ema", lambda model: model(batches[0].x), calibration_policy=policy)
    assert prediction.shape == (4, 1) and torch.isfinite(prediction).all()
    assert not prediction.requires_grad
    path = tmp_path / "state.pt"
    saved = owner.state_dict()
    owner.save(path)
    fresh_live = factory()
    restored = FamilyTrainingState(fresh_live, BoundForeachEMA(fresh_live), clean_factory=factory)
    before_load_rng = capture_rng_state()
    restored.load(path, calibration_policy=policy)
    equal(capture_rng_state(), before_load_rng)
    equal(restored.state_dict(), saved)
    assert restored.clean_status == "ready"
    equal(restored.clean_model.state_dict(), clean)


@pytest.mark.parametrize("fault", ["exception", "nonfinite", "short", "row-order", "boundary", "parameter-write"])
def test_failed_calibration_rolls_back_prior_clean_live_raw_and_rng(fault):
    owner, optimizer, batches, policy = make_case()
    calibrate(owner, batches, policy)
    before, optim, rng = owner.state_dict(), deepcopy(optimizer.state_dict()), capture_rng_state()
    call = 0
    def forward(model, batch):
        nonlocal call
        call += 1
        result = model(batch.x).reshape(-1)
        torch.rand(1)
        if fault == "exception" and call == 2:
            raise RuntimeError("synthetic interruption")
        if fault == "nonfinite":
            return result * float("nan")
        if fault == "parameter-write":
            model[0].weight.add_(1.)
        return result
    if fault == "short":
        batches = batches[:1]
    elif fault == "row-order":
        batches = list(reversed(batches))
    elif fault == "boundary":
        batches[0].num_graphs = 3
    with pytest.raises((ValueError, RuntimeError)):
        calibrate(owner, batches, policy, forward=forward)
    equal(owner.state_dict(), before)
    equal(optimizer.state_dict(), optim)
    equal(capture_rng_state(), rng)
    assert owner.clean_model[1].momentum == .1 and owner.clean_model[1].training


@pytest.mark.parametrize("external", [False, True])
def test_raw_update_makes_clean_stale_including_external_update_and_checkpoint_roundtrip(external, tmp_path):
    owner, _, batches, policy = make_case()
    calibrate(owner, batches, policy)
    with torch.no_grad():
        owner.live_model[0].weight.add_(2.)
    if external:
        owner.raw_ema.update(owner.live_model)
    else:
        owner.update_raw_ema()
    assert owner.clean_status == "stale"
    with pytest.raises(ValueError, match="stale"):
        owner.evaluate("clean-ema", lambda model: model(batches[0].x), calibration_policy=policy)
    path = tmp_path / "stale.pt"
    owner.save(path)
    fresh = factory()
    restored = FamilyTrainingState(fresh, BoundForeachEMA(fresh), clean_factory=factory)
    restored.load(path, calibration_policy=policy)
    assert restored.clean_status == "stale"
    calibrate(restored, batches, policy)
    assert restored.clean_status == "ready"


def test_explicit_evaluation_calibrates_first_and_retains_buffers():
    owner, _, batches, policy = make_case()
    with pytest.raises(ValueError, match="missing"):
        owner.evaluate("clean-ema", lambda model: None, calibration_policy=policy)
    result = owner.calibrate_and_evaluate(policy, batches,
        forward=lambda model, batch: model(batch.x).reshape(-1),
        source_indices=lambda batch: batch.source_idx,
        evaluator=lambda model: (model[1].num_batches_tracked.item(), model[2].training))
    assert result == (2, False)
    assert owner.clean_model[1].num_batches_tracked.item() == 2
    other = CalibrationPolicy("3" * 64, "2" * 64, policy.source_order_sha256, 2, 4)
    with pytest.raises(ValueError, match="differently calibrated"):
        owner.evaluate("clean-ema", lambda model: None, calibration_policy=other)
    with pytest.raises(ValueError, match="differently calibrated"):
        owner.evaluate("clean-ema", lambda model: None)


def test_live_and_raw_evaluation_reuse_existing_view_and_restore_live_modes_and_tensors():
    owner, _, batches, _ = make_case()
    with torch.no_grad():
        owner.live_model[0].weight.add_(3.)
    live = deepcopy(owner.live_model.state_dict())
    modes = [module.training for module in owner.live_model.modules()]
    for weights in ("live", "raw-ema"):
        result = owner.evaluate(weights, lambda model: model(batches[0].x))
        assert torch.isfinite(result).all()
        equal(owner.live_model.state_dict(), live)
        assert [module.training for module in owner.live_model.modules()] == modes
    with pytest.raises(RuntimeError, match="fixture"):
        owner.evaluate("raw-ema", lambda model: (_ for _ in ()).throw(RuntimeError("fixture")))
    equal(owner.live_model.state_dict(), live)


@pytest.mark.parametrize("mutation", ["buffer", "policy", "staleness", "missing-raw", "schema"])
def test_checkpoint_corruption_is_rejected_without_mutation(mutation):
    owner, _, batches, policy = make_case()
    calibrate(owner, batches, policy)
    before = owner.state_dict()
    wrong = deepcopy(before)
    if mutation == "buffer":
        wrong["clean"]["model"]["1.running_mean"].add_(1.)
    elif mutation == "policy":
        wrong["clean"]["calibration"]["policy"]["batches"] += 1
    elif mutation == "staleness":
        wrong["clean"]["stale"] = True
    elif mutation == "missing-raw":
        wrong["raw_ema"] = None
    else:
        wrong["live"]["0.weight"] = torch.ones(1)
    rng = capture_rng_state()
    with pytest.raises(ValueError):
        owner.load_state_dict(wrong, calibration_policy=policy)
    equal(owner.state_dict(), before)
    equal(capture_rng_state(), rng)


@pytest.mark.parametrize("field,value", [("role", "development"), ("batches", 0), ("batches", True),
                                          ("batch_size", 1), ("source_order_sha256", "bad")])
def test_calibration_policy_fails_closed(field, value):
    values = dict(input_identity="1" * 64, training_role_identity="2" * 64,
                  source_order_sha256="3" * 64, batches=2, batch_size=4)
    values[field] = value
    with pytest.raises(ValueError):
        CalibrationPolicy(**values)


def test_calibration_rejects_no_bn_or_alias_factory_and_keeps_rng_and_clean_missing():
    for model_factory in (lambda: torch.nn.Linear(2, 1), factory):
        live = model_factory()
        owner = FamilyTrainingState(live, BoundForeachEMA(live), clean_factory=(lambda: live))
        policy = CalibrationPolicy("1" * 64, "2" * 64, source_order_sha256(torch.arange(4)), 1, 4)
        batch = SimpleNamespace(x=torch.zeros(4, 2), source_idx=torch.arange(4), num_graphs=4)
        rng = capture_rng_state()
        with pytest.raises(ValueError, match="independent"):
            calibrate(owner, [batch], policy)
        equal(capture_rng_state(), rng)
        assert owner.clean_status == "missing"
    live = torch.nn.Linear(2, 1)
    owner = FamilyTrainingState(live, BoundForeachEMA(live), clean_factory=lambda: torch.nn.Linear(2, 1))
    with pytest.raises(ValueError, match="tracked BatchNorm"):
        calibrate(owner, [batch], policy)


def test_calibration_consumes_exact_declared_prefix_and_closes_owned_iterator():
    owner, _, batches, policy = make_case()
    consumed = []
    closed = []
    def stream():
        try:
            for batch in batches:
                consumed.append(batch)
                yield batch
            raise AssertionError("Budget consumed extra batches")
        finally:
            closed.append(True)
    calibrate(owner, stream(), policy)
    assert len(consumed) == policy.batches and closed == [True]


def test_iterator_cleanup_failure_does_not_publish_clean_or_change_prior_calibration():
    owner, _, batches, policy = make_case()
    calibrate(owner, batches, policy)
    before, rng = owner.state_dict(), capture_rng_state()
    class BadCleanup:
        def __init__(self):
            self.source = iter(batches)
        def __iter__(self):
            return self
        def __next__(self):
            return next(self.source)
        def close(self):
            raise RuntimeError("fixture close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        calibrate(owner, BadCleanup(), policy)
    equal(owner.state_dict(), before)
    equal(capture_rng_state(), rng)


def test_clean_evaluation_and_checkpoint_reject_post_calibration_buffer_changes():
    owner, _, batches, policy = make_case()
    calibrate(owner, batches, policy)
    with torch.no_grad():
        owner.clean_model[1].running_mean.add_(1.)
    with pytest.raises(ValueError, match="tensors changed"):
        owner.evaluate("clean-ema", lambda model: None, calibration_policy=policy)
    with pytest.raises(ValueError, match="tensors changed"):
        owner.state_dict()


def test_clean_evaluator_cannot_silently_modify_persisted_state():
    owner, _, batches, policy = make_case()
    calibrate(owner, batches, policy)
    def bad_evaluator(model):
        model[1].running_mean.add_(1.)
    with pytest.raises(ValueError, match="Evaluation changed"):
        owner.evaluate("clean-ema", bad_evaluator, calibration_policy=policy)


def test_checkpoint_restore_failure_rolls_back_live_raw_clean_and_rng():
    owner, _, batches, policy = make_case()
    calibrate(owner, batches, policy)
    before = owner.state_dict()
    updated = deepcopy(before)
    updated["live"]["0.weight"].add_(2.)
    original = owner.live_model.load_state_dict
    calls = []
    def fail_once(state, **kwargs):
        calls.append(True)
        result = original(state, **kwargs)
        if len(calls) == 1:
            torch.rand(1)
            raise RuntimeError("fixture restore failure")
        return result
    owner.live_model.load_state_dict = fail_once
    rng = capture_rng_state()
    with pytest.raises(RuntimeError, match="restore failure"):
        owner.load_state_dict(updated, calibration_policy=policy)
    equal(owner.state_dict(), before)
    equal(capture_rng_state(), rng)


def test_calibration_receipt_is_detached_from_owned_identity():
    owner, _, batches, policy = make_case()
    record = calibrate(owner, batches, policy)
    record["policy"]["batches"] = 100
    assert owner.calibration["policy"]["batches"] == policy.batches


def test_clean_factory_rejects_overlapping_views_at_different_storage_offsets_before_loading():
    class SharedBackingModel(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.weight = torch.nn.Parameter(weights)
            self.bn = torch.nn.BatchNorm1d(2)
        def forward(self, batch):
            return self.bn(batch.x * self.weight).sum(dim=1)
    backing = torch.tensor([1., 2., 3.])
    live = SharedBackingModel(backing[:2])
    owner = FamilyTrainingState(live, BoundForeachEMA(live),
                                clean_factory=lambda: SharedBackingModel(backing[1:]))
    assert backing[:2].data_ptr() != backing[1:].data_ptr()
    batch = SimpleNamespace(x=torch.arange(8, dtype=torch.float32).reshape(4, 2),
                            source_idx=torch.arange(4), num_graphs=4)
    policy = CalibrationPolicy("1" * 64, "2" * 64, source_order_sha256(batch.source_idx), 1, 4)
    before, rng = owner.state_dict(), capture_rng_state()
    with pytest.raises(ValueError, match="aliases Live/Raw tensor storage"):
        calibrate(owner, [batch], policy, forward=lambda model, value: model(value))
    equal(owner.state_dict(), before)
    equal(capture_rng_state(), rng)
    assert torch.equal(backing, torch.tensor([1., 2., 3.])) and owner.clean_status == "missing"


@pytest.mark.parametrize("indices", [torch.tensor([0., 1.]), torch.tensor([True, False]),
                                      torch.tensor([-1, 0]), torch.tensor([[0, 1]])])
def test_source_order_requires_exact_nonnegative_int64_vector(indices):
    with pytest.raises(ValueError):
        source_order_sha256(indices)
