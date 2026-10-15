"""The retained loop observer must preserve optimizer/model/RNG state."""
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch

from molgap.family_execution_profile import ProfileConfig
from molgap.k1_legacy_execution_profile import LegacyK1ExecutionProfiler, state_fingerprint
from test_family_training_session import assert_equal


def fixture():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Dropout(.2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    owner = SimpleNamespace()
    owner._forward = lambda model, batch: model(batch)
    def step(model, optimizer, batch, *args):
        optimizer.zero_grad(set_to_none=True)
        loss = owner._forward(model, batch).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        return loss.detach()
    owner._optimizer_step = step
    owner._evaluate = lambda *args: None
    owner._train_loader = lambda *args: [torch.ones(128, 1)] * 3
    return owner, model, optimizer


def test_legacy_observer_reuses_capture_and_preserves_complete_state(tmp_path):
    owner, model, optimizer = fixture()
    for _ in range(2):
        for batch in owner._train_loader():
            owner._optimizer_step(model, optimizer, batch)
    expected = state_fingerprint(model, optimizer)
    owner, model, optimizer = fixture()
    originals = dict(vars(owner))
    with LegacyK1ExecutionProfiler(owner, {}, tmp_path / "profile", ProfileConfig(0, 2), model=model) as prof:
        for _ in range(2):
            for batch in owner._train_loader():
                owner._optimizer_step(model, optimizer, batch)
    assert state_fingerprint(model, optimizer) == expected
    assert dict(vars(owner)) == originals
    assert prof.report["captured_updates"] == 2
    assert prof.report["observed_updates"] == 6
    assert prof.report["capture_off_training_spans"][0]["optimizer_updates"] == 3
    assert prof.report["capture_off_training_spans"][0]["sample_presentations"] == 384
    phases = {row["name"]: row["calls"] for row in prof.report["phases_by_host_self_time"]}
    assert phases["forward"] == phases["backward"] == phases["optimizer"] == 2


def test_legacy_observer_restores_failed_owner(tmp_path):
    owner, model, optimizer = fixture()
    def failure(*args):
        raise RuntimeError("legacy update fault")
    owner._optimizer_step = failure
    with pytest.raises(RuntimeError, match="legacy update fault"):
        with LegacyK1ExecutionProfiler(owner, {}, tmp_path / "profile", ProfileConfig(0, 2), model=model) as prof:
            owner._optimizer_step(model, optimizer, torch.ones(128, 1))
    assert owner._optimizer_step is failure
    assert prof.report["status"] == "ERROR"
