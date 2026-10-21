"""Synthetic fault injection: no molecular assets, GPU or scientific execution."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from molgap.family_training_outputs import SessionOutputs, inspect_session_output
from molgap.family_training_runner import PROFILES, _begin_cost_interval, _finish_cost_interval
from molgap.family_training_session import validate_session_resume
from molgap.family_training_session import FamilyTrainingSession
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_kernel import make_family_optimizer
from molgap.gptrans_execution_hotpath import BoundForeachEMA
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.training_reproducibility import sha256_file
from test_family_training_runner import contract_file, context
from test_family_training_session import SyntheticKernel, assert_equal, events, make_session


def prepared(tmp_path, family="neural_atom_k1", development=True):
    session = make_session(family, development=development, schedule_kind="cosine_step", events=events())
    path, record = contract_file(tmp_path, session)
    ctx = context(family, path)
    def outputs():
        return SessionOutputs(tmp_path / "output", ctx, contract=path, adapter=PROFILES[family],
            trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"])
    out = outputs()
    bind(session, out)
    return session, out, outputs, record, ctx


def bind(session, out):
    session.checkpoint, session.selected, session.pass_finished = out.checkpoint, out.selected, out.pass_finished


def finish(out, progress, ctx):
    return out.complete(progress, runtime={"platform": ctx.platform, "account": ctx.account, "precision": "fp32",
        "source_commit": ctx.source_commit, "source_archive_sha256": ctx.source_archive_sha256},
        costs=[{"metric": "wall_seconds", "unit": "seconds", "status": "measured", "value": 1.,
            "semantics": "process_wall", "hardware": "synthetic-cpu"}])


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
def test_deleted_known_touched_moment_is_rejected_before_any_restore(family):
    log = events()
    first = make_session(family, development=False, schedule_kind="cosine_step", events=log)
    first.run(should_pause=lambda: True)
    state = deepcopy(log["checkpoint"][-1])
    assert 0 in state["optimizer"]["state"]
    del state["optimizer"]["state"][0]
    fresh = make_session(family, development=False, schedule_kind="cosine_step", events=events())
    before = deepcopy(fresh.state_dict())
    try:
        with pytest.raises(ValueError, match="moments"):
            validate_session_resume(fresh.recipe, state)
        with pytest.raises(ValueError, match="moments"):
            fresh.load_state_dict(state)
        # Even a rewritten participation list cannot delete required base moments.
        state["cursor"]["optimizer_state_ids"].remove(0)
        with pytest.raises(ValueError, match="moments"):
            fresh.load_state_dict(state)
        assert_equal(before, fresh.state_dict())
    finally:
        fresh.close()


def conditional_session(log, *, touched=False, post_addon=False):
    template = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    recipe, data = replace(template.recipe, mode="ssma"), template.loaders.data
    template.close()
    kernel = SyntheticKernel(recipe)
    base = kernel.model[:3]
    kernel.model.add_module("k1_joint_aggregation", torch.nn.Linear(1, 1))
    if post_addon:
        kernel.model.add_module("post_addon", torch.nn.Linear(1, 1))
    kernel.optimizer = make_family_optimizer(recipe, kernel.model.parameters())
    kernel.ema = BoundForeachEMA(kernel.model, recipe.ema_decay)
    def forward(batch):
        value = base(batch.source_idx.float().reshape(-1, 1) / 257.)
        if touched:
            value = value + kernel.model.k1_joint_aggregation(value) * 0.
        if post_addon:
            value = value + kernel.model.post_addon(value) * 0.
        return value.reshape(-1)
    kernel.forward = forward
    return FamilyTrainingSession(kernel, FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0)),
        checkpoint=lambda progress, state: log.append(deepcopy(state)),
        selected=lambda *args: None, pass_finished=lambda *args: None)


def test_conditional_addon_keeps_legitimate_lazy_state_but_not_deleted_acknowledged_moments():
    log = []
    first = conditional_session(log)
    first.run(should_pause=lambda: True)
    state = log[-1]
    assert len(state["optimizer"]["state"]) == 4
    assert len(state["optimizer"]["param_groups"][0]["params"]) == 6
    fresh = conditional_session([])
    try:
        fresh.load_state_dict(state)
    finally:
        fresh.close()
    legacy = deepcopy(state)
    legacy["cursor"].pop("optimizer_parameter_names")
    legacy_fresh = conditional_session([])
    try:
        legacy_fresh.load_state_dict(legacy)
    finally:
        legacy_fresh.close()
    log = []
    touched = conditional_session(log, touched=True)
    touched.run(should_pause=lambda: True)
    state = log[-1]
    assert len(state["optimizer"]["state"]) == 6
    del state["optimizer"]["state"][4]
    fresh = conditional_session([], touched=True)
    try:
        with pytest.raises(ValueError, match="moments"):
            fresh.load_state_dict(state)
    finally:
        fresh.close()


def test_conditional_addon_can_precede_required_parameters_in_optimizer_order():
    log = []
    first = conditional_session(log, post_addon=True)
    first.run(should_pause=lambda: True)
    state = log[-1]
    assert state["cursor"]["optimizer_parameter_names"][4].startswith("k1_joint_aggregation.")
    assert state["cursor"]["optimizer_parameter_names"][6].startswith("post_addon.")
    assert set(state["optimizer"]["state"]) == {0, 1, 2, 3, 6, 7}
    fresh = conditional_session([], post_addon=True)
    try:
        fresh.load_state_dict(state)
    finally:
        fresh.close()
    legacy = deepcopy(state)
    legacy["cursor"].pop("optimizer_parameter_names")
    legacy_fresh = conditional_session([], post_addon=True)
    try:
        with pytest.raises(ValueError, match="moments"):
            legacy_fresh.load_state_dict(legacy)
    finally:
        legacy_fresh.close()
    del state["optimizer"]["state"][6]
    state["cursor"]["optimizer_state_ids"].remove(6)
    fresh = conditional_session([], post_addon=True)
    try:
        with pytest.raises(ValueError, match="moments"):
            fresh.load_state_dict(state)
    finally:
        fresh.close()


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("corruption", ["key", "shape", "dtype"])
def test_selected_development_schema_cannot_pass_with_rebound_file_hash(tmp_path, family, corruption):
    session, out, _, record, ctx = prepared(tmp_path, family)
    progress = session.run()
    assert finish(out, progress, ctx)["status"] == "MECHANICALLY_VERIFIED"
    path = out.root / "selected_model.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(state["model"]))
    if corruption == "key":
        del state["model"][key]
    elif corruption == "shape":
        state["model"][key] = torch.ones(13)
    else:
        state["model"][key] = state["model"][key].double()
    torch.save(state, path)
    manifest_path = out.root / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["selected_model"]["sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    report = inspect_session_output(out.root, context=ctx, expected=record["acceptance_requirements"])
    assert report["status"] == "BLOCKED"
    assert any("schema" in message for message in report["blockers"])


@pytest.mark.parametrize("failure,pause_at", [
    *[(failure, 1) for failure in ("callback", "journal", "selected", "trace", "checkpoint_before", "checkpoint_after")],
    *[(failure, 3) for failure in ("trace", "checkpoint_before", "checkpoint_after")],
])
def test_interrupted_boundary_recovers_whole_bundle_and_exact_continuation(tmp_path, monkeypatch, failure, pause_at):
    import molgap.family_training_outputs as owner
    session, out, reopen, record, ctx = prepared(tmp_path)
    pauses = iter([False] * (pause_at - 1) + [True])
    session.run(should_pause=lambda: next(pauses))
    original_sha = sha256_file(out.root / "last_checkpoint.pt")
    retained = {name: sha256_file(out.root / name) for name in
        ("selected_model.pt", "development_predictions.pt", "canonical_trace.json")
        if (out.root / name).is_file()}
    state = out.retained_state()
    resumed = make_session("neural_atom_k1", development=True, schedule_kind="cosine_step", events=events())
    bind(resumed, out)
    resumed.load_state_dict(state)
    native_replace, native_write = owner.os.replace, owner.atomic_write
    hit = []
    def replace(source, destination):
        source, destination = Path(source), Path(destination)
        target = {"selected": "selected_model.pt", "trace": "canonical_trace.json",
            "checkpoint_before": "last_checkpoint.pt", "checkpoint_after": "last_checkpoint.pt"}.get(failure)
        if not hit and source.parent == out.pending and source.name == target and destination.parent == out.root:
            hit.append(True)
            if failure == "checkpoint_after":
                native_replace(source, destination)
            raise OSError("injected publication failure")
        return native_replace(source, destination)
    def write(path, payload):
        if failure == "journal" and Path(path).name == "session_commit.json":
            hit.append(True)
            raise OSError("injected publication failure")
        return native_write(path, payload)
    def callback(progress, payload):
        if failure == "callback":
            hit.append(True)
            raise OSError("injected publication failure")
        return out.checkpoint(progress, payload)
    resumed.checkpoint = callback
    with monkeypatch.context() as scoped:
        scoped.setattr(owner.os, "replace", replace)
        scoped.setattr(owner, "atomic_write", write)
        with pytest.raises(OSError, match="injected"):
            resumed.run()
    assert hit
    recovered = reopen()
    kept = recovered.retained_state()
    committed = failure == "checkpoint_after"
    assert kept["optimizer_step"] == (pause_at + 1 if committed else pause_at)
    if not committed:
        assert sha256_file(out.root / "last_checkpoint.pt") == original_sha
        assert all(sha256_file(out.root / name) == digest for name, digest in retained.items())
        if pause_at == 1:
            assert not (out.root / "selected_model.pt").exists()
    assert not (out.root / "session_commit.json").exists()
    continued = make_session("neural_atom_k1", development=True, schedule_kind="cosine_step", events=events())
    bind(continued, recovered)
    continued.load_state_dict(kept)
    progress = continued.run()
    assert finish(recovered, progress, ctx)["status"] == "MECHANICALLY_VERIFIED"
    control = make_session("neural_atom_k1", development=True, schedule_kind="cosine_step", events=events())
    control.run()
    assert_equal(control.state_dict(), continued.state_dict())


def test_interrupted_invocation_never_reports_stale_cumulative_cost_as_measured(tmp_path):
    session, out, _, _, ctx = prepared(tmp_path)
    assert _begin_cost_interval(out.root, ctx, had_checkpoint=False) == 0.
    session.run(should_pause=lambda: True)
    assert _finish_cost_interval(out.root, ctx, 0., 10.) == 10.
    assert _begin_cost_interval(out.root, ctx, had_checkpoint=True) == 10.
    # The next start observes the unfinished invocation, even without a new save.
    assert _begin_cost_interval(out.root, ctx, had_checkpoint=True) is None
    assert _finish_cost_interval(out.root, ctx, None, 3.) is None
    assert json.loads((out.root / "allocation_costs.json").read_text())["wall_seconds"] is None


def test_cost_ledger_must_bind_to_acknowledged_checkpoint_and_context(tmp_path):
    session, out, _, _, ctx = prepared(tmp_path)
    _begin_cost_interval(out.root, ctx, had_checkpoint=False)
    session.run(should_pause=lambda: True)
    _finish_cost_interval(out.root, ctx, 0., 10.)
    path = out.root / "allocation_costs.json"
    record = json.loads(path.read_text())
    record["checkpoint_sha256"] = "f" * 64
    path.write_text(json.dumps(record))
    assert _begin_cost_interval(out.root, ctx, had_checkpoint=True) is None
    record["context"]["arm_id"] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="context"):
        _begin_cost_interval(out.root, ctx, had_checkpoint=True)


@pytest.mark.parametrize("prior,elapsed", [(True, 1.), (-1., 1.), (float("nan"), 1.),
    (1., float("inf")), (1e308, 1e308)])
def test_invalid_cost_values_cannot_be_published(tmp_path, prior, elapsed):
    session, out, _, _, ctx = prepared(tmp_path)
    try:
        with pytest.raises(ValueError, match="cost"):
            _finish_cost_interval(out.root, ctx, prior, elapsed)
        assert not (out.root / "allocation_costs.json").exists()
    finally:
        session.close()
