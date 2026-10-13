"""Server-only shared-session hooks under synthetic CPU fault injection."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from molgap.experiment_family_workflow import tensor_digest
from molgap.family_training_data import DatasetSelection
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_runner import INTERFACES, PROFILES, build_screen_recipe
from molgap.family_training_session import FamilyTrainingSession
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.training_reproducibility import sha256_file
from test_family_recovery_guards import bind, finish
from test_family_training_runner import context
from test_family_training_session import SyntheticKernel, assert_equal, events, make_session


MODES = ("degree_bond_local_ema999", "degree_local_cycle_common_ema999")


def bounded_session(mode, *, passes=2):
    template = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    data, recipe = template.loaders.data, template.recipe
    template.close()
    recipe = replace(recipe, mode=mode, optimizer_steps=passes * recipe.loop_plan.steps_per_pass,
        schedule=replace(recipe.schedule, total_units=passes * recipe.loop_plan.steps_per_pass),
        input_sidecar=DatasetSelection("synthetic-cycles.json", "5" * 64, True) if mode == MODES[1] else None)
    return FamilyTrainingSession(SyntheticKernel(recipe), FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0)),
        checkpoint=lambda *args: None, selected=lambda *args: None, pass_finished=lambda *args: None)


def prepared(tmp_path, mode, *, passes=2):
    session = bounded_session(mode, passes=passes)
    data = session.loaders.data
    record = build_screen_recipe(mode, configuration=session.recipe, family=INTERFACES["gptrans_t"],
        train_role_identity="3" * 64, development_role_identity="4" * 64,
        source_idx_sha256=tensor_digest(torch.cat([g.source_idx for g in data.development]), role="source_idx"),
        target_sha256=tensor_digest(torch.cat([g.y for g in data.development]), role="target"))
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    ctx = context("gptrans_t", path)
    def reopen():
        return SessionOutputs(tmp_path / "output", ctx, contract=path, adapter=PROFILES["gptrans_t"],
            trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"])
    out = reopen()
    bind(session, out)
    return session, out, reopen, ctx


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("failure", ["selected", "trace", "checkpoint_before", "checkpoint_after"])
def test_bounded_pair_recovers_selected_trace_and_checkpoint_together(tmp_path, monkeypatch, mode, failure):
    import molgap.family_training_outputs as owner
    first, out, reopen, ctx = prepared(tmp_path, mode)
    pauses = iter([False, False, True])
    first.run(should_pause=lambda: next(pauses))
    kept_hashes = {name: sha256_file(out.root / name) for name in
        ("selected_model.pt", "development_predictions.pt", "canonical_trace.json", "last_checkpoint.pt")}
    resumed = bounded_session(mode)
    bind(resumed, out)
    resumed.load_state_dict(out.retained_state())
    native_replace, hit = owner.os.replace, []
    target = {"selected": "selected_model.pt", "trace": "canonical_trace.json",
              "checkpoint_before": "last_checkpoint.pt", "checkpoint_after": "last_checkpoint.pt"}[failure]
    def interrupt(source, destination):
        source, destination = Path(source), Path(destination)
        if not hit and source.parent == out.pending and source.name == target and destination.parent == out.root:
            hit.append(True)
            if failure == "checkpoint_after":
                native_replace(source, destination)
            raise OSError("injected bounded-pair publication failure")
        return native_replace(source, destination)
    with monkeypatch.context() as scoped:
        scoped.setattr(owner.os, "replace", interrupt)
        with pytest.raises(OSError, match="injected bounded-pair"):
            resumed.run()
    assert hit
    recovered = reopen()
    state = recovered.retained_state()
    assert state["optimizer_step"] == (4 if failure == "checkpoint_after" else 3)
    if failure != "checkpoint_after":
        assert all(sha256_file(out.root / name) == digest for name, digest in kept_hashes.items())
    continued = bounded_session(mode)
    bind(continued, recovered)
    continued.load_state_dict(state)
    progress = continued.run()
    report = finish(recovered, progress, ctx)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    trace = json.loads((out.root / "canonical_trace.json").read_text())
    rows = trace["observations"]
    assert [row["optimizer_step"] for row in rows] == [2, 4]
    assert all(row["live_dev_metric"] is not None and row["ema_dev_metric"] is not None for row in rows)
    assert all(row["wall_time_seconds"] > 0 and row["device_time_seconds"] > 0 for row in rows)
    assert trace["metric_semantics"]["live_train_metric"]["unit"] == "eV"
    assert trace["device_time_semantics"] == "sum_over_devices"
    assert rows[-1]["checkpoint_identity"] == sha256_file(out.root / "last_checkpoint.pt")
    assert not (out.root / "session_commit.json").exists()
    control = bounded_session(mode)
    control.run()
    assert_equal(control.state_dict(), continued.state_dict())


@pytest.mark.parametrize("mode", MODES)
def test_ten_pass_transport_hook_only_sees_committed_bundle(tmp_path, monkeypatch, mode):
    import molgap.experiment_resume as owner
    session, out, _, _ = prepared(tmp_path, mode, passes=11)
    out.resume_trajectory = b"synthetic-prospective"
    chunks = []
    def retain(root, ctx, trajectory, destination):
        assert not (root / "session_commit.json").exists()
        trace = json.loads((root / "canonical_trace.json").read_text())
        assert trace["observations"][-1]["optimizer_step"] == 20
        assert trace["observations"][-1]["checkpoint_identity"] == sha256_file(root / "last_checkpoint.pt")
        assert trajectory == b"synthetic-prospective"
        chunks.append(destination)
    monkeypatch.setattr(owner, "create_resume_bundle", retain)
    session.run()
    assert chunks == [out.root.parent / "checkpoint_chunks" / out.context.arm_id / "pass_10"]


def test_distance_override_is_not_silently_replaced(monkeypatch):
    from types import MethodType
    import molgap.gptrans_execution_backend as owner
    import molgap.gptrans as family
    class SmallModel:
        def _dense_inputs(self):
            pass
        @staticmethod
        def _local_edges():
            pass
        def _shortest_path(self):
            pass
    monkeypatch.setattr(family, "OGBGPTransTiny", SmallModel)
    model = SmallModel()
    model._shortest_path = MethodType(lambda *args: None, model)
    with pytest.raises(ValueError, match="Distance-changing"):
        owner.bind_model_layout(model)
