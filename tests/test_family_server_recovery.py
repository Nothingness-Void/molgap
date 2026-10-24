"""Server-only shared-session hooks under synthetic CPU fault injection."""
from dataclasses import replace
import hashlib
import json
import os
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
    trajectory = {"trajectory_id": "synthetic-trajectory", "record_mode": "prospective",
        "state_at_start": {"source_commit": ctx.source_commit, "source_config_identity": ctx.arm_identity},
        "actions": [{"action_id": "A001", "run_ids": [ctx.logical_run_id + ":" + ctx.arm_id + ":downstream"],
                      "attempt_ids": ["v1"]}],
        "decision": {"outcome": "ACTIVE"}}
    trajectory_bytes = json.dumps(trajectory, sort_keys=True).encode()
    def reopen():
        return SessionOutputs(tmp_path / "output", ctx, contract=path, adapter=PROFILES["gptrans_t"],
            trajectory_id="synthetic-trajectory", train_role_identity=record["train_role_identity"],
            resume_trajectory=trajectory_bytes)
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
    session, out, _, _ = prepared(tmp_path, mode, passes=11)
    trajectory = json.loads(out.resume_trajectory)

    # A parent-side ledger tick may atomically publish a newer generation after
    # the worker has captured the bytes for this transport chunk.  The resume
    # bundle must retain the generation it actually captured, including its
    # manifest digest, without rereading the live source path.
    import molgap.experiment_resume as owner
    from molgap.experiment_allocation import AllocationLedger
    ledger = out.root / "allocation_ledger.json"
    elapsed = [0.]
    allocation = AllocationLedger(spec_identity=out.context.spec_identity,
        hardware=("synthetic-cpu",), assignments={out.context.arm_id: 0}, clock=lambda: elapsed[0])
    elapsed[0] = 10.
    old_payload = json.dumps(allocation.snapshot("running"), sort_keys=True).encode()
    elapsed[0] = 20.
    new_payload = json.dumps(allocation.snapshot("running"), sort_keys=True).encode()
    ledger.write_bytes(old_payload)
    original_read_bytes = owner.Path.read_bytes
    original_sha256_file = owner.sha256_file
    read_replaced = False
    hash_replaced = False

    def capture_then_replace(path):
        nonlocal read_replaced
        payload = original_read_bytes(path)
        if Path(path).resolve() == ledger.resolve() and not read_replaced:
            read_replaced = True
            replacement = ledger.with_name(".allocation_ledger.next.json")
            replacement.write_bytes(new_payload)
            os.replace(replacement, ledger)
            return payload
        return payload

    def capture_digest_then_replace(path):
        nonlocal hash_replaced
        digest = original_sha256_file(path)
        if Path(path).resolve() == ledger.resolve() and not hash_replaced:
            hash_replaced = True
            replacement = ledger.with_name(".allocation_ledger.next.json")
            replacement.write_bytes(new_payload)
            os.replace(replacement, ledger)
        return digest

    with monkeypatch.context() as scoped:
        scoped.setattr(owner.Path, "read_bytes", capture_then_replace)
        scoped.setattr(owner, "sha256_file", capture_digest_then_replace)
        native_bundle = owner.create_resume_bundle
        def export_committed(root, context, trajectory, destination):
            assert not (root / "session_commit.json").exists()
            trace = json.loads((root / "canonical_trace.json").read_bytes())
            row = trace["observations"][-1]
            assert row["optimizer_step"] == 20
            assert row["checkpoint_identity"] == sha256_file(root / "last_checkpoint.pt")
            return native_bundle(root, context, trajectory, destination)
        scoped.setattr(owner, "create_resume_bundle", export_committed)
        session.run()

    assert read_replaced is True
    assert hash_replaced is False
    chunk_root = out.root.parent / "checkpoint_chunks" / out.context.arm_id
    assert sorted(path.name for path in chunk_root.iterdir() if path.is_dir()) == ["pass_10"]
    bundle = chunk_root / "pass_10"
    checked = owner.validate_resume_bundle(bundle)
    assert checked["status"] == "TRANSPORT_VERIFIED"
    manifest = checked["manifest"]
    assert manifest["cursor"]["optimizer_step"] == 20
    assert manifest["cursor"]["sample_presentations"] == 2560
    ledger_pointer = manifest["cost_segments"][0]
    assert ledger_pointer["sha256"] == hashlib.sha256(old_payload).hexdigest()
    assert (bundle / ledger_pointer["path"]).read_bytes() == old_payload
    assert ledger.read_bytes() == new_payload


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
