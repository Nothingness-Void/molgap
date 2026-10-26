"""Real tiny CPU sessions feed the existing saved-artifact inspector."""
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import json
import pytest
import torch

from molgap.experiment_family_workflow import RunContext, inspect_output
from molgap.family_training_outputs import SessionOutputs
from molgap.research_memory.trace import file_digest
from test_modern2d_formal_training import case, actual_session
from test_modern2d_addon_session import composed


@pytest.mark.parametrize("step_clock", [False, True], ids=["pass-clock", "step-clock"])
def test_configured_addon_outputs_are_independently_inspected(case, tmp_path, step_clock):
    case.recipe = composed(case)
    if step_clock:
        from molgap.modern2d_training import Modern2DFormalTrainingProfile
        case.recipe = replace(case.recipe, optimizer_steps=3,
            schedule=replace(case.recipe.schedule, total_units=3),
            formal_training=Modern2DFormalTrainingProfile("configured-screen", horizon_passes=2,
                horizon_steps=3, evaluation_every_steps=3))
        checkpoints = []
        prefix = actual_session(case, checkpoints)
        assert prefix.run(should_pause=lambda: True).optimizer_step == 1
        assert checkpoints[-1]["cursor"]["selected_step"] is None
        resumed = actual_session(case, [])
        resumed.load_state_dict(deepcopy(checkpoints[-1]))
        assert resumed.run().optimizer_step == 3
        resumed_state = deepcopy(resumed.state_dict())
        prefix.close()
        resumed.close()
    session = actual_session(case, [])
    recipe = case.recipe
    from molgap.family_training_runner import build_screen_recipe
    from molgap.experiment_family_workflow import tensor_digest
    record = build_screen_recipe(recipe.mode, family=recipe.interface, configuration=recipe,
        train_role_identity="3" * 64, development_role_identity="4" * 64,
        source_idx_sha256=tensor_digest(torch.tensor(case.data.development_source_indices), role="source_idx"),
        target_sha256=tensor_digest(torch.cat([g.y for g in case.data.development]), role="target"))
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    context = RunContext("fixture", "fixture-run", "candidate", "a" * 64, "b" * 64,
        recipe.family, recipe.family_version, "c" * 40, "d" * 64, "e" * 64,
        file_digest(path), "local", "fixture", "fixture/run", None)
    trajectory = json.dumps(dict(trajectory_id="fixture-trajectory", record_mode="prospective",
        state_at_start=dict(source_commit=context.source_commit, source_config_identity=context.arm_identity),
        actions=[dict(action_id="A001", run_ids=[context.logical_run_id + ":candidate:downstream"],
                      attempt_ids=["v1"])], decision=dict(outcome="ACTIVE"))).encode()
    output = SessionOutputs(tmp_path / "output", context, contract=path,
        adapter="modern2d-formal-session-v1", trajectory_id="fixture-trajectory",
        train_role_identity=record["train_role_identity"],
        resume_trajectory=trajectory if step_clock else None)
    if step_clock:
        # Shorten only the fixture's chunk clock to cover pending step-window
        # metrics within the four-update synthetic safety bound.
        output.policy = replace(output.policy, chunk_passes=1)
    session.loaders.attach_role_audit(output.role_audit)
    session.checkpoint, session.selected, session.pass_finished = output.checkpoint, output.selected, output.pass_finished
    progress = session.run()
    if step_clock:
        from test_family_training_session import assert_equal
        assert_equal(resumed_state, session.state_dict())
        assert progress.next_batch == 1
    runtime = dict(platform="local", account="fixture", precision="fp32",
        source_commit=context.source_commit, source_archive_sha256=context.source_archive_sha256)
    costs = [dict(metric="wall_seconds", unit="seconds", status="missing", value=None,
                  semantics="process_wall", hardware="synthetic-cpu"),
             dict(metric="device_seconds", unit="seconds", status="missing", value=None,
                  semantics="allocated_device", hardware="synthetic-cpu")]
    report = output.complete(progress, runtime=runtime, costs=costs)
    rows = json.loads((output.root / "canonical_trace.json").read_bytes())["observations"]
    observed = [row for row in rows if row["event"] == "observation"]
    assert observed
    if step_clock:
        assert [row["optimizer_step"] for row in observed] == [3]
        assert [row["epoch_or_pass"] for row in observed] == [1.5]
        from molgap.experiment_resume import validate_resume_bundle, restore_resume_bundle
        chunk = output.root.parent / "checkpoint_chunks/candidate/pass_01"
        manifest = validate_resume_bundle(chunk)["manifest"]
        assert manifest["cursor"]["pass_index"] == 1
        assert len(manifest["cursor"]["pending_absolute"]) == 2
        assert all(type(value) is float for value in manifest["cursor"]["pending_absolute"])
        restored = tmp_path / "restored-chunk"
        restore_resume_bundle(chunk, restored)
        state = torch.load(restored / "last_checkpoint.pt", map_location="cpu", weights_only=True)
        assert all(isinstance(value, torch.Tensor) for value in state["cursor"]["pending_absolute"])
        state.pop("context")
        state.setdefault("ema", None)
        continuation = actual_session(case, [])
        continuation.load_state_dict(state)
        continuation.run()
        assert_equal(session.state_dict(), continuation.state_dict())
        continuation.close()
    assert all(row["wall_time_seconds"] is not None and
               row["cumulative_wall_time_seconds"] is not None and
               row["checkpoint_identity"] is not None and
               row["learning_rate"] is not None for row in observed)
    assert observed[-1]["learning_rate"] == session.state_dict()["optimizer"]["param_groups"][0]["lr"]
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["observed"]["execution_scope"] == "CONFIGURED_SCREEN"
    assert report["scientific_acceptance"] == "NOT_EVALUATED"
    assert all(c["status"] == "missing" for c in report["observed"]["costs"])
    snapshot = {p.name: file_digest(p) for p in output.root.iterdir() if p.is_file()}
    inspected = inspect_output(output.root, context=context, expected=record["acceptance_requirements"])
    assert inspected["status"] == "MECHANICALLY_VERIFIED", inspected
    assert snapshot == {p.name: file_digest(p) for p in output.root.iterdir() if p.is_file()}
    with (output.root / "canonical_trace.json.context.json").open("ab") as stream:
        stream.write(b" ")
    assert inspect_output(output.root, context=context, expected=record["acceptance_requirements"])["status"] == "BLOCKED"
    session.close()
