"""Real tiny CPU sessions feed the existing saved-artifact inspector."""
from pathlib import Path
import json
import torch

from molgap.experiment_family_workflow import RunContext, inspect_output
from molgap.family_training_outputs import SessionOutputs
from molgap.research_memory.trace import file_digest
from test_modern2d_formal_training import case, actual_session
from test_modern2d_addon_session import composed


def test_configured_addon_outputs_are_independently_inspected(case, tmp_path):
    case.recipe = composed(case)
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
    output = SessionOutputs(tmp_path / "output", context, contract=path,
        adapter="modern2d-formal-session-v1", trajectory_id="fixture-trajectory",
        train_role_identity=record["train_role_identity"])
    session.checkpoint, session.selected, session.pass_finished = output.checkpoint, output.selected, output.pass_finished
    progress = session.run()
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
