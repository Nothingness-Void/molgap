"""Research mechanisms traverse the real shared update/output/inspection owners."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from molgap.experiment_execution import build_addon_declaration
from molgap.experiment_family_workflow import RunContext, inspect_output, tensor_digest
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_recipe import ScheduleSpec
from molgap.family_training_runner import build_screen_recipe
from molgap.modern2d_training import Modern2DFormalTrainingProfile
from molgap.research_memory.trace import file_digest
from test_modern2d_formal_training import case, actual_session
from test_family_training_session import assert_equal


@pytest.mark.parametrize("variant", ["mean_readout", "residual_scale", "both"])
def test_research_mechanism_session_resume_and_complete_outputs(case, tmp_path, variant):
    ppgt = case.recipe.family == "ppgt_small"
    name = ("ppgt_research_" + variant
            if ppgt else "esa_identity_residual")
    addon = build_addon_declaration(case.recipe.interface, name,
        repo_root=Path(__file__).resolve().parents[1])
    case.recipe = replace(case.recipe, mode="composed", addons=(addon,),
        formal_training=Modern2DFormalTrainingProfile("configured-screen", horizon_passes=4,
            horizon_steps=4, execution_mode="batched-v1" if ppgt else "original"),
        schedule=ScheduleSpec("cosine_step", 4, reference_epochs=4))
    first, saved = actual_session(case, []), []
    first.checkpoint = lambda progress, state: saved.append(deepcopy(state))
    first.run(should_pause=lambda: True)
    resumed = actual_session(case, [])
    resumed.load_state_dict(deepcopy(saved[-1]))
    resumed.run()
    full = actual_session(case, [])
    recipe = case.recipe
    record = build_screen_recipe(recipe.mode, family=recipe.interface, configuration=recipe,
        train_role_identity="3" * 64, development_role_identity="4" * 64,
        source_idx_sha256=tensor_digest(torch.tensor(case.data.development_source_indices), role="source_idx"),
        target_sha256=tensor_digest(torch.cat([g.y for g in case.data.development]), role="target"))
    contract = tmp_path / "research-contract.json"
    contract.write_text(json.dumps(record), encoding="utf-8")
    context = RunContext("fixture", "research-fixture", "candidate", "a" * 64, "b" * 64,
        recipe.family, recipe.family_version, "c" * 40, "d" * 64, "e" * 64,
        file_digest(contract), "local", "fixture", "fixture/run", None)
    outputs = SessionOutputs(tmp_path / "output", context, contract=contract,
        adapter="modern2d-formal-session-v1", trajectory_id="research-fixture-trajectory",
        train_role_identity=record["train_role_identity"])
    full.checkpoint, full.selected, full.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    progress = full.run()
    assert_equal(full.state_dict(), resumed.state_dict())
    runtime = dict(platform="local", account="fixture", precision="fp32",
        source_commit=context.source_commit, source_archive_sha256=context.source_archive_sha256)
    costs = [dict(metric=key, unit="seconds", status="missing", value=None,
        semantics=semantics, hardware="synthetic-cpu") for key, semantics in
        (("wall_seconds", "process_wall"), ("device_seconds", "allocated_device"))]
    report = outputs.complete(progress, runtime=runtime, costs=costs)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert inspect_output(outputs.root, context=context, expected=record["acceptance_requirements"])["status"] == "MECHANICALLY_VERIFIED"
    for file in ("selected_model.pt", "last_checkpoint.pt", "development_predictions.pt", "canonical_trace.json"):
        assert (outputs.root / file).is_file()
    observations = json.loads((outputs.root / "canonical_trace.json").read_bytes())["observations"]
    assert observations[-1]["event"] == "observation"
    assert observations[-1]["optimizer_step"] == 4
    for session in (first, resumed, full):
        session.close()
