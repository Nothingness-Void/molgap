"""Small CPU fixtures: observation equivalence and profiling-only release scope."""
from copy import deepcopy
import json

import pytest

from molgap.family_execution_profile import FamilyExecutionProfiler, ProfileConfig
from molgap.modern2d_training_release import PROFILE_PURPOSE, validate_configured_authority
from test_family_training_session import assert_equal
from test_modern2d_integration_session import make_preview_session
from test_modern2d_scientific_input import configured
from test_modern2d_formal_training import case
from test_experiment_workflow import _git, _write_json


def test_ppgt_observer_preserves_state_and_restores_functions(tmp_path):
    from molgap import ppgt_author, ppgt_author_features
    original = (ppgt_author.validated_topology_views, ppgt_author.positional_inputs,
                ppgt_author.sinusoidal_pe, ppgt_author_features.rrwp)
    baseline, *_ = make_preview_session(tmp_path / "base", "ppgt_small", "2", [])
    baseline.run(should_pause=lambda: True)
    expected = deepcopy(baseline.state_dict())
    observed, *_ = make_preview_session(tmp_path / "observed", "ppgt_small", "2", [])
    with FamilyExecutionProfiler(observed, tmp_path / "profile",
            ProfileConfig(0, 1, record_shapes=False, ppgt_phases=True)) as profiler:
        observed.run(should_pause=lambda: True)
    assert_equal(expected, observed.state_dict())
    assert original == (ppgt_author.validated_topology_views, ppgt_author.positional_inputs,
                        ppgt_author.sinusoidal_pe, ppgt_author_features.rrwp)
    names = {p["name"] for p in profiler.report["phases_by_host_self_time"]}
    assert {"ppgt/topology_views", "ppgt/rrwp", "ppgt/pair_stem", "ppgt/backbone",
            "ppgt/attention", "ppgt/ffn", "ppgt/readout", "backward", "optimizer"} <= names
    baseline.close()
    observed.close()


def test_profile_only_compilation_and_phase_gate(configured):
    from molgap.experiment_input import resolve_experiment_input
    from molgap.family_training_runner import read_recipe
    from molgap.modern2d_training_release import execution_extension, check_plan
    c = configured
    if c.value["arms"][0]["model"]["family"] != "ppgt_small":
        pytest.skip("PPGT-only diagnostic; ESA has no release through this gate")
    value, authority = deepcopy(c.value), deepcopy(c.authority)
    # The profile phase permits one context_only arm and no same-run pair.
    value["arms"] = [dict(value["arms"][1], purpose="context_only")]
    value.pop("same_run")
    value["arms"][0]["model"]["addons"] = []
    value["arms"][0]["training"]["profiling"] = dict(ppgt_phases=True, record_shapes=False)
    authority.update(purpose=PROFILE_PURPOSE, scientific_training_authorized=False, arms=["candidate"],
                     interfaces=authority["interfaces"][:1], recipes={"candidate": authority["recipes"]["candidate"]},
                     scratch_optimizer_updates_per_model=6)
    _write_json(c.path, value)
    _write_json(c.repo / value["authority_ref"], authority)
    _git(c.repo, "add", ".")
    _git(c.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Bounded profiling fixture")
    _, spec, workflow, _ = resolve_experiment_input(c.path, c.repo)
    plan = json.loads((c.repo / workflow["acceptance_plan"]).read_bytes())
    assert check_plan(spec, c.repo, plan)["noReplay"]
    extension = execution_extension(spec, c.repo)
    assert [p.name for p in extension.phases] == ["preflight"]
    recipe = read_recipe(c.repo / workflow["arms"][0]["recipe"])[1]
    assert validate_configured_authority(authority, spec.to_dict(), {"candidate": recipe}) == 1800
    for patch in (dict(scientific_training_authorized=True), dict(scratch_optimizer_updates_per_model=7),
                  dict(roles=["train_500k", "common_dev"], scale_500k_authorized=True),
                  dict(allocated_t4_hours_ceiling=2)):
        with pytest.raises(ValueError):
            validate_configured_authority({**authority, **patch}, spec.to_dict())


def test_profile_runtime_rejects_default_before_payload(case, tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from molgap.experiment_family_workflow import RunContext
    from molgap.modern2d_training import Modern2DFormalTrainingContext, Modern2DFormalTrainingProfile
    from molgap.family_training_runner import normalized_file_sha
    from test_modern2d_qualification import permit_record
    if case.recipe.family != "ppgt_small":
        pytest.skip("PPGT-only authority")
    recipe = replace(case.recipe, train_rows=100000, development_rows=50000, optimizer_steps=46860,
        dataset=replace(case.recipe.dataset, manifest="training_input.json"),
        formal_training=Modern2DFormalTrainingProfile(purpose="configured-screen"))
    context = RunContext("fixture", "profile-run", "candidate", "a" * 64, "b" * 64,
        "ppgt_small", "3", "c" * 40, "d" * 64, "e" * 64, "f" * 64,
        "kaggle", "fixture", "fixture/profile", None)
    declaration = dict(logical_run_id=context.logical_run_id,
        arms=[dict(arm_id="candidate", family=dict(name="ppgt_small", version="3"))],
        platform=dict(name="kaggle", accelerator="NvidiaTeslaT4", device_count=2))
    authority = dict(format="molgap-modern2d-configured-authority-v1", owner="server",
        purpose=PROFILE_PURPOSE, user_authorized=True, scientific_training_authorized=False,
        logical_run_id=context.logical_run_id, arms=["candidate"], interfaces=[["ppgt_small", "3"]],
        seeds=[42], physical_batch_size=128, roles=["train_100k", "common_dev"],
        scale_500k_authorized=False, protected_role_access_authorized=False, extra_seed_authorized=False,
        automatic_successor_authorized=False, full_training_authorized=False,
        allocated_t4_hours_ceiling=1, scratch_optimizer_updates_per_model=6, single_arm_reason="Profile only",
        recipes=dict(candidate=dict(horizon_passes=60, horizon_steps=46860)),
        run_reference=context.run_reference,
        gpu=dict(owner="fixture", accelerator="NvidiaTeslaT4", allocated_devices=2,
                 wall_ceiling_seconds=1800, allocated_t4_hours_ceiling=1))
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    native = Modern2DFormalTrainingContext(context, recipe.identity, normalized_file_sha(path),
        permit_record("train_100k")[0], permit_record("common_dev")[0], "1" * 64, "2" * 64,
        max_wall_seconds=1800, configured=True)
    bindings = []
    monkeypatch.setattr(Modern2DFormalTrainingContext, "_validate_input_binding",
                        lambda *args: bindings.append(True))
    options = dict(recipe=recipe, context=context,
        spec=SimpleNamespace(identity=context.spec_identity, to_dict=lambda: declaration),
        authority_path=path, input_root=tmp_path)
    with pytest.raises(ValueError, match="cannot open development or release formal training"):
        native.validate_before_payload(**options)
    assert bindings == []
    native.validate_before_payload(**options, training_only=True)
    assert bindings == [True]
