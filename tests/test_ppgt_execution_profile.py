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
    value["arms"][0]["model"]["addons"] = []
    value["arms"][0]["training"]["profiling"] = dict(ppgt_phases=True, record_shapes=False)
    authority.update(purpose=PROFILE_PURPOSE, scientific_training_authorized=False,
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
