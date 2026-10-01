"""Real saved-evidence checks; no model loading, inference or remote calls."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from molgap.research_memory.candidate_reference import verify_candidate_reference, _terminal_expected
from molgap.research_memory.ema_replay import ema_intervention_worlds
from molgap.server_acceptance import validate_server_scientific_prelaunch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/pcqm_gptrans_recipe_paths_100k"


def test_terminal_ema_comparator_is_exact_observed_reference():
    pointer = "experiments/pcqm_gptrans_recipe_paths_100k/reference/candidate_reference_qualification.json"
    view = verify_candidate_reference(ROOT, pointer)
    assert view["reference_id"] == "pcqm-gptrans-g1-degree-scale-ema999-100k-s42"
    assert view["trajectory_id"] == "TC-gptrans-g1-ema999-100k-s42"
    assert view["comparability_identity"]["ema_semantics"] == "ema999-each-step-selection"
    record = json.loads((ROOT / pointer).read_text())
    forged = deepcopy(record)
    forged["original_manifest_ref"] = "experiments/pcqm_gptrans_input_ema_100k/reference/reference_view.json"
    with pytest.raises(ValueError, match="outside its finalized transaction"):
        _terminal_expected(ROOT, forged)
    forged = deepcopy(record)
    forged["acceptance_arm"] = "degree_path_bond_mean"
    with pytest.raises(ValueError, match="identity"):
        _terminal_expected(ROOT, forged)


@pytest.mark.parametrize("mode,purpose,fields", [
    ("degree_group_decay_ema999", "optimizer_comparison", {"optimizer_identity", "optimizer_mode"}),
    ("degree_path_endpoints_ema999", "mechanism_comparison", {"architecture_config_identity"}),
])
def test_new_arms_bind_only_declared_intervention(mode, purpose, fields):
    prelaunch = json.loads((BASE / "gpu" / mode / "comparison_readiness_prelaunch.json").read_text())
    assert set(prelaunch["mismatched_fields"]) == fields
    result = validate_server_scientific_prelaunch(comparison_prelaunch=prelaunch,
        experiment_purpose=purpose, reference_bundle=json.loads((BASE / "reference/reference_bundle.json").read_text()),
        repo_root=ROOT, reference_bundle_path=BASE / "reference/reference_bundle.json")
    assert result["gate"] == "PASS"


def test_forged_optimizer_replay_world_fails_closed():
    records = {"trajectories": [], "reference_bundles": [], "comparison_readiness": [
        (BASE / "forged.json", {"experiment_purpose": "optimizer_comparison", "strict_ready": True,
         "comparison_class": "STRICT_CAUSAL", "mismatched_fields": {"optimizer_identity": {"candidate": "x", "reference": "y"}}})]}
    with pytest.raises((ValueError, KeyError)):
        ema_intervention_worlds(ROOT, [], records)
