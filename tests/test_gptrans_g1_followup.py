"""Release/metadata regressions only; no training, inference or platform call."""
from copy import deepcopy
from pathlib import Path
import json
import pytest

from molgap.gptrans_author_variants import MODES, PATH_MODES, SCALED_MODES
from molgap.pcqm_gptrans_v4 import _ema_decay, _scientific_fields
from molgap.research_memory.candidate_reference import verify_candidate_reference
from molgap.server_acceptance import validate_server_scientific_prelaunch
from molgap.research_memory.ema_replay import ema_intervention_worlds
from molgap.research_memory.backtest import _comparison_key

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/pcqm_gptrans_input_ema_100k"


def test_modes_and_ema_delta_are_explicit():
    assert {"degree_path_bond_mean", "degree_scale_ema999"} <= set(MODES)
    assert "degree_path_bond_mean" in PATH_MODES & SCALED_MODES
    assert "degree_scale_ema999" in SCALED_MODES - PATH_MODES
    assert _ema_decay("degree_scale") == .9999
    assert _ema_decay("degree_path_bond_mean") == .9999
    assert _ema_decay("degree_scale_ema999") == .999
    a = _scientific_fields("degree_scale")
    b = _scientific_fields("degree_scale_ema999")
    assert {k for k in a if a[k] != b[k]} == {"selection_fingerprint"}


def test_real_candidate_reference_is_a_view_not_another_trace():
    pointer = "experiments/pcqm_gptrans_input_ema_100k/reference/candidate_reference_qualification.json"
    view = verify_candidate_reference(ROOT, pointer)
    assert view["comparison_role"] == "reference"
    assert view["reference_id"] == "pcqm-gptrans-author-degree-scale-100k-s42"
    assert view["trajectory_id"] == "TC-gptrans-author-degree-scale-100k-s42"
    assert view["exposure"]["optimizer_steps"] == 46860
    bundle = json.loads((BASE / "reference/reference_bundle.json").read_text())
    forged = deepcopy(bundle)
    forged["checkpoint_identity"] = "0" * 64
    with pytest.raises(ValueError, match="observed identities"):
        verify_candidate_reference(ROOT, pointer, expected_bundle=forged)


@pytest.mark.parametrize("mode,purpose,fields", [
    ("degree_path_bond_mean", "mechanism_comparison", {"architecture_config_identity"}),
    ("degree_scale_ema999", "ema_comparison", {"ema_decay", "checkpoint_selection_identity"}),
])
def test_each_arm_passes_real_reference_release(mode, purpose, fields):
    prelaunch = json.loads((BASE / "gpu" / mode / "comparison_readiness_prelaunch.json").read_text())
    assert set(prelaunch["mismatched_fields"]) == fields
    result = validate_server_scientific_prelaunch(comparison_prelaunch=prelaunch,
        experiment_purpose=purpose, reference_bundle=json.loads((BASE / "reference/reference_bundle.json").read_text()),
        repo_root=ROOT, reference_bundle_path=BASE / "reference/reference_bundle.json")
    assert result["gate"] == "PASS"


def test_ema_worlds_do_not_relax_default_grouping():
    reference = json.loads((BASE / "reference/reference_view.json").read_text())
    candidate = deepcopy(reference)
    candidate["comparison_role"] = "candidate"
    candidate["comparability_identity"]["ema_semantics"] = "ema999-each-step-selection"
    assert _comparison_key(candidate) != _comparison_key(reference)
    records = {"trajectories": [], "reference_bundles": [], "comparison_readiness": []}
    assert ema_intervention_worlds(ROOT, [candidate, reference], records) == [candidate, reference]


def test_forged_ema_world_fails_closed():
    records = {"trajectories": [], "reference_bundles": [], "comparison_readiness": [
        (BASE / "forged.json", {"experiment_purpose": "ema_comparison", "strict_ready": True,
            "comparison_class": "STRICT_CAUSAL", "mismatched_fields": {"ema_decay": {"candidate": .999, "reference": .9999}}})]}
    with pytest.raises((ValueError, KeyError)):
        ema_intervention_worlds(ROOT, [], records)


def test_acceptance_mode_flag_is_not_shadowed_by_native_field_loop():
    import ast
    import inspect
    from molgap.gptrans_author_acceptance import accept_training_outputs
    tree = ast.parse(inspect.getsource(accept_training_outputs))
    assert all(not any(isinstance(n, ast.Name) and n.id == "legacy" for n in ast.walk(loop.target))
               for loop in ast.walk(tree) if isinstance(loop, ast.For))


@pytest.mark.parametrize("delta,interval,passed", [
    (-.006, [-.007, -.005], True),
    (-.006, [-.012, .001], False),
    (-.002, [-.003, -.001], False),
    (-.003, [-.004, -.002], False),
])
def test_followup_gate_requires_magnitude_and_paired_sign(delta, interval, passed):
    from molgap.gptrans_author_acceptance import _followup_material_gate
    assert _followup_material_gate({"candidate_minus_reference_eV": delta,
        "paired_row_bootstrap": {"ci95": interval}}, .003) is passed


def test_terminal_followups_admit_actual_distinct_replay_worlds():
    if not (BASE / "gpu/results/acceptance.json").is_file():
        pytest.skip("actual terminal outputs not yet accepted")
    from molgap.research_memory.validate import validate_repository_records
    from molgap.research_memory.replay import build_replay_pool
    pool = build_replay_pool(ROOT, validate_repository_records(ROOT)["records"])
    ids = {"TC-gptrans-g1-path-mean-100k-s42", "TC-gptrans-g1-ema999-100k-s42"}
    candidates = [e for e in pool["entries"] if e["trajectory_id"] in ids and e["comparison_role"] == "candidate"]
    assert {e["trajectory_id"] for e in candidates} == ids
    assert len(candidates) == 2
    for entry in candidates:
        assert entry["capability"] == "complete" and len(entry["prefix_observations"]) == 60
        assert any(e["comparison_role"] == "reference" and e["comparability_key"] == entry["comparability_key"]
                   for e in pool["entries"])
    ema = next(e for e in candidates if "ema999" in e["trajectory_id"])
    assert ema["intervention_world"]["purpose"] == "ema_comparison"
    assert ema["observed_comparability_identity"]["ema_semantics"] == "ema999-each-step-selection"
    assert ema["terminal_label"] == {"winner": True, "promotion_passed": True}
