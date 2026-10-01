"""Release/metadata regressions only; no training, inference or platform call."""
from copy import deepcopy
from pathlib import Path
import json
import pytest

from molgap.gptrans_author_variants import MODES, PATH_MODES, SCALED_MODES
from molgap.pcqm_gptrans_v4 import _ema_decay, _scientific_fields
from molgap.research_memory.candidate_reference import verify_candidate_reference
from molgap.server_acceptance import validate_server_scientific_prelaunch

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
