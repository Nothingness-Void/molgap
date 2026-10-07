"""Static/metadata checks only; no model construction, training or inference."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest

from molgap.gptrans_scale_profile import model_binding
from molgap.evidence_pointers import load_json_object
from molgap.research_memory import reference_qualification

ROOT = Path(__file__).resolve().parents[1]
MARKER = "experiments/pcqm_gptrans_capacity_relations_100k/gpu/scale_ema/rml_plan/reference_qualification.json"


def test_legacy_g1_default_preserved():
    assert model_binding({}) == ("degree_scale_ema999", "degree_initial_state.pt", 5246817)


def test_frozen_local_model_binding():
    assert model_binding({"model_variant": "degree_bond_local_ema999"}) == (
        "degree_bond_local_ema999", "degree_bond_local_ema999_initial.pt", 5871201)


@pytest.mark.parametrize("patch", [
    {"model_variant": "degree_bond_local_cap_ema999"},
    {"model_variant": "degree_bond_local_ema999", "initial_file": "degree_initial_state.pt"},
    {"model_variant": "degree_bond_local_ema999", "expected_parameters": 5246817},
])
def test_changed_scale_binding_rejected(patch):
    with pytest.raises(ValueError):
        model_binding(patch)


def test_local_transport_reuses_owning_train_and_profile():
    root = Path(__file__).resolve().parents[1]
    for name in ("gptrans_scale_profile.py", "gptrans_scale_ema.py", "gptrans_followup_release.py", "gptrans_scale_reference.py"):
        ast.parse((root / "src/molgap" / name).read_text(encoding="utf-8"))
    source = (root / "src/molgap/gptrans_scale_ema.py").read_text(encoding="utf-8")
    assert "native._make_training_state(initial, variant)" in source
    assert "native._optimizer_step" in source and "native._evaluate" in source
    assert '"parameters": parameters' in source
    assert "model_binding(config)" in source


def test_terminal_control_enrollment_does_not_promote_old_claim():
    original = load_json_object(ROOT / "experiments/pcqm_gptrans_capacity_relations_100k/gpu/scale_ema/rml_plan/rml_finalized/trace_manifest.json")
    enrolled = reference_qualification.verify_reference_qualification(ROOT, MARKER)
    assert original["comparison_role"] == "candidate"
    assert original["backtest_eligibility"]["eligible"] is False
    assert enrolled["comparison_role"] == "reference"
    assert enrolled["trace_artifact_sha256"] == original["trace_artifact_sha256"]
    evidence = load_json_object(ROOT / original["terminal_evidence_ref"])
    assert evidence["outcome"]["comparison_status"] == "paired_endpoint"


@pytest.mark.parametrize("attack", ["promotion", "source", "checkpoint", "exposure", "original_claim", "trace"])
def test_control_enrollment_rejects_forged_projection(monkeypatch, attack):
    real = reference_qualification.load_json_object

    def attacked(path):
        value = deepcopy(real(path))
        name = Path(path).name
        if name == "reference_qualification.json" and attack == "promotion":
            value["scientific_promotion"] = True
        elif name == "reference_qualification.json" and attack == "original_claim":
            value["original_comparison_status"] = "strict_causal"
        elif name == "acceptance_projection.json" and attack == "source":
            value["source_acceptance_sha256"] = "0" * 64
        elif name == "acceptance_projection.json" and attack == "checkpoint":
            value["best_ema_model_sha256"] = "0" * 64
        elif name == "acceptance_projection.json" and attack == "exposure":
            value["optimizer_steps"] += 1
        elif name == "qualified_trace_manifest.json" and attack == "trace":
            value["comparability_identity"]["lr_schedule_identity"] = "invented"
        return value

    monkeypatch.setattr(reference_qualification, "load_json_object", attacked)
    with pytest.raises(ValueError):
        reference_qualification.verify_reference_qualification(ROOT, MARKER)
