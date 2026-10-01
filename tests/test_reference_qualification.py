"""Read-only reference recovery and fail-closed enrollment regressions."""
from copy import deepcopy
from pathlib import Path

import pytest

from molgap.comparison_readiness import reference_bundle_digest
from molgap.evidence_pointers import load_json_object
from molgap.research_memory import reference_qualification as qualification
from molgap.research_memory.finalize import verified_receipt
from molgap.server_acceptance import validate_server_scientific_prelaunch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/pcqm_gptrans_v5_audit_reference"
MARKER = "experiments/pcqm_gptrans_v5_audit_reference/rml_plan/reference_qualification.json"
BUNDLE = BASE / "results/reference_qualification/reference_bundle.json"


def test_reference_enrollment_preserves_immutable_receipt_and_observations():
    verified_receipt(BASE / "rml_plan/rml_finalized")
    original = load_json_object(BASE / "rml_plan/rml_finalized/trace_manifest.json")
    enrolled = qualification.verify_reference_qualification(ROOT, MARKER)
    assert original["backtest_eligibility"]["eligible"] is False
    assert enrolled["backtest_eligibility"]["eligible"] is True
    assert enrolled["reference_id"] == "pcqm-gptrans-v5-audit-reference-s42"
    assert enrolled["trace_artifact_sha256"] == original["trace_artifact_sha256"]


def test_future_release_accepts_qualified_reference_without_rewriting_old_plan():
    bundle = load_json_object(BUNDLE)
    old = load_json_object(ROOT / "experiments/pcqm_gptrans_author_alignment/gpu/degree_scale/comparison_readiness_prelaunch.json")
    future = deepcopy(old)
    future["reference_bundle_id"] = bundle["reference_bundle_id"]
    future["reference_bundle_sha256"] = reference_bundle_digest(bundle)
    result = validate_server_scientific_prelaunch(
        comparison_prelaunch=future, experiment_purpose="mechanism_comparison",
        reference_bundle=bundle, repo_root=ROOT, reference_bundle_path=BUNDLE)
    assert result["gate"] == "PASS"
    assert result["planned_status"] == "PRELAUNCH_STRICT_PLANNED"
    assert old["reference_bundle_id"] != future["reference_bundle_id"]


def test_original_candidate_claims_and_reference_release_remain_unchanged():
    old_path = BASE / "results/terminal/reference_bundle.json"
    old_bundle = load_json_object(old_path)
    plan = load_json_object(ROOT / "experiments/pcqm_gptrans_author_alignment/gpu/degree_scale/comparison_readiness_prelaunch.json")
    with pytest.raises(ValueError, match="distinct"):
        validate_server_scientific_prelaunch(comparison_prelaunch=plan,
            experiment_purpose="mechanism_comparison", reference_bundle=old_bundle,
            repo_root=ROOT, reference_bundle_path=old_path)
    for mode in ("degree_scale", "path_bond_mean"):
        readiness = load_json_object(ROOT / f"experiments/pcqm_gptrans_author_alignment/gpu/{mode}/results/comparison_readiness.json")
        assert readiness["comparison_class"] == "PAIRED_ENDPOINT"
        assert readiness["strict_ready"] is False


@pytest.mark.parametrize("attack", ["trace_identity", "reference_identity", "eligibility",
                                    "events", "source_identity", "missing_binding", "digest"])
def test_reference_enrollment_rejects_forged_metadata(monkeypatch, attack):
    real_load = qualification.load_json_object

    def attacked_load(path):
        value = deepcopy(real_load(path))
        name = Path(path).name
        if attack == "trace_identity" and name == "qualified_trace_manifest.json":
            value["exposure"]["optimizer_steps"] += 1
        elif attack == "reference_identity" and name == "qualified_trace_manifest.json":
            value["reference_id"] = "fake-reference"
        elif attack == "eligibility" and name == "qualified_trace_manifest.json":
            value["backtest_eligibility"]["exclusion_reasons"] = ["unavailable"]
        elif attack == "events" and name == "role_history.json":
            value["roles"] = []
        elif attack == "source_identity" and Path(path) == BUNDLE:
            value["comparison_identity"]["optimizer_steps"] += 1
        elif attack in {"missing_binding", "digest"} and name == "reference_qualification.json":
            key = value["qualified_bundle_ref"]
            if attack == "missing_binding":
                del value["artifact_hashes"][key]
            else:
                value["artifact_hashes"][key] = "0" * 64
        return value

    monkeypatch.setattr(qualification, "load_json_object", attacked_load)
    with pytest.raises(ValueError):
        qualification.verify_reference_qualification(ROOT, MARKER)


def test_release_rechecks_qualification_instead_of_only_bundle_structure(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("qualification evidence unavailable")

    monkeypatch.setattr(qualification, "verify_reference_qualification", unavailable)
    bundle = load_json_object(BUNDLE)
    plan = load_json_object(ROOT / "experiments/pcqm_gptrans_author_alignment/gpu/degree_scale/comparison_readiness_prelaunch.json")
    plan["reference_bundle_id"] = bundle["reference_bundle_id"]
    plan["reference_bundle_sha256"] = reference_bundle_digest(bundle)
    with pytest.raises(ValueError, match="qualification evidence unavailable"):
        validate_server_scientific_prelaunch(comparison_prelaunch=plan,
            experiment_purpose="mechanism_comparison", reference_bundle=bundle,
            repo_root=ROOT, reference_bundle_path=BUNDLE)


def test_qualification_cannot_be_borrowed_by_another_bundle():
    bundle = deepcopy(load_json_object(BUNDLE))
    bundle["reference_bundle_id"] = "forged-copy"
    with pytest.raises(ValueError, match="supplied reference bundle"):
        qualification.verify_reference_qualification(ROOT, MARKER, expected_bundle=bundle)
    with pytest.raises(ValueError, match="supplied bundle path"):
        qualification.verify_reference_qualification(ROOT, MARKER,
            expected_bundle_path=BASE / "results/terminal/reference_bundle.json")
