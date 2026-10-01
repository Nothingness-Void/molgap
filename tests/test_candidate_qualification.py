"""Saved-metadata qualification and real replay admission; no model execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from molgap.evidence_pointers import load_json_object
from molgap.gptrans_candidate_qualification import qualify_author_candidates
from molgap.research_memory import candidate_qualification as qualification
from molgap.research_memory.finalize import verified_receipt
from molgap.research_memory.compiler import _source_digest
from molgap.research_memory.replay import build_replay_pool
from molgap.research_memory.validate import validate_repository_records

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/pcqm_gptrans_author_alignment/gpu"


def marker(mode):
    return f"experiments/pcqm_gptrans_author_alignment/gpu/{mode}/rml_plan/candidate_qualification.json"


@pytest.mark.parametrize("mode", ["degree_scale", "path_bond_mean"])
def test_candidate_qualification_recomputes_strict_without_rewriting_history(mode):
    arm = BASE / mode
    verified_receipt(arm / "rml_plan/rml_finalized")
    original = load_json_object(arm / "rml_plan/rml_finalized/trace_manifest.json")
    original_readiness = load_json_object(arm / "results/comparison_readiness.json")
    manifest, readiness = qualification.verify_candidate_qualification(ROOT, marker(mode))
    assert readiness["comparison_class"] == "STRICT_CAUSAL"
    assert readiness["strict_ready"] is True
    assert readiness["blocker_codes"] == []
    assert original_readiness["comparison_class"] == "PAIRED_ENDPOINT"
    assert original["backtest_eligibility"]["eligible"] is False
    assert manifest["backtest_eligibility"] == {"eligible": True, "exclusion_reasons": []}
    assert manifest["model_identity"] == manifest["comparability_identity"]["architecture_identity"]
    unchanged = set(original) - {"model_identity", "backtest_eligibility"}
    assert {k: original[k] for k in unchanged} == {k: manifest[k] for k in unchanged}
    assert readiness["candidate_artifact_bindings"] == original_readiness["candidate_artifact_bindings"]
    assert readiness["matched_fields"] == original_readiness["matched_fields"]
    assert readiness["mismatched_fields"] == original_readiness["mismatched_fields"]


def test_actual_replay_pool_contains_both_candidates_and_exact_reference():
    records = validate_repository_records(ROOT)["records"]
    pool = build_replay_pool(ROOT, records)
    ids = {"TC-gptrans-author-degree-scale-100k-s42", "TC-gptrans-author-path-bond-mean-100k-s42",
           "TC-gptrans-v5-audit-reference-100k"}
    entries = [e for e in pool["entries"] if e["trajectory_id"] in ids]
    assert len(entries) == 3
    assert {e["trajectory_id"] for e in entries} == ids
    assert len({e["comparability_key"] for e in entries}) == 1
    assert all(e["capability"] == "complete" and e["terminal_endpoint"] == (
        5998080 if e["axis"] == "sample_presentations" else 46860) for e in entries)
    assert all(len(e["prefix_observations"]) == 60 for e in entries)
    assert all(e["native_cost"]["device_hours"] > 0 for e in entries)
    assert all(e["native_cost"]["planned_measurements"] for e in entries)
    assert all(m["status"] == "measured" for e in entries for m in e["native_cost"]["measurements"])
    assert sum(e["native_cost"]["device_hours"] for e in entries if e["comparison_role"] == "candidate") == pytest.approx(7.152377706028333)
    candidates = [e for e in entries if e["comparison_role"] == "candidate"]
    assert all(e["terminal_label"] == {"winner": True, "promotion_passed": True} for e in candidates)
    assert next(e for e in entries if e["comparison_role"] == "reference")["terminal_label"] is None
    assert not any(e["trajectory_id"] in ids for e in pool["exclusions"])


@pytest.mark.parametrize("attack", ["reference_swap", "old_bundle_digest", "candidate_identity",
    "intervention", "events", "costs", "endpoint", "protected_role", "prediction_alignment",
    "missing_binding", "forged_strict", "forged_manifest"])
def test_candidate_qualification_rejects_unbound_or_reinterpreted_evidence(monkeypatch, attack):
    real_load = qualification.load_json_object

    def attacked_load(path):
        value = deepcopy(real_load(path))
        name = Path(path).name
        if attack == "reference_swap" and name == "candidate_qualification.json":
            value["original_bundle_ref"] = "experiments/pcqm_gptrans_v5_audit_reference/results/reference_qualification/reference_bundle.json"
        elif attack == "missing_binding" and name == "candidate_qualification.json":
            del value["artifact_hashes"][value["acceptance_ref"]]
        elif attack == "old_bundle_digest" and name == "comparison_readiness_prelaunch.json":
            value["reference_bundle_sha256"] = "0" * 64
        elif attack == "candidate_identity" and name == "acceptance.json":
            value["arms"]["degree_scale"]["comparison_identity"]["optimizer_steps"] += 1
        elif attack == "intervention" and name == "comparison_readiness_prelaunch.json":
            value["mechanism_id"] = "unapproved"
        elif attack == "events" and name == "role_history.json":
            value["roles"] = []
        elif attack == "costs" and name == "cost_records.json":
            value["costs"] = []
        elif attack == "endpoint" and name == "completion_manifest.json":
            value["optimizer_steps"] -= 1
        elif attack == "protected_role" and name == "completion_manifest.json":
            value["test_dev_role_read"] = True
        elif attack == "prediction_alignment" and name == "prediction_manifest.json":
            value["source_idx_sha256"] = "0" * 64
        elif attack == "forged_strict" and name == "qualified_comparison_readiness.json":
            value["matched_fields"]["optimizer_steps"] += 1
        elif attack == "forged_manifest" and name == "qualified_trace_manifest.json":
            value["exposure"]["sample_presentations"] += 1
        return value

    monkeypatch.setattr(qualification, "load_json_object", attacked_load)
    with pytest.raises(ValueError):
        qualification.verify_candidate_qualification(ROOT, marker("degree_scale"))


def test_qualification_repeated_publication_is_idempotent():
    paths = [BASE / mode / "rml_plan/candidate_qualification.json" for mode in ("degree_scale", "path_bond_mean")]
    before = [p.read_bytes() for p in paths]
    assert len(qualify_author_candidates(ROOT)) == 2
    assert [p.read_bytes() for p in paths] == before
    for path in paths:
        descriptor = load_json_object(path)
        assert descriptor["successor_authorized"] is False
        assert descriptor["model_inference_executed"] is False
        assert descriptor["local_training_executed"] is False


def test_effective_qualification_changes_compiler_source_digest_without_rewriting_bytes():
    path = BASE / "degree_scale/rml_plan/rml_finalized/trace_manifest.json"
    original = load_json_object(path)
    manifest, _ = qualification.verify_candidate_qualification(ROOT, marker("degree_scale"))
    assert _source_digest({"traces": [(path, original)]}, ROOT) != _source_digest({"traces": [(path, manifest)]}, ROOT)


def test_unknown_observed_cost_is_not_replaced_by_budget_estimate():
    records = validate_repository_records(ROOT)["records"]
    altered = deepcopy(records)
    for _, cost in altered["costs"]:
        if cost["cost_event_id"] == "cost-TC-gptrans-author-degree-scale-100k-s42-observed-v1":
            cost["measurement"]["device_hours"] = {"status": "measurement_missing", "value": None}
    pool = build_replay_pool(ROOT, altered)
    entry = next(e for e in pool["entries"] if e["trajectory_id"] == "TC-gptrans-author-degree-scale-100k-s42")
    assert entry["native_cost"]["device_hours"] is None
    assert entry["native_cost"]["planned_measurements"][0]["status"] == "estimated"
