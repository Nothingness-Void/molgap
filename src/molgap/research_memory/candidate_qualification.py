"""Additive qualification against the exact, prospectively frozen control.

This recovers terminal evidence bindings, not prospective knowledge or a new
scientific outcome. Original finalization inventories remain immutable.
"""
from copy import deepcopy
from pathlib import Path

from molgap.comparison_readiness import (
    assess_comparison_readiness, reference_bundle_digest,
    validate_comparison_prelaunch, validate_comparison_readiness,
)
from molgap.evidence_pointers import load_json_object
from .finalize import verified_receipt
from .paths import resolve_repo_pointer, verify_bound_artifact
from .reference_qualification import verify_reference_qualification
from .schemas import validate_trace_manifest
from .trace import load_canonical_trace, validate_manifest_trace


def assess_candidate_qualification(root: Path, record: dict) -> tuple[dict, dict]:
    """Derive admission only from retained, hash-bound original observations."""
    root = Path(root).resolve()
    if record.get("format") != "molgap-candidate-qualification-v1":
        raise ValueError("unsupported candidate qualification")
    bindings = record["artifact_hashes"]
    for ref, digest in bindings.items():
        verify_bound_artifact(root, ref, digest)

    def load(field):
        ref = record[field]
        if ref not in bindings:
            raise ValueError(f"candidate qualification has unbound {field}")
        return load_json_object(resolve_repo_pointer(root, ref))

    trajectory = load("trajectory_ref")
    directory = resolve_repo_pointer(root, record["trajectory_ref"]).parent
    verified_receipt(directory)
    if record["original_manifest_ref"] != (directory / "trace_manifest.json").relative_to(root).as_posix():
        raise ValueError("candidate manifest is not from its immutable transaction")
    original = validate_trace_manifest(load("original_manifest_ref"))
    evidence = load_json_object(directory / "v5_evidence.json")
    terminal = load_json_object(directory / "terminal_input.json")
    prelaunch = validate_comparison_prelaunch(load("prelaunch_ref"))
    old = load("original_readiness_ref")
    if (terminal["comparison_readiness_ref"] != record["original_readiness_ref"]
            or any(record[k] not in bindings or terminal["artifact_hashes"].get(record[k]) != bindings.get(record[k])
                   for k in ("original_readiness_ref", "prelaunch_ref", "acceptance_ref"))):
        raise ValueError("qualification inputs were not frozen by the terminal transaction")
    if (original["comparison_role"] != "candidate"
            or trajectory["record_mode"] != "prospective"
            or trajectory["decision"]["outcome"] == "ACTIVE"
            or evidence["outcome"]["execution_status"] != "complete"
            or evidence["outcome"]["artifact_status"] != "accepted"
            or old["candidate_id"] != evidence["evidence_id"]
            or evidence["evidence_id"] not in trajectory["result"]["evidence_ids"]
            or (directory / "v5_evidence.json").relative_to(root).as_posix()
                not in trajectory["result"]["evidence_refs"]
            or (original["trajectory_id"], original["run_id"]) != (
                trajectory["trajectory_id"], terminal["run_id"])):
        raise ValueError("qualification requires the candidate's own accepted terminal identity")

    ref_record = load("reference_qualification_ref")
    ref_manifest = verify_reference_qualification(root, record["reference_qualification_ref"])
    if ref_record["original_bundle_ref"] != record["original_bundle_ref"]:
        raise ValueError("qualification cannot substitute another prospective reference")
    bundle = load_json_object(resolve_repo_pointer(root, ref_record["qualified_bundle_ref"]))
    original_bundle = load("original_bundle_ref")
    frozen_ref = original["reference_id"]
    if (frozen_ref not in trajectory["state_at_start"]["reference_ids"]
            or frozen_ref not in trajectory["decision_state"]["active_reference_ids"]
            or frozen_ref != bundle["reference_id"]
            or frozen_ref != old["reference_id"] or frozen_ref != prelaunch["reference_id"]
            or any(r["reference_bundle_id"] != original_bundle["reference_bundle_id"]
                   or r["reference_bundle_sha256"] != reference_bundle_digest(original_bundle)
                   for r in (old, prelaunch))):
        raise ValueError("candidate did not prospectively freeze this exact reference")
    for key in ("experiment_purpose", "declared_intervention_fields", "mechanism_id", "intervention_group_id",
                "matched_fields", "mismatched_fields"):
        if old[key] != prelaunch[key]:
            raise ValueError("qualification changed the declared intervention or planned identity")

    acceptance = load("acceptance_ref")
    arm = acceptance["arms"][record["acceptance_arm"]]
    if acceptance.get("accepted") is not True or arm.get("accepted") is not True:
        raise ValueError("candidate saved outputs were not accepted")
    identity = dict(old["matched_fields"])
    identity.update({k: v["candidate"] for k, v in old["mismatched_fields"].items()})
    if identity != arm["comparison_identity"]:
        raise ValueError("observed candidate identity differs from accepted outputs")
    candidate_bindings = old["candidate_artifact_bindings"]
    for binding in candidate_bindings.values():
        if (bindings.get(binding["ref"]) != binding["sha256"]
                or terminal["artifact_hashes"].get(binding["ref"]) != binding["sha256"]):
            raise ValueError("candidate binding differs from immutable terminal input")
    def artifact(name):
        return load_json_object(resolve_repo_pointer(root, candidate_bindings[name]["ref"]))
    source = artifact("source_config")
    if (source["reference_bundle_ref"] != record["original_bundle_ref"]
            or source["reference_bundle_sha256"] != bindings[record["original_bundle_ref"]]):
        raise ValueError("source package froze another reference bundle")
    prediction = artifact("prediction_manifest")
    if (prediction != arm["prediction_manifest"] or artifact("paired_analysis") != arm["paired_analysis"]
            or artifact("runtime_certificate") != arm["runtime_certificate"]
            or bindings.get(prediction["artifact_locator"]) != prediction["artifact_sha256"]):
        raise ValueError("candidate endpoint is not bound to independent acceptance")
    for key in ("evaluation_role_identity", "ordering_semantics", "row_count", "source_idx_sha256", "target_sha256"):
        if prediction[key] != bundle["prediction_manifest"][key]:
            raise ValueError("candidate/reference predictions are not aligned")
    roles = artifact("role_history")
    costs = artifact("cost_records")
    for sidecar, key in ((roles, "roles"), (costs, "costs")):
        verify_bound_artifact(root, sidecar["source_ref"], sidecar["source_sha256"])
        if sidecar[key] != terminal[key]:
            raise ValueError("qualification fabricated candidate role/cost events")
    trace = load_canonical_trace(resolve_repo_pointer(root, original["trace_artifact_ref"]))
    verify_bound_artifact(root, original["trace_artifact_ref"], original["trace_artifact_sha256"])
    validate_manifest_trace(original, trace)
    completion = load("completion_ref")
    last = trace["observations"][-1]
    if (bindings[record["completion_ref"]] != arm["completion_manifest_sha256"]
            or completion["complete"] is not True
            or completion["trajectory_id"] != original["trajectory_id"]
            or completion["logical_run_id"] != original["run_id"]
            or completion["variant"] != record["acceptance_arm"]
            or completion["best_model_sha256"] != candidate_bindings["checkpoint"]["sha256"]
            or completion["development_predictions_sha256"] != prediction["artifact_sha256"]
            or completion["canonical_trace_sha256"] != original["trace_artifact_sha256"]
            or not all(original["trace_fields"].values())
            or any(row[field] is None for row in trace["observations"] for field in original["trace_fields"])
            or len(trace["observations"]) != completion["epochs"]
            or any(completion[k] is not False for k in (
                "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))
            or last["checkpoint_identity"] != "sha256:" + completion["checkpoint_sha256"]
            or any(completion[k] != original["exposure"][k] or completion[k] != identity[k]
                   or completion[k] != last[{"optimizer_steps": "optimizer_step",
                                           "sample_presentations": "sample_presentations"}[k]]
                   for k in ("optimizer_steps", "sample_presentations"))):
        raise ValueError("candidate trace/checkpoint/endpoint is incomplete")

    reference_bindings = deepcopy(old["reference_artifact_bindings"])
    # The qualification verifier proves these are representations of the SAME
    # immutable control; no candidate-side observation or identity is replaced.
    for kind in ("trace_manifest", "role_history", "cost_records", "acceptance"):
        ref = bundle[kind + "_ref"]
        reference_bindings[kind] = {"ref": ref, "sha256": ref_record["artifact_hashes"][ref]}
    for kind, binding in reference_bindings.items():
        verify_bound_artifact(root, binding["ref"], binding["sha256"])
        if kind + "_ref" in bundle and binding["ref"] != bundle[kind + "_ref"]:
            raise ValueError("reference evidence differs from its qualified bundle")
    if reference_bindings["checkpoint"]["sha256"] != bundle["checkpoint_identity"]:
        raise ValueError("reference checkpoint identity changed")

    def side(who, identity, artifact_bindings, events, fields):
        return {"comparison_identity": identity, "artifact_bindings": artifact_bindings,
            "artifacts": {k: "complete" for k in artifact_bindings}, "terminal_complete": True,
            "prediction_status": "complete", "row_alignment_status": "aligned",
            "runtime_certificate_status": "accepted", "trace_status": "complete",
            "stochasticity_status": old["stochasticity_status"],
            "role_applicability_plan": old[who + "_role_applicability_plan"],
            "observed_role_event_kinds": sorted({e["access_kind"] for e in events}),
            "trace_field_availability": fields}
    ref_roles = load_json_object(resolve_repo_pointer(root, bundle["role_history_ref"]))["roles"]
    reference = side("reference", bundle["comparison_identity"], reference_bindings,
                     ref_roles, ref_manifest["trace_fields"])
    reference.update(reference_bundle_id=bundle["reference_bundle_id"],
                     reference_bundle_sha256=reference_bundle_digest(bundle))
    readiness = assess_comparison_readiness(candidate_id=evidence["evidence_id"],
        candidate=side("candidate", identity, candidate_bindings, roles["roles"], original["trace_fields"]),
        reference_id=frozen_ref, reference=reference,
        **{k: old[k] for k in ("experiment_purpose", "declared_intervention_fields",
                              "intervention_group_id", "mechanism_id")})
    validate_comparison_readiness(readiness, evidence_verifier=lambda p, s: verify_bound_artifact(root, p, s))
    if readiness["comparison_class"] != "STRICT_CAUSAL" or not readiness["strict_ready"]:
        raise ValueError("candidate qualification did not establish strict observed comparison")
    manifest = deepcopy(original)
    manifest["backtest_eligibility"] = {"eligible": True, "exclusion_reasons": []}
    # Correct the inherited control label from accepted native candidate identity.
    manifest["model_identity"] = identity["architecture_config_identity"]
    if manifest["comparability_identity"]["architecture_identity"] != manifest["model_identity"]:
        raise ValueError("candidate architecture identity differs from trace manifest")
    return validate_trace_manifest(manifest), readiness


def verify_candidate_qualification(root: Path, pointer: str) -> tuple[dict, dict]:
    record = load_json_object(resolve_repo_pointer(root, pointer))
    manifest, readiness = assess_candidate_qualification(root, record)
    for field, expected in (("qualified_manifest_ref", manifest), ("qualified_readiness_ref", readiness)):
        ref = record[field]
        if ref not in record["artifact_hashes"] or load_json_object(resolve_repo_pointer(root, ref)) != expected:
            raise ValueError("published candidate qualification differs from recomputed evidence")
    return manifest, readiness


def qualified_candidate_record(root: Path, path: Path, record: dict, *, kind: str) -> dict:
    marker = (path.parent.parent if kind == "manifest" else path.parent.parent / "rml_plan") / "candidate_qualification.json"
    if not marker.is_file():
        return record
    pointer = marker.resolve().relative_to(Path(root).resolve()).as_posix()
    descriptor = load_json_object(marker)
    field = "original_manifest_ref" if kind == "manifest" else "original_readiness_ref"
    if descriptor[field] != path.resolve().relative_to(Path(root).resolve()).as_posix():
        raise ValueError("candidate qualification marker targets another record")
    manifest, readiness = verify_candidate_qualification(root, pointer)
    return manifest if kind == "manifest" else readiness
