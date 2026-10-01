"""Hash-bound reuse of a terminal candidate as a later comparator.

This adds a comparison-world view, not a new run, trace, cost or result.
The original candidate and its first comparison remain immutable.
"""
from copy import deepcopy
from pathlib import Path

from molgap.comparison_readiness import validate_reference_bundle
from molgap.evidence_pointers import load_json_object, validate_release_reference_bundle_evidence
from .candidate_qualification import verify_candidate_qualification
from .paths import resolve_repo_pointer, verify_bound_artifact
from .trace import atomic_write, file_digest, json_bytes


def _expected(root, record):
    if record.get("terminal_candidate_ref"):
        return _terminal_expected(root, record)
    manifest, readiness = verify_candidate_qualification(root, record["candidate_qualification_ref"])
    qualification = load_json_object(resolve_repo_pointer(root, record["candidate_qualification_ref"]))
    acceptance = load_json_object(resolve_repo_pointer(root, qualification["acceptance_ref"]))
    accepted = acceptance["arms"][qualification["acceptance_arm"]]
    if not accepted["accepted"] or not readiness["strict_ready"]:
        raise ValueError("only an independently accepted strict candidate may become a comparator")
    evidence = load_json_object(resolve_repo_pointer(root, manifest["terminal_evidence_ref"]))
    if evidence["outcome"]["execution_status"] != "complete" or evidence["outcome"]["artifact_status"] != "accepted":
        raise ValueError("candidate evidence is not terminally accepted")
    original = load_json_object(resolve_repo_pointer(root, qualification["reference_qualification_ref"]))
    bundle = deepcopy(load_json_object(resolve_repo_pointer(root, original["qualified_bundle_ref"])))
    bundle.pop("qualification_ref", None)
    bindings = readiness["candidate_artifact_bindings"]
    for field, key in (("runtime_certificate_ref", "runtime_certificate"), ("row_manifest_ref", "row_manifest"),
                       ("target_manifest_ref", "target_manifest"), ("role_history_ref", "role_history"),
                       ("target_transform_asset_ref", "target_transform_asset"), ("cost_records_ref", "cost_records"),
                       ("decision_ref", "decision")):
        bundle[field] = bindings[key]["ref"]
    bundle.update(reference_bundle_id=record["reference_bundle_id"], reference_id=evidence["evidence_id"],
        contract_ref=bindings["source_config"]["ref"], comparison_identity=accepted["comparison_identity"],
        architecture_config_identity=accepted["comparison_identity"]["architecture_config_identity"],
        checkpoint_identity=bindings["checkpoint"]["sha256"], source_commit_or_archive=acceptance["source_archive_sha256"],
        prediction_manifest=accepted["prediction_manifest"], trace_manifest_ref=record["reference_view_ref"],
        acceptance_ref=qualification["acceptance_ref"], candidate_reference_qualification_ref=record["qualification_ref"])
    view = deepcopy(manifest)
    view.update(comparison_role="reference", reference_id=evidence["evidence_id"])
    return bundle, view


def _terminal_expected(root, record):
    """Enroll an already strict terminal candidate; never qualify missing proof."""
    from .finalize import verified_receipt
    from .trace import load_canonical_trace, validate_manifest_trace
    from molgap.comparison_readiness import validate_comparison_readiness
    directory = resolve_repo_pointer(root, record["terminal_candidate_ref"]).parent
    if resolve_repo_pointer(root, record["original_manifest_ref"]) != directory / "trace_manifest.json":
        raise ValueError("terminal comparator manifest is outside its finalized transaction")
    verified_receipt(directory)
    load = lambda p: load_json_object(resolve_repo_pointer(root, p))
    trajectory, original = load(record["terminal_candidate_ref"]), load(record["original_manifest_ref"])
    evidence = load_json_object(directory / "v5_evidence.json")
    terminal = load_json_object(directory / "terminal_input.json")
    ready = load(record["terminal_readiness_ref"])
    validate_comparison_readiness(ready, evidence_verifier=lambda p, s: verify_bound_artifact(root, p, s))
    accepted = load(record["acceptance_ref"])
    arm = accepted["arms"][record["acceptance_arm"]]
    if (not ready["strict_ready"] or ready["comparison_class"] != "STRICT_CAUSAL"
            or not accepted["accepted"] or not arm["accepted"]
            or trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] == "ACTIVE"
            or evidence["outcome"]["execution_status"] != "complete" or evidence["outcome"]["artifact_status"] != "accepted"
            or ready["candidate_id"] != evidence["evidence_id"] or evidence["evidence_id"] not in trajectory["result"]["evidence_ids"]
            or terminal["comparison_readiness_ref"] != record["terminal_readiness_ref"]
            or original["trajectory_id"] != trajectory["trajectory_id"] or original["run_id"] != terminal["run_id"]
            or original["comparison_role"] != "candidate" or not original["backtest_eligibility"]["eligible"]):
        raise ValueError("terminal comparator lacks its own accepted strict identity")
    for binding in ready["candidate_artifact_bindings"].values():
        if terminal["artifact_hashes"].get(binding["ref"]) != binding["sha256"]:
            raise ValueError("terminal comparator binding differs from immutable transaction")
    if terminal["artifact_hashes"].get(record["acceptance_ref"]) != file_digest(resolve_repo_pointer(root, record["acceptance_ref"])):
        raise ValueError("independent acceptance is not frozen in terminal transaction")
    identity = dict(ready["matched_fields"])
    identity.update({k: v["candidate"] for k, v in ready["mismatched_fields"].items()})
    if identity != arm["comparison_identity"] or arm["trajectory_id"] != trajectory["trajectory_id"]:
        raise ValueError("terminal comparator scientific identity changed")
    bindings = ready["candidate_artifact_bindings"]
    prediction = load(bindings["prediction_manifest"]["ref"])
    if prediction != arm["prediction_manifest"]:
        raise ValueError("terminal comparator prediction not independently accepted")
    verify_bound_artifact(root, prediction["artifact_locator"], prediction["artifact_sha256"])
    trace = load_canonical_trace(resolve_repo_pointer(root, original["trace_artifact_ref"]))
    validate_manifest_trace(original, trace)
    verify_bound_artifact(root, original["trace_artifact_ref"], original["trace_artifact_sha256"])
    if (not all(original["trace_fields"].values())
            or trace["observations"][-1]["optimizer_step"] != identity["optimizer_steps"]
            or trace["observations"][-1]["sample_presentations"] != identity["sample_presentations"]):
        raise ValueError("terminal comparator trace is incomplete")
    source = load(bindings["source_config"]["ref"])
    bundle = deepcopy(load(source["reference_bundle_ref"]))
    if file_digest(resolve_repo_pointer(root, source["reference_bundle_ref"])) != source["reference_bundle_sha256"]:
        raise ValueError("terminal comparator original reference changed")
    validate_release_reference_bundle_evidence(bundle, repo_root=root, reference_bundle_path=root / source["reference_bundle_ref"])
    bundle.pop("qualification_ref", None)
    for field, key in (("runtime_certificate_ref", "runtime_certificate"), ("row_manifest_ref", "row_manifest"),
                       ("target_manifest_ref", "target_manifest"), ("role_history_ref", "role_history"),
                       ("target_transform_asset_ref", "target_transform_asset"), ("cost_records_ref", "cost_records"),
                       ("decision_ref", "decision")):
        bundle[field] = bindings[key]["ref"]
    bundle.update(reference_bundle_id=record["reference_bundle_id"], reference_id=evidence["evidence_id"],
        contract_ref=bindings["source_config"]["ref"], comparison_identity=identity,
        architecture_config_identity=identity["architecture_config_identity"], checkpoint_identity=bindings["checkpoint"]["sha256"],
        source_commit_or_archive=accepted["source_archive_sha256"], prediction_manifest=prediction,
        trace_manifest_ref=record["reference_view_ref"], acceptance_ref=record["acceptance_ref"],
        candidate_reference_qualification_ref=record["qualification_ref"])
    view = deepcopy(original)
    view.update(comparison_role="reference", reference_id=evidence["evidence_id"])
    return bundle, view


def verify_candidate_reference(root: Path, pointer: str, *, expected_bundle=None):
    root = Path(root).resolve()
    record = load_json_object(resolve_repo_pointer(root, pointer))
    if record.get("format") != "molgap-accepted-candidate-reference-v1" or record["qualification_ref"] != pointer:
        raise ValueError("unsupported candidate reference qualification")
    for ref, digest in record["artifact_hashes"].items():
        verify_bound_artifact(root, ref, digest)
    bundle, view = _expected(root, record)
    if (load_json_object(resolve_repo_pointer(root, record["bundle_ref"])) != bundle
            or load_json_object(resolve_repo_pointer(root, record["reference_view_ref"])) != view
            or expected_bundle is not None and dict(expected_bundle) != bundle):
        raise ValueError("candidate comparator changed accepted observed identities")
    qualification = record.get("candidate_qualification_ref", record.get("terminal_candidate_ref"))
    required = {record["bundle_ref"], record["reference_view_ref"], qualification,
        *[bundle[k] for k in bundle if k.endswith("_ref") and k != "candidate_reference_qualification_ref"],
        bundle["prediction_manifest"]["artifact_locator"]}
    ready_ref = record.get("terminal_readiness_ref") or load_json_object(resolve_repo_pointer(root, record["candidate_qualification_ref"]))["qualified_readiness_ref"]
    readiness = load_json_object(resolve_repo_pointer(root, ready_ref))
    required.add(readiness["candidate_artifact_bindings"]["checkpoint"]["ref"])
    if record.get("terminal_candidate_ref"):
        required.update((record["original_manifest_ref"], record["terminal_readiness_ref"], record["acceptance_ref"]))
    if not required <= record["artifact_hashes"].keys():
        raise ValueError("candidate reference has unbound evidence")
    validate_release_reference_bundle_evidence(bundle, repo_root=root, reference_bundle_path=root / record["bundle_ref"])
    return view


def enroll_terminal_candidate_reference(root: Path, *, finalized_ref: str, readiness_ref: str,
        acceptance_ref: str, acceptance_arm: str, destination: str, bundle_id: str):
    """Reference view of exact finalized observations; no new run/cost event."""
    root = Path(root).resolve()
    pointer = destination + "/candidate_reference_qualification.json"
    record = {"format": "molgap-accepted-candidate-reference-v1", "qualification_ref": pointer,
        "terminal_candidate_ref": finalized_ref + "/trajectory.json", "original_manifest_ref": finalized_ref + "/trace_manifest.json",
        "terminal_readiness_ref": readiness_ref, "acceptance_ref": acceptance_ref, "acceptance_arm": acceptance_arm,
        "reference_bundle_id": bundle_id, "bundle_ref": destination + "/reference_bundle.json",
        "reference_view_ref": destination + "/reference_view.json"}
    bundle, view = _expected(root, record)
    validate_reference_bundle(bundle)
    for ref, payload in ((record["bundle_ref"], bundle), (record["reference_view_ref"], view)):
        path = root / ref
        if path.exists() and path.read_bytes() != json_bytes(payload):
            raise ValueError("terminal reference enrollment is immutable")
        atomic_write(path, json_bytes(payload))
    ready = load_json_object(root / readiness_ref)
    refs = {record["bundle_ref"], record["reference_view_ref"], record["terminal_candidate_ref"], record["original_manifest_ref"],
        readiness_ref, acceptance_ref, finalized_ref + "/v5_evidence.json", finalized_ref + "/terminal_input.json",
        finalized_ref + "/finalization.json", view["trace_artifact_ref"], bundle["prediction_manifest"]["artifact_locator"],
        *[bundle[k] for k in bundle if k.endswith("_ref") and k != "candidate_reference_qualification_ref"],
        *[v["ref"] for v in ready["candidate_artifact_bindings"].values()]}
    record["artifact_hashes"] = {ref: file_digest(resolve_repo_pointer(root, ref)) for ref in sorted(refs)}
    path = root / pointer
    if path.exists() and path.read_bytes() != json_bytes(record):
        raise ValueError("terminal comparator proof is immutable")
    atomic_write(path, json_bytes(record))
    verify_candidate_reference(root, pointer, expected_bundle=bundle)
    return bundle


def enroll_candidate_reference(root: Path, qualification_ref: str, destination: str, bundle_id: str):
    root = Path(root).resolve()
    directory = root / destination
    pointer = destination + "/candidate_reference_qualification.json"
    record = {"format": "molgap-accepted-candidate-reference-v1", "qualification_ref": pointer,
        "candidate_qualification_ref": qualification_ref, "reference_bundle_id": bundle_id,
        "bundle_ref": destination + "/reference_bundle.json", "reference_view_ref": destination + "/reference_view.json"}
    bundle, view = _expected(root, record)
    validate_reference_bundle(bundle)
    for ref, payload in ((record["bundle_ref"], bundle), (record["reference_view_ref"], view)):
        path = root / ref
        if path.exists() and path.read_bytes() != json_bytes(payload):
            raise ValueError("reference enrollment refuses to overwrite different evidence")
        atomic_write(path, json_bytes(payload))
    refs = {qualification_ref, record["bundle_ref"], record["reference_view_ref"],
        *[bundle[k] for k in bundle if k.endswith("_ref") and k != "candidate_reference_qualification_ref"],
        bundle["prediction_manifest"]["artifact_locator"]}
    q = load_json_object(resolve_repo_pointer(root, qualification_ref))
    r = load_json_object(resolve_repo_pointer(root, q["qualified_readiness_ref"]))
    refs.add(r["candidate_artifact_bindings"]["checkpoint"]["ref"])
    record["artifact_hashes"] = {ref: file_digest(resolve_repo_pointer(root, ref)) for ref in sorted(refs)}
    path = directory / "candidate_reference_qualification.json"
    if path.exists() and path.read_bytes() != json_bytes(record):
        raise ValueError("reference qualification is immutable")
    atomic_write(path, json_bytes(record))
    verify_candidate_reference(root, pointer, expected_bundle=bundle)
    return bundle
