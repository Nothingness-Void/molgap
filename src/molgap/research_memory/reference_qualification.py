"""Verify additive reference enrollment without rewriting finalized evidence."""
from pathlib import Path

from molgap.comparison_readiness import validate_reference_bundle, validate_target_transform_asset
from molgap.evidence_pointers import load_json_object
from .paths import resolve_repo_pointer, verify_bound_artifact
from .schemas import validate_trace_manifest
from .trace import load_canonical_trace, validate_manifest_trace


def verify_reference_qualification(root: Path, pointer: str, *, expected_bundle=None,
                                   expected_bundle_path=None) -> dict:
    """Admission is recovery of an accepted control, never a new causal claim."""
    from .finalize import verified_receipt

    root = Path(root).resolve()
    path = resolve_repo_pointer(root, pointer)
    if path is None:
        raise ValueError("reference qualification must be locally retained")
    record = load_json_object(path)
    control_enrollment = record.get("format") == "molgap-terminal-control-enrollment-v1"
    if record.get("format") != "molgap-reference-qualification-v1" and not control_enrollment:
        raise ValueError("unsupported reference qualification")
    bindings = record["artifact_hashes"]
    for ref, digest in bindings.items():
        verify_bound_artifact(root, ref, digest)

    def load(field):
        ref = record[field]
        if ref not in bindings:
            raise ValueError(f"qualification has unbound {field}")
        local = resolve_repo_pointer(root, ref)
        if local is None:
            raise ValueError("qualification cannot trust remote-only evidence")
        return load_json_object(local)

    original = load("original_manifest_ref")
    manifest = validate_trace_manifest(load("qualified_manifest_ref"))
    old_bundle = validate_reference_bundle(load("original_bundle_ref"))
    bundle = validate_reference_bundle(load("qualified_bundle_ref"))
    if expected_bundle is not None and bundle != dict(expected_bundle):
        raise ValueError("qualification does not bind the supplied reference bundle")
    if expected_bundle_path is not None and resolve_repo_pointer(root, record["qualified_bundle_ref"]) != (
            root / expected_bundle_path).resolve():
        raise ValueError("qualification does not bind the supplied bundle path")
    trajectory = load("trajectory_ref")
    directory = resolve_repo_pointer(root, record["trajectory_ref"]).parent
    verified_receipt(directory)
    if record["original_manifest_ref"] != (directory / "trace_manifest.json").relative_to(root).as_posix():
        raise ValueError("qualification manifest is not from accepted transaction")
    evidence = load_json_object(directory / "v5_evidence.json")
    terminal = load_json_object(directory / "terminal_input.json")
    if (original["comparison_role"] not in ({"reference", "candidate"} if control_enrollment else {"reference"})
            or manifest["comparison_role"] != "reference"
            or trajectory["decision"]["outcome"] == "ACTIVE"
            or trajectory["record_mode"] != "prospective"
            or evidence["outcome"]["execution_status"] != "complete"
            or evidence["outcome"]["artifact_status"] != "accepted"
            or manifest["reference_id"] != evidence["evidence_id"]
            or evidence["evidence_id"] not in trajectory["result"]["evidence_ids"]
            or (directory / "v5_evidence.json").relative_to(root).as_posix()
                not in trajectory["result"]["evidence_refs"]
            or (manifest["trajectory_id"], manifest["run_id"]) != (
                trajectory["trajectory_id"], terminal["run_id"])):
        raise ValueError("qualification requires the control's own accepted terminal identity")
    mutable = {"reference_id", "backtest_eligibility"}
    if control_enrollment:
        # A complete observed control need not have won an earlier comparison.
        # This view changes custody in a later comparison, never the old claim.
        if (record.get("scientific_promotion") is not False
                or record.get("original_comparison_status") != evidence["outcome"]["comparison_status"]):
            raise ValueError("control enrollment must preserve the original scientific claim")
        mutable.add("comparison_role")
    if {k: v for k, v in original.items() if k not in mutable} != {
            k: v for k, v in manifest.items() if k not in mutable}:
        raise ValueError("reference enrollment changed observed trace semantics")
    if manifest["backtest_eligibility"] != {"eligible": True, "exclusion_reasons": []}:
        raise ValueError("qualified reference eligibility is incomplete")
    mutable_bundle = {"reference_bundle_id", "trace_manifest_ref", "role_history_ref",
                      "cost_records_ref", "acceptance_ref", "qualification_ref"}
    if {k: v for k, v in old_bundle.items() if k not in mutable_bundle} != {
            k: v for k, v in bundle.items() if k not in mutable_bundle}:
        raise ValueError("qualification changed frozen reference scientific identity")
    if (bundle["reference_bundle_id"] == old_bundle["reference_bundle_id"]
            or bundle["reference_id"] != manifest["reference_id"]
            or bundle["trace_manifest_ref"] != record["qualified_manifest_ref"]
            or bundle.get("qualification_ref") != pointer):
        raise ValueError("qualified bundle is not independently bound")
    for field, key in (("role_history_ref", "roles"), ("cost_records_ref", "costs")):
        ref = bundle[field]
        if ref not in bindings:
            raise ValueError("qualification event sidecar is unbound")
        sidecar = load_json_object(resolve_repo_pointer(root, ref))
        source = sidecar["source_ref"]
        verify_bound_artifact(root, source, sidecar["source_sha256"])
        if (sidecar[key] != terminal[key]
                or sidecar[key] != load_json_object(resolve_repo_pointer(root, source))[key]):
            raise ValueError("qualification fabricated observed events")
    pointers = [bundle[k] for k in bundle if k.endswith("_ref") and k != "qualification_ref"]
    if len(pointers) != len(set(pointers)):
        raise ValueError("qualified reference requires distinct evidence pointers")
    for ref in pointers:
        if ref not in bindings:
            raise ValueError("qualified reference pointer lacks bound evidence")
    transform = validate_target_transform_asset(load_json_object(resolve_repo_pointer(root, bundle["target_transform_asset_ref"])))
    identity = bundle["comparison_identity"]
    if (identity["target_transform_identity"], identity["target_transform_asset_sha256"], identity["target_identity"]) != (
            transform["asset_id"], transform["asset_sha256"], transform["target_identity"]):
        raise ValueError("qualified target transform identity changed")
    acceptance = load_json_object(resolve_repo_pointer(root, bundle["acceptance_ref"]))
    if control_enrollment:
        source = acceptance["source_acceptance_ref"]
        digest = acceptance["source_acceptance_sha256"]
        if terminal["artifact_hashes"].get(source) != digest or bindings.get(source) != digest:
            raise ValueError("control acceptance is not bound to the original terminal transaction")
        verify_bound_artifact(root, source, digest)
        native = load_json_object(resolve_repo_pointer(root, source))
        if native.get("accepted") is not True:
            raise ValueError("control lacked independently accepted native artifacts")
        # This enrollment format currently supports the retained shared-live
        # scale producer only. Do not trust a newly written acceptance boolean.
        if native.get("format") != "molgap-scale-ema-acceptance-v1":
            raise ValueError("unsupported terminal control producer")
        result = native["result"]
        projected = {"best_ema_model_sha256": result["files"]["ema999/best_model.pt"],
            "development_predictions_sha256": result["files"]["ema999/development_predictions.pt"],
            "final_checkpoint_sha256": result["files"]["last_checkpoint.pt"],
            "optimizer_steps": result["optimizer_steps"], "sample_presentations": result["sample_presentations"]}
        if (any(acceptance[key] != value for key, value in projected.items())
                or result["complete"] is not True or result["live_optimizer_streams"] != 1
                or (result["optimizer_steps"], result["sample_presentations"], result["parameters"]) != (46860, 5998080, 5246817)
                or any(result[key] is not False for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))
                or identity["ema_decay"] != .999
                or identity["optimizer_steps"] != result["optimizer_steps"]
                or identity["sample_presentations"] != result["sample_presentations"]):
            raise ValueError("control projection differs from accepted primary native evidence")
    prediction = load("prediction_manifest_ref")
    if (acceptance.get("accepted") is not True
            or acceptance["best_ema_model_sha256"] != bundle["checkpoint_identity"]
            or acceptance["development_predictions_sha256"] != prediction["artifact_sha256"]
            or any(prediction[k] != v for k, v in bundle["prediction_manifest"].items())
            or bindings.get(record["checkpoint_ref"]) != bundle["checkpoint_identity"]
            or bindings.get(prediction["artifact_locator"]) != prediction["artifact_sha256"]):
        raise ValueError("qualified reference lacks accepted checkpoint/prediction bindings")
    trace = load_canonical_trace(resolve_repo_pointer(root, manifest["trace_artifact_ref"]))
    validate_manifest_trace(manifest, trace)
    last = trace["observations"][-1]
    # This producer recorded checkpoints, not a synthetic terminal trace event.
    # Completion comes from accepted counters plus the existing final transaction.
    if (not all(manifest["trace_fields"].values())
            or last["optimizer_step"] != manifest["exposure"]["optimizer_steps"]
            or last["sample_presentations"] != manifest["exposure"]["sample_presentations"]
            or last["optimizer_step"] != acceptance["optimizer_steps"]
            or last["sample_presentations"] != acceptance["sample_presentations"]
            or len(trace["observations"]) != acceptance["trace_observations"]
            or last["checkpoint_identity"] != "sha256:" + acceptance["final_checkpoint_sha256"]
            or acceptance["protected_roles_read"] is not False):
        raise ValueError("qualified reference trace is not complete")
    return manifest


def qualified_reference_manifest(root: Path, path: Path, manifest: dict) -> dict:
    """The marker lives outside the immutable finalization inventory."""
    marker = path.parent.parent / "reference_qualification.json"
    if not marker.is_file():
        return manifest
    pointer = marker.resolve().relative_to(Path(root).resolve()).as_posix()
    record = load_json_object(marker)
    if manifest["comparison_role"] != "reference" and record.get("format") != "molgap-terminal-control-enrollment-v1":
        return manifest
    if record.get("original_manifest_ref") != path.resolve().relative_to(Path(root).resolve()).as_posix():
        raise ValueError("reference qualification marker targets another manifest")
    return verify_reference_qualification(root, pointer)
