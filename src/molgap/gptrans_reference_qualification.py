"""Recover GPTrans reference bindings exclusively from retained accepted metadata."""
from copy import deepcopy
from pathlib import Path

from .evidence_pointers import load_json_object, resolve_repo_pointer
from .research_memory.finalize import verified_receipt
from .research_memory.reference_qualification import verify_reference_qualification
from .research_memory.trace import atomic_write, file_digest, json_bytes


def qualify_reference(repo_root: Path) -> dict:
    root = Path(repo_root).resolve()
    base = root / "experiments/pcqm_gptrans_v5_audit_reference"
    finalized = base / "rml_plan/rml_finalized"
    verified_receipt(finalized)
    destination = base / "results/reference_qualification"
    marker = base / "rml_plan/reference_qualification.json"
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    if marker.exists():
        verify_reference_qualification(root, rel(marker))
        return {"status": "ALREADY_QUALIFIED", "qualification_ref": rel(marker)}
    original_bundle_path = base / "results/terminal/reference_bundle.json"
    old_bundle = load_json_object(original_bundle_path)
    original_manifest_path = finalized / "trace_manifest.json"
    manifest = load_json_object(original_manifest_path)
    trajectory = load_json_object(finalized / "trajectory.json")
    evidence = load_json_object(finalized / "v5_evidence.json")
    terminal = load_json_object(finalized / "terminal_input.json")
    metadata = resolve_repo_pointer(root, old_bundle["role_history_ref"])
    observed = load_json_object(metadata)
    if observed["roles"] != terminal["roles"] or observed["costs"] != terminal["costs"]:
        raise ValueError("observed events differ from immutable reference transaction")
    hashes = {ref: digest for ref, digest in terminal["artifact_hashes"].items()}
    # Validate the real original transaction inputs, including native costs and outputs.
    from .research_memory.paths import verify_bound_artifact
    for ref, digest in hashes.items():
        verify_bound_artifact(root, ref, digest)
    for item in observed["native_cost_sources"]:
        verify_bound_artifact(root, item["ref"], item["sha256"])
    prediction_path = base / "results/terminal/prediction_manifest.json"
    prediction = load_json_object(prediction_path)
    checkpoint = resolve_repo_pointer(root, prediction["artifact_locator"]).parent / "best_model.pt"
    if file_digest(checkpoint) != old_bundle["checkpoint_identity"]:
        raise ValueError("reference checkpoint differs from accepted bundle")
    manifest["reference_id"] = evidence["evidence_id"]
    manifest["backtest_eligibility"] = {"eligible": True, "exclusion_reasons": []}
    bundle = deepcopy(old_bundle)
    bundle.update(reference_bundle_id=old_bundle["reference_bundle_id"] + "-qualified-v1",
                  trace_manifest_ref=rel(destination / "qualified_trace_manifest.json"),
                  role_history_ref=rel(destination / "role_history.json"),
                  cost_records_ref=rel(destination / "cost_records.json"),
                  acceptance_ref=rel(base / "results/final_acceptance.json"),
                  qualification_ref=rel(marker))
    outputs = {destination / "reference_bundle.json": bundle,
               destination / "qualified_trace_manifest.json": manifest}
    for name, key in (("role_history", "roles"), ("cost_records", "costs")):
        outputs[destination / (name + ".json")] = {
            key: observed[key], "source_ref": rel(metadata), "source_sha256": file_digest(metadata),
            "semantics": "exact existing accepted events; no new access or cost"}
    for path, value in outputs.items():
        if path.exists() and path.read_bytes() != json_bytes(value):
            raise ValueError("qualification refuses to overwrite different evidence")
        atomic_write(path, json_bytes(value))
    pointers = [bundle[k] for k in bundle if k.endswith("_ref") and k != "qualification_ref"]
    paths = [original_bundle_path, original_manifest_path, finalized / "trajectory.json",
             finalized / "finalization.json", finalized / "v5_evidence.json", prediction_path, checkpoint,
             *outputs, *(resolve_repo_pointer(root, ref) for ref in pointers)]
    hashes.update({rel(path): file_digest(path) for path in paths})
    descriptor = {
        "format": "molgap-reference-qualification-v1",
        "original_bundle_ref": rel(original_bundle_path),
        "original_manifest_ref": rel(original_manifest_path),
        "trajectory_ref": rel(finalized / "trajectory.json"),
        "qualified_bundle_ref": rel(destination / "reference_bundle.json"),
        "qualified_manifest_ref": rel(destination / "qualified_trace_manifest.json"),
        "prediction_manifest_ref": rel(prediction_path), "checkpoint_ref": rel(checkpoint),
        "artifact_hashes": hashes,
        "semantics": "Additive control enrollment; original scientific outcomes and prospective bytes unchanged",
        "local_training_executed": False, "model_inference_executed": False,
        "successor_authorized": False}
    atomic_write(marker, json_bytes(descriptor))
    verify_reference_qualification(root, rel(marker))
    return {"status": "QUALIFIED", "qualification_ref": rel(marker),
            "reference_bundle_ref": rel(destination / "reference_bundle.json"),
            "trajectory_id": trajectory["trajectory_id"], "reference_id": bundle["reference_id"]}
