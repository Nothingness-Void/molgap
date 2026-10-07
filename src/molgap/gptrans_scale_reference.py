"""Enroll the retained 500K EMA999 control; no model execution or new claim."""
from copy import deepcopy
import hashlib
from pathlib import Path

from .evidence_pointers import load_json_object, validate_release_reference_bundle_evidence
from .gptrans_reference_qualification import qualify_reference
from .research_memory.finalize import verified_receipt
from .research_memory.paths import verify_bound_artifact
from .research_memory.trace import atomic_write, file_digest, json_bytes, load_canonical_trace

BASE = "experiments/pcqm_gptrans_local_transfer_500k"
OLD = "experiments/pcqm_gptrans_capacity_relations_100k"
RECORDS = "platforms/_records/kaggle/training/gptrans_capacity_relations_v1/gptrans_local_scale"


def enroll_scale_reference(root: Path):
    """Project accepted native metadata, preserving every finalized original byte."""
    import torch
    root = Path(root).resolve()
    load = lambda ref: load_json_object(root / ref)
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    destination = root / BASE / "reference"
    finalized = root / OLD / "gpu/scale_ema/rml_plan/rml_finalized"
    verified_receipt(finalized)
    terminal = load_json_object(finalized / "terminal_input.json")
    for ref, digest in terminal["artifact_hashes"].items():
        verify_bound_artifact(root, ref, digest)
    native_ref = OLD + "/gpu/results/scale_acceptance.json"
    accepted = load(native_ref)
    if terminal["artifact_hashes"].get(native_ref) != file_digest(root / native_ref) or accepted["accepted"] is not True:
        raise ValueError("Retained native acceptance is not immutable terminal evidence")
    result = accepted["result"]
    config_ref = OLD + "/gpu/screen_config.json"
    config = load(config_ref)
    if result["configuration"] != config["scale_study"] or result["source_identity"] != accepted["source_identity"]:
        raise ValueError("Native source/configuration binding differs")
    if (result["parameters"], result["optimizer_steps"], result["sample_presentations"]) != (5246817, 46860, 5998080):
        raise ValueError("Retained reference recipe/exposure differs")
    identity = deepcopy(config["arms"]["scale_ema"]["comparison_identity"])
    if identity["ema_decay"] != .999 or identity["physical_batch_per_device"] != 128 or identity["precision"] != "fp32":
        raise ValueError("Retained primary reference identity differs")
    folder = root / RECORDS / "scale_ema/training"
    for name, digest in result["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or file_digest(path) != digest:
            raise ValueError("Retained reference file binding differs: " + name)
    for flag in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if result[flag] is not False:
            raise ValueError("Reference consumed a protected role")
    prediction_path = folder / "ema999/development_predictions.pt"
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if not torch.equal(payload["source_idx"], torch.arange(500000, 550000)):
        raise ValueError("Reference development row identity differs")
    if not all(bool(torch.isfinite(payload[key]).all()) and len(payload[key]) == 50000 for key in ("prediction_eV", "target_eV")):
        raise ValueError("Reference predictions/targets are incomplete")
    trace = load_canonical_trace(folder / "ema999/canonical_trace.json")
    best = min(range(60), key=lambda i: trace["observations"][i]["ema_dev_metric"])
    mae = float((payload["prediction_eV"].double() - payload["target_eV"].double()).abs().mean())
    if best != result["best"]["ema999"]["rung"] or abs(mae - result["best"]["ema999"]["mae_eV"]) >= 1e-7:
        raise ValueError("Reference endpoint selector/MAE differs")
    digest_tensor = lambda name: hashlib.sha256(payload[name].contiguous().numpy().tobytes()).hexdigest()
    prediction = {"artifact_locator": rel(prediction_path), "artifact_sha256": file_digest(prediction_path),
        "row_count": 50000, "unique_source_idx": 50000, "ordering_semantics": "source_idx ascending",
        "evaluation_role_identity": identity["evaluation_role_identity"],
        "source_idx_sha256": digest_tensor("source_idx"), "target_sha256": digest_tensor("target_eV"),
        "prediction_sha256": digest_tensor("prediction_eV")}
    checkpoint = folder / "ema999/best_model.pt"
    projection = {"accepted": True, "source_acceptance_ref": native_ref, "source_acceptance_sha256": file_digest(root / native_ref),
        "best_ema_model_sha256": file_digest(checkpoint), "development_predictions_sha256": file_digest(prediction_path),
        "optimizer_steps": 46860, "sample_presentations": 5998080, "trace_observations": len(trace["observations"]),
        "final_checkpoint_sha256": file_digest(folder / "last_checkpoint.pt"), "protected_roles_read": False,
        "comparison_identity": identity, "model_inference_executed": False, "local_training_executed": False,
        "scientific_promotion": False, "original_comparison_class": accepted["comparison_class"]}
    metadata = {"roles": terminal["roles"], "costs": terminal["costs"],
        "native_cost_sources": [{"ref": RECORDS + "/native_cost.json", "sha256": file_digest(root / RECORDS / "native_cost.json")}]}
    row = {"evaluation_role_identity": identity["evaluation_role_identity"], "row_count": 50000,
        "source_idx_sha256": prediction["source_idx_sha256"], "ordering_semantics": prediction["ordering_semantics"],
        "source_prediction_ref": prediction["artifact_locator"], "source_prediction_sha256": prediction["artifact_sha256"]}
    target = {"target_identity": identity["target_identity"], "unit": "eV", "target_sha256": prediction["target_sha256"],
        "source_prediction_ref": prediction["artifact_locator"], "source_prediction_sha256": prediction["artifact_sha256"]}
    ref = lambda name: BASE + "/reference/" + name
    # The original control bundle is a new binding, not a rewrite of old outcomes.
    bundle = deepcopy(load(OLD + "/reference/reference_bundle.json"))
    bundle.pop("candidate_reference_qualification_ref", None)
    bundle.update(reference_bundle_id="reference-gptrans-g1-equal-update500k-s42-v1",
        reference_id="pcqm-gptrans-g1-scale-ema-equal-updates-500k-s42", comparison_identity=identity,
        architecture_config_identity=identity["architecture_config_identity"], checkpoint_identity=file_digest(checkpoint),
        contract_ref=OLD + "/gpu/scale_ema/contract.json", prediction_manifest=prediction,
        runtime_certificate_ref=RECORDS + "/scale_ema/training/qualification.json",
        row_manifest_ref=ref("row_manifest.json"), target_manifest_ref=ref("target_manifest.json"),
        role_history_ref=ref("observed_metadata.json"), cost_records_ref=ref("native_cost_binding.json"),
        acceptance_ref=ref("acceptance_projection.json"), decision_ref=OLD + "/gpu/scale_ema/results/decision.md",
        trace_manifest_ref=OLD + "/gpu/scale_ema/rml_plan/rml_finalized/trace_manifest.json",
        source_commit_or_archive=result["source_identity"]["archive_sha256"])
    objects = {"prediction_manifest.json": prediction, "acceptance_projection.json": projection,
        "observed_metadata.json": metadata, "row_manifest.json": row, "target_manifest.json": target,
        "native_cost_binding.json": {"costs": terminal["costs"], "native_cost_ref": RECORDS + "/native_cost.json"},
        "reference_bundle.json": bundle}
    for name, value in objects.items():
        path = destination / name
        if path.exists() and path.read_bytes() != json_bytes(value):
            raise ValueError("Reference enrollment refuses to overwrite different bytes")
        atomic_write(path, json_bytes(value))
    qualified = qualify_reference(root, base_ref=BASE + "/reference", finalized_ref=rel(finalized),
        original_bundle_ref=ref("reference_bundle.json"), prediction_ref=ref("prediction_manifest.json"),
        acceptance_ref=ref("acceptance_projection.json"), terminal_control=True)
    final_path = root / qualified["reference_bundle_ref"]
    final_bundle = load_json_object(final_path)
    validate_release_reference_bundle_evidence(final_bundle, repo_root=root, reference_bundle_path=final_path)
    return qualified
