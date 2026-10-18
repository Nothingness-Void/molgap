"""Prepare V4 qualification closeout from pinned metadata; never publish or train.

Call ``prepare_qualification_terminal(repo_root=..., output=..., finalized_at=...)``.
For each returned arm, the controller may separately call
``research_memory.finalize.finalize(repo_root, arm['trajectory'], arm['terminal'])``
without a trace. Preparation neither finalizes nor rebuilds canonical RML.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .common.wire import canonical_fingerprint
from .research_memory.paths import repo_local_path
from .research_memory.roles import validate_observed_role_truth
from .research_memory.schemas import validate_cost_event, validate_role_event, validate_trajectory
from .research_memory.trace import atomic_write, json_bytes
from .v5_common import validate_v5_evidence_envelope

BASE = "experiments/modern2d_model_integration/native_preflight/retry_v4"
INSPECTION = Path("E:/MolGap/artifacts/modern2d-native-preflight/retry_v4/inspection_20261017T223429Z")
ARMS = {"ppgt_author_v2": ("ppgt_small", "2"), "esa_native_v3": ("esa_small", "3")}
RUN = "modern2d-native-100k-s42-v4"
KERNEL = "kaseichou/molgap-modern2d-native-100k-v4"
SOURCE = "24419d42e49fe3aa1852674c2463bf58a77463a3"
SPEC = "a207acd2e09b18f5f04bb310154a58e7799eb4c4f5948397289c2c469b2dbe23"
LAUNCH = "e12c3817624d1ec1f43533dd273abbd0c38c8bb01b4b696d1e5399d6167035b2"
_INPUT_PINS = {
    "platform_observation.json": "336e04d694023e7b792812f83b52aee654a7cd43533c0809a57e4526aff84b28",
    "outputs/qualification/qualification_manifest.json": "d2d752da084b3dfe5a2378aa1a5dda799fed0b960b86b39a5522ed25eb37d33f",
    "checkpoint_hash_acceptance.json": "8ae59232513854f8f691c8fd16b67f48bd16c226d3bc8aacdf2c21734e126000",
    "luna_states/checkpoint_verification.json": "87b254c73fba3ce1b169a7f05b4081acd0cc4db541d787db5dd923d5578dff95",
    "scratchinspection/luna_metadata/native_certificate_audit.json": "6ef178b713b9bba59788f9d420d1895b60c1710818a294cff7cdbe6b3b9f15b1",
    "outputs/bootstrap_allocation.json": "e987e7298af71047b8ea961e05126295765210f4530f9cbab39ba818007af351",
}
_REPO_PINS = {
    f"{BASE}/authority.json": "973613feb8d9ee70b96ca6d2c7b8e2fd9b18a4eb36c6fcc31f680ae0ecf02a63",
    f"{BASE}/role_plan.json": "d5b1d4c5d9f00c5182a398b980d3d27a80bacc8a734e6cce85805e5048ccc39c",
    f"{BASE}/ppgt_author_v2/rml/trajectory.json": "2b50fdf0d1424619a04e8b27c1e8bfb89c950cd54fd455c9a3e831b2c095d64c",
    f"{BASE}/esa_native_v3/rml/trajectory.json": "c8d5e61f1860bcfcf296b4cbc58b984241cb507a698e287452b7adfd629fc862",
    f"{BASE}/ppgt_author_v2/contract.json": "2975c12beca40db56c117f16243b361285d0a1f1b69cd638992332b1f58eec37",
    f"{BASE}/esa_native_v3/contract.json": "cd86787955af9b6f7f4fbf85dc9ed6c8b97d02c1b66fe83ec5b93d3f59c0ef46",
}


def _match(record, expected, label):
    if any(record.get(k) != v or type(record.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError(f"{label}: binding or scope mismatch")


def _read(root, relative, digest, retained):
    path = repo_local_path(root, relative)
    if path.suffix != ".json" or path.stat().st_size > 1024**2:
        raise ValueError("Only bounded JSON metadata may be read")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"SHA256 mismatch: {relative}")
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError(f"Expected metadata object: {relative}")
    retained[relative] = payload
    return record


def _check_states(certificate, verified, manifest_files, transport, arm):
    proofs = certificate["proofs"]
    _match(certificate["resume_origin"], {"path": "resumed-first.pt"}, "interruption path")
    if {p["path"] for p in proofs} != {"repeat-a.pt", "repeat-b.pt", "resumed.pt"} or len(proofs) != 3:
        raise ValueError("Exactly three independent qualification paths required")
    if set(verified["proofs"]) != {p["path"] for p in proofs}:
        raise ValueError("Independent checkpoint proof inventory mismatch")
    fingerprints = set()
    for proof in [*proofs, certificate["resume_origin"]]:
        name = proof["path"]
        origin = name == "resumed-first.pt"
        state = verified["resume_origin"] if origin else verified["proofs"][name]
        pointer = f"{arm}/scratch/{name}"
        digest = manifest_files[pointer]["sha256"]
        if proof["sha256"] != digest or transport.get(f"qualification/{pointer}") != digest:
            raise ValueError("Independent checkpoint byte binding mismatch")
        _match(state, {"status": "paused" if origin else "complete",
            "optimizer_step": 1 if origin else 2, "sample_presentations": 128 if origin else 256,
            "next_batch": 1 if origin else 2, "optimizer_step_values": [1 if origin else 2],
            "development": False, "ema": False, "cuda_rng_count": 1}, "checkpoint state")
        if not origin:
            if state["state_fingerprint"] != proof["state_fingerprint"]:
                raise ValueError("Independent checkpoint fingerprint mismatch")
            if state["model_state_sha256"] == verified["initial_state"]["model_state_sha256"]:
                raise ValueError("Qualification endpoint echoes initialization")
            fingerprints.add(state["state_fingerprint"])
    if len(fingerprints) != 1:
        raise ValueError("Repeat/resume states differ")


def prepare_qualification_terminal(*, repo_root, output, finalized_at, inspection_root=INSPECTION):
    """Write only a new root-confined retained package, or return an identical one.

    Fixed pins select kernel139279865/version1 and its original V4 prospectives.
    No .pt is opened, even for hashing. The independent verifier's pinned report
    is consumed as evidence; this function does not repeat checkpoint validation.
    Costs stay unknown, scratch updates are not scientific training, and this
    closeout releases neither training nor Replay. Use a stable UTC timestamp.
    """
    if not isinstance(finalized_at, str) or datetime.fromisoformat(finalized_at.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("Explicit timezone-aware finalized_at required")
    root, inspection = Path(repo_root).resolve(), Path(inspection_root).resolve()
    destination = repo_local_path(root, output)
    destination.relative_to(root / "experiments")
    if any(part in {"rml", "rml_finalized"} for part in destination.relative_to(root).parts):
        raise ValueError("Preparation output must be separate from canonical RML")
    prefix = destination.relative_to(root).as_posix()
    raw, repo_bytes = {}, {}
    inputs = {ref: _read(inspection, ref, sha, raw) for ref, sha in _INPUT_PINS.items()}
    repo = {ref: _read(root, ref, sha, repo_bytes) for ref, sha in _REPO_PINS.items()}
    observation = inputs["platform_observation.json"]
    _match(observation, {"kernel": KERNEL, "kernel_id": 139279865, "version": 1,
        "status": "KernelWorkerStatus.COMPLETE", "failure_message": "", "private": True,
        "entry_bytes_equal": True, "machine_shape": "NvidiaTeslaT4"}, "platform observation")
    manifest = inputs["outputs/qualification/qualification_manifest.json"]
    _match(manifest, {"format": "molgap-modern2d-qualification-output-v1", "status": "complete",
        "scope": "QUALIFICATION_ONLY", "native_execution_released": False,
        "scientific_acceptance": "NOT_EVALUATED", "spec_identity": SPEC, "launch_sha256": LAUNCH}, "manifest")
    files = {item["path"]: item for item in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("Duplicate manifest paths")
    # Only explicitly selected small metadata enters the retained terminal package.
    selected = ["execution_report.json", "allocation_ledger.json"]
    selected += [f"{arm}/scratch/{name}.json" for arm in ARMS for name in
                 ("runtime_certificate", "runtime_provenance", "resource_qualification", "worker_result")]
    saved = {}
    for name in selected:
        ref = f"outputs/qualification/{name}"
        saved[name] = _read(inspection, ref, files[name]["sha256"], raw)
        if len(raw[ref]) != files[name]["bytes"]:
            raise ValueError("Manifest size mismatch")
    execution = saved["execution_report.json"]
    _match(execution, {"status": "complete", "phase": "sealed", "formal_training_exposure": 0,
        "native_execution_released": False, "scientific_acceptance": "NOT_EVALUATED",
        "spec_identity": SPEC, "launch_sha256": LAUNCH}, "execution")
    if set(execution["workers"]) != set(ARMS):
        raise ValueError("Unexpected qualification workers")
    _match(saved["allocation_ledger.json"], {"status": "complete", "spec_identity": SPEC,
        "allocation_device_count": 2}, "allocation observation")
    _match(inputs["outputs/bootstrap_allocation.json"], {"status": "complete", "phase": "runtime_returned",
        "full_allocation_seconds": None, "full_allocation_device_seconds": None}, "bootstrap observation")
    audit = inputs["scratchinspection/luna_metadata/native_certificate_audit.json"]
    _match(audit, {"result": "PASS_METADATA_BINDING_PENDING_FULL_COST"}, "metadata audit")
    _match(audit["run"], {"kernel": KERNEL, "kernel_id": 139279865, "version": 1,
        "status": "KernelWorkerStatus.COMPLETE"}, "audited run")
    _match(audit["checks"]["data_version_roles"], {"status": "PASS", "opened_roles": ["train_100k"],
        "common_dev_permit_frozen_not_opened_for_metrics": True, "development_rows": 0,
        "development_metrics": False, "head_to_head_comparison": False}, "role access")
    _match(audit["cost"], {"status": "INSUFFICIENT_EVIDENCE", "native_cost_fully_accepted": False,
        "overlap_not_additive": True, "full_allocation_seconds": None,
        "full_allocation_device_seconds": None}, "full cost")
    verification = inputs["luna_states/checkpoint_verification.json"]
    _match(verification, {"status": "PASS", "cuda_initialized_before_load": False,
        "cuda_initialized_after_load": False}, "independent verification")
    if set(verification["arms"]) != set(ARMS):
        raise ValueError("Independent verification arm inventory mismatch")
    transport_record = inputs["checkpoint_hash_acceptance.json"]
    _match(transport_record, {"kernel": KERNEL, "receipt_version": 1,
        "platform_version_verified_by_output_api": True}, "checkpoint transport")
    transport = {item["path"]: item["sha256"] for item in transport_record["files"] if item["state"] == "RETAINED"}
    _match(repo[f"{BASE}/role_plan.json"], {"roles_opened": ["train"], "development_metrics": False}, "role plan")
    pending = {f"{prefix}/metadata/{ref}": payload for ref, payload in raw.items()}
    bound = {ref: hashlib.sha256(payload).hexdigest() for ref, payload in {**repo_bytes, **pending}.items()}
    result = {"status": "PREPARED", "arms": {}}
    for device, (arm, interface) in enumerate(ARMS.items()):
        trajectory_ref = f"{BASE}/{arm}/rml/trajectory.json"
        frozen = repo[trajectory_ref]
        validate_trajectory(frozen)
        _match(frozen, {"owner": "server", "record_mode": "prospective",
            "trajectory_id": f"TC-modern2d-qualification-{arm}-s42-v4"}, "prospective")
        _match(frozen["state_at_start"], {"source_commit": SOURCE, "reference_ids": []}, "frozen source")
        _match(frozen["decision"], {"outcome": "ACTIVE"}, "prospective decision")
        if len(frozen["actions"]) != 1:
            raise ValueError("One frozen qualification action required")
        action = frozen["actions"][0]
        run_id = f"{RUN}:{arm}:downstream"
        _match(action, {"action_id": "A001", "run_ids": [run_id], "type": "execution_diagnostic"}, "frozen run")
        contract = repo[f"{BASE}/{arm}/contract.json"]
        _match(contract, {"family": interface[0], "family_version": interface[1], "seed": 42,
            "batch_size": 128, "train_rows": 100000, "development_rows": 0, "optimizer_steps": 2}, "recipe")
        _match(contract["dataset"], {"development": False}, "recipe roles")
        certificate, provenance, resources, worker = [saved[f"{arm}/scratch/{n}.json"] for n in
            ("runtime_certificate", "runtime_provenance", "resource_qualification", "worker_result")]
        _match(execution["workers"][arm], {"device": device, "returncode": 0, "status": "QUALIFICATION_VERIFIED"}, "worker exit")
        _match(certificate, {"status": "QUALIFICATION_VERIFIED", "scope": "QUALIFICATION_ONLY",
            "formal_training_exposure": 0, "scratch_optimizer_updates": 6,
            "native_execution_released": False, "scientific_acceptance": "NOT_EVALUATED",
            "provenance_sha256": canonical_fingerprint(provenance),
            "resource_qualification_sha256": files[f"{arm}/scratch/resource_qualification.json"]["sha256"]}, "certificate")
        context = provenance["context"]
        _match(context, {"logical_run_id": RUN, "arm_id": arm, "family_name": interface[0],
            "family_version": interface[1], "source_commit": SOURCE, "spec_identity": SPEC,
            "arm_identity": frozen["state_at_start"]["source_config_identity"],
            "run_reference": KERNEL, "platform_version": None, "account": "kaseichou",
            "source_archive_sha256": audit["source_pins"]["source_archive_sha256"],
            "package_identity": audit["source_pins"]["package_identity"],
            "training_recipe_sha256": _REPO_PINS[f"{BASE}/{arm}/contract.json"]}, "run context")
        q = provenance["qualification_context"]
        identity = canonical_fingerprint(q)
        _match(q, {"run_context": context, "recipe_identity": provenance["recipe_identity"]}, "qualification context")
        _match(q["train_permit"], {"role": "train_100k"}, "train permit")
        _match(q["development_permit"], {"role": "common_dev",
            "version_identity": q["train_permit"]["version_identity"]}, "development permit")
        _match(audit["arms"][arm], {"context_identity": identity, "binding": "PASS",
            "physical_batch_size": 128, "seed": 42, "scratch_optimizer_updates": 6,
            "formal_training_exposure": 0}, "audited arm")
        _match(certificate, {"qualification_context_identity": identity,
            "probe_recipe_identity": provenance["recipe_identity"]}, "recipe identity")
        _match(worker, {"status": "QUALIFICATION_VERIFIED", "device": device, "launch_sha256": LAUNCH,
            "context_identity": identity, "native_execution_released": False,
            "certificate_sha256": files[f"{arm}/scratch/runtime_certificate.json"]["sha256"]}, "worker binding")
        _match(resources, {"scope": "QUALIFICATION_ONLY", "physical_batch_size": 128,
            "scratch_optimizer_updates": 6, "scratch_sample_presentations": 768,
            "allocated_device_seconds": None}, "resources")
        _match(verification["arms"][arm]["initial_state"], {
            "model_state_sha256": provenance["initialization"]["state_sha256"]}, "verified initialization")
        _check_states(certificate, verification["arms"][arm], files, transport, arm)
        arm_prefix = f"{prefix}/{arm}"
        acceptance_ref, decision_ref = f"{arm_prefix}/acceptance.json", f"{arm_prefix}/decision.json"
        tid = frozen["trajectory_id"]
        role_use = {r: "untouched" for r in ("common_dev", "audit", "official_validation", "test_dev", "test_challenge")}
        role_use["train_100k"] = "consumed"
        roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-scratch-{kind}",
            "trajectory_id": tid, "action_id": "A001", "run_id": run_id,
            "dataset_identity": q["train_permit"]["version_identity"],
            "row_manifest_hash": q["train_permit"]["manifest_sha256"], "role_name": "train_100k",
            "access_kind": kind, "selection_used": False, "evidence_ref": acceptance_ref}
            for kind in ("prediction_input", "labels_read")]
        costs = [{"schema": "molgap-cost-event-v1", "cost_event_id": f"cost-{tid}-full-missing",
            "trajectory_id": tid, "action_id": "A001", "run_id": run_id, "attempt_id": f"{arm}-v4",
            "platform": "kaggle2", "hardware": "Tesla_T4_16GB", "category": "preflight",
            "measurement": {unit: {"status": "measurement_missing", "value": None}
                for unit in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")},
            "evidence_ref": acceptance_ref}]
        for event in roles:
            validate_role_event(event)
        validate_cost_event(costs[0])
        validate_observed_role_truth([role_use], roles)
        outcome = {"execution_status": "complete", "artifact_status": "accepted",
            "comparison_status": "context_only", "scientific_status": "NO_TRAIN",
            "transfer_status": "not_evaluated", "budget_decision": "measurement_missing",
            "full_handoff_status": "not_authorized"}
        decision = {"decision_ref": decision_ref, "outcome": "NO_TRAIN", "final": True,
            "next_allowed_actions": [], "reopen_conditions": ["separate prospective authority"]}
        facts = {"native_training_release": False, "native_execution_released": False,
            "full_cost_status": "measurement_missing",
            "noReplay": True, "formal_training_exposure": 0, "scratch_optimizer_updates": 6,
            "scientific_acceptance": "NOT_EVALUATED", "roles_opened": ["train"],
            "common_dev_permit_is_access": False, "protected_roles_opened": False,
            "actual_run": {"kernel": KERNEL, "kernel_id": 139279865, "version": 1, "arm_id": arm},
            "cost_scope": "One shared T4x2 allocation; retained windows overlap and are not additive. Full billing unknown.",
            "exposure_scope": "Three independent two-update scratch paths; no full100K training exposure or model-quality claim."}
        pending[decision_ref] = json_bytes({**decision, **facts})
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "legacy_contract": "none-prospective-v5", "evidence_id": f"modern2d-qualification-{arm}-s42-v4",
            "track": frozen["track"], "scope": "QUALIFICATION_ONLY_NO_TRAIN", "outcome": outcome,
            "role_use": role_use, "authority": {"pointers": [trajectory_ref, f"{BASE}/authority.json", decision_ref]},
            "artifacts": [{"name": ref.removeprefix(prefix + "/metadata/").replace("/", "_"),
                "locator": ref, "availability": "repository_retained", "sha256": sha}
                for ref, sha in sorted(bound.items()) if ref.startswith(prefix + "/metadata/")],
            "migration": {"migrated_at": finalized_at, "training_executed": False, "inference_executed": False,
                "scientific_reinterpretation": False, "verification_scope": "Pinned saved metadata only; no local checkpoint loading or execution."}}
        validate_v5_evidence_envelope(evidence)
        pending[acceptance_ref] = json_bytes({"format": "molgap-rml-terminal-acceptance-adapter-v1",
            "evidence_id": evidence["evidence_id"], "run_id": run_id, "outcome": outcome,
            "trajectory_decision": decision, "role_use": role_use, "costs": costs, "roles": roles, **facts})
        terminal_ref = f"{arm_prefix}/terminal.json"
        pending[f"{arm_prefix}/evidence.json"] = json_bytes(evidence)
        pending[f"{arm_prefix}/cost_role.json"] = json_bytes({"costs": costs, "roles": roles})
        hashes = {**bound, **{ref: hashlib.sha256(pending[ref]).hexdigest() for ref in (acceptance_ref, decision_ref)}}
        pending[terminal_ref] = json_bytes({"format": "molgap-rml-terminal-package-v1",
            "trajectory_id": tid, "action_id": "A001", "run_id": run_id, "finalized_at": finalized_at,
            "acceptance_ref": acceptance_ref, "artifact_hashes": hashes, "evidence": evidence,
            "decision": decision, "costs": costs, "roles": roles, "role_use": role_use})
        result["arms"][arm] = {"trajectory": trajectory_ref, "terminal": terminal_ref}
    if destination.exists():
        inventory = {p.relative_to(root).as_posix() for p in destination.rglob("*") if p.is_file()}
        if inventory != set(pending) or any(repo_local_path(root, ref).read_bytes() != data for ref, data in pending.items()):
            raise ValueError("PREPARATION_CONFLICT: existing output differs; refusing to clobber")
        return {**result, "status": "ALREADY_PREPARED"}
    destination.mkdir(parents=True, exist_ok=False)
    for ref, data in pending.items():
        atomic_write(repo_local_path(root, ref), data)
    return result
