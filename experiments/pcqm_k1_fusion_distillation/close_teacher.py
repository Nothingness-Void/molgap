"""Export the retained teacher prerequisite through the existing RML finalizer."""
import json
from datetime import datetime, timezone
from pathlib import Path

from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()


def main():
    destination = HERE / "cache_prospective/rml_finalized"
    if destination.exists():
        print(json.dumps(verified_receipt(destination), indent=2))
        return
    trajectory_path = HERE / "cache_prospective/trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    report = json.loads((HERE / "teacher_generation.json").read_text())
    inputs = json.loads((HERE / "teacher_inputs.json").read_text())
    manifest_path = HERE / "teacher_cache/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = manifest_path.parent / manifest["path"]
    if (report["status"], report["rows"], report["source_idx_range"], report["parameter_updates"], report["geometry_constructed"], report["protected_roles_untouched"]) != ("complete_no_training", 100000, [0, 100000], False, False, True):
        raise ValueError("Teacher prerequisite execution is incomplete or outside its contract")
    for path, digest in ((HERE / "teacher_inputs.json", report["inputs_sha256"]), (trajectory_path, report["prospective_sha256"]), (manifest_path, report["manifest_sha256"]), (payload, report["payload_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError("Retained teacher input/output digest changed: " + str(path))
    if (manifest["format"], manifest["role"], manifest["source_idx_range"], manifest["teacher_identity"], manifest["dataset_manifest_sha256"], manifest["sha256"]) != ("molgap-k1-teacher-cache-v1", "internal_training", [0, 100000], inputs["teacher_identity"], inputs["dataset_manifest_sha256"], report["payload_sha256"]):
        raise ValueError("Teacher manifest identity changed")
    sources = trajectory["decision_state"]["source_hashes"]
    for pointer in trajectory["state_at_start"]["contract_refs"]:
        if sha256_file(ROOT / pointer) != sources[pointer]:
            raise ValueError("Frozen teacher source changed: " + pointer)
    verified_inputs = []
    for name, binding in inputs["files"].items():
        actual = sha256_file(Path(binding["path"]))
        if actual != binding["sha256"]:
            raise ValueError("External retained teacher input changed: " + name)
        verified_inputs.append({"name": name, **binding, "observed_sha256": actual, "status": "exact_bytes_verified"})
    atomic_json(HERE / "teacher_input_verification.json", {"format": "molgap-retained-input-verification-v1", "verified_at": datetime.now(timezone.utc).isoformat(), "inputs": verified_inputs, "scope": "Hash-only verification of retained external inputs; no model loading, inference or training during closure."})
    tid = trajectory["trajectory_id"]
    run = trajectory["actions"][0]["run_ids"][0]
    action = trajectory["actions"][0]["action_id"]
    attempt = trajectory["actions"][0]["attempt_ids"][0]
    eid = "pcqm-k1-fusion-distillation-teacher-cache-20261004"
    acceptance_ref = f"{REL}/teacher_cache_acceptance.json"
    decision_ref = f"{REL}/teacher_cache_decision.md"
    decision = {"decision_ref": decision_ref, "outcome": "NO_TRAIN", "next_allowed_actions": [], "reopen_conditions": ["Student training belongs to its separate authorized prospective two-strength contract; cache acceptance alone does not release training, scale-up or protected roles."]}
    outcome = {"execution_status": "complete_no_training", "artifact_status": "train_only_finite_hash_verified_teacher_cache", "comparison_status": "not_applicable_cache_prerequisite", "scientific_status": "NO_TRAIN", "transfer_status": "teacher_cache_input_verified", "budget_decision": "prerequisite_closed_no_training_updates", "full_handoff_status": "not_applicable"}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-" + tid + "-measured-inference", "trajectory_id": tid, "action_id": action, "run_id": run, "attempt_id": attempt, "category": "inference", "platform": "local-windows", "hardware": report["hardware"], "evidence_ref": acceptance_ref, "measurement": {"device_hours": {"value": report["allocation_seconds"]/3600, "status": "measured"}, "wall_hours": {"value": report["wall_seconds"]/3600, "status": "measured"}, "cpu_hours": {"value": report["process_cpu_seconds"]/3600, "status": "measured"}, "queue_hours": {"value": None, "status": "not_applicable"}}}
    roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-train-{kind}", "trajectory_id": tid, "action_id": action, "run_id": run, "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1", "row_manifest_hash": report["source_idx_sha256"], "role_name": "official_train_prefix_0_100000", "access_kind": kind, "selection_used": False, "evidence_ref": acceptance_ref} for kind in ("prediction_input", "labels_read", "metric_computed")]
    role_use = {"official_train_prefix_0_100000": "used", "internal_development": "untouched", "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    limitations = ["Teacher training MAE is a descriptive training-membership endpoint, not student learnability or generalization evidence.", "No optimizer updates, student results, V5 READY or model promotion.", "Assigned RTX inference cost window excludes startup, graph loading, hashes, cache export and metric validation; CPU is measured process time, not allocated CPU time.", "External input locators remain retained identity pointers; local verification metadata records their exact hashes."]
    atomic_json(HERE / "teacher_cache_acceptance.json", {"format": "molgap-k1-teacher-cache-acceptance-v1", "evidence_id": eid, "run_id": run, "trajectory_id": tid, "outcome": outcome, "trajectory_decision": decision, "role_use": role_use, "costs": [cost], "roles": roles, "measurement_ref": f"{REL}/teacher_generation.json", "cost_scope": report["cost_scope"], "checks": {"prospective_input_identity": True, "exact_manifest_payload_hashes": True, "exact_ascending100k_training_rows": True, "finite_cache_export_validator_passed_during_generation": True, "no_parameter_updates": True, "protected_roles_untouched": True}, "limitations": limitations})
    refs = [f"{REL}/{name}" for name in ("teacher_cache_acceptance.json", "teacher_cache_decision.md", "teacher_cache_attribution.md", "teacher_generation.json", "teacher_inputs.json", "teacher_input_verification.json", "cache_plan_input.json", "close_teacher.py", "teacher_cache/manifest.json", "teacher_cache/teacher_predictions.pt")]
    refs += trajectory["state_at_start"]["contract_refs"]
    hashes = {pointer: sha256_file(ROOT / pointer) for pointer in sorted(set(refs))}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL", "evidence_id": eid, "track": "B", "scope": "desktop_train_role_frozen_teacher_cache_prerequisite", "legacy_contract": "pcqm-k1-fusion-distillation-teacher-cache-v1", "outcome": outcome, "authority": {"pointers": [decision_ref, acceptance_ref, f"{REL}/protocol.md", f"{REL}/teacher_generation.json"]}, "artifacts": [{"name": p, "locator": p, "sha256": h, "availability": "locally_retained_hash_verified"} for p,h in hashes.items()], "role_use": role_use, "migration": {"migrated_at": "2026-10-04", "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False, "verification_scope": "Metadata-only publication of separately completed prospective teacher inference; no model execution during closure."}, "observed_execution": {"training_executed": False, "inference_executed": True, "execution_ref": f"{REL}/teacher_generation.json"}}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid, "run_id": run, "action_id": action, "finalized_at": datetime.now(timezone.utc).isoformat(), "acceptance_ref": acceptance_ref, "artifact_hashes": hashes, "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles}
    atomic_json(HERE / "teacher_cache_terminal.json", terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/cache_prospective", f"{REL}/teacher_cache_terminal.json"), indent=2))


if __name__ == "__main__":
    main()
