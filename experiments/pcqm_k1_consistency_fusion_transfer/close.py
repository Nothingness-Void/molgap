"""Translate the reviewed inference result into the existing RML finalizer."""
import json
from datetime import datetime, timezone
from pathlib import Path

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-consistency-fusion-transfer-20261004"
RUN = "local-k1-consistency-fusion-transfer-20261004"
EID = "pcqm-k1-consistency-fusion-transfer-20261004"


def main():
    if (HERE / "rml/rml_finalized").exists():
        print(verified_receipt(HERE / "rml/rml_finalized")["finalization_id"])
        return
    inputs = json.loads((HERE / "inputs.json").read_text())
    report = json.loads((HERE / "results/measurement.json").read_text())
    trajectory = json.loads((HERE / "rml/trajectory.json").read_text())
    if report["status"] != "complete_no_training" or report["rows"] != 50000:
        raise ValueError("Inference is incomplete")
    if report["inputs_sha256"] != sha256_file(HERE / "inputs.json") or report["prospective_sha256"] != sha256_file(HERE / "rml/trajectory.json"):
        raise ValueError("Execution was not bound to the prospective inputs")
    for pointer, digest in trajectory["decision_state"]["source_hashes"].items():
        if sha256_file(ROOT / pointer) != digest:
            raise ValueError("Frozen source changed: " + pointer)
    for pointer, digest in report["artifact_hashes"].items():
        if sha256_file(ROOT / pointer) != digest:
            raise ValueError("Output artifact changed")
    for arm in ("mean2", "consistency2"):
        if report["equivalence"][arm]["max_abs_delta_eV"] > 1e-4:
            raise ValueError("Reconstruction not qualified")
    for binding in inputs["bindings"].values():
        if sha256_file(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError("Retained input changed")
    decision_ref = f"{REL}/terminal_decision.md"
    if not (ROOT / decision_ref).is_file():
        raise ValueError("Parent reviewed terminal decision required")
    decision = {"decision_ref": decision_ref, "outcome": "NO_TRAIN", "next_allowed_actions": [],
                "reopen_conditions": ["New training, distillation, scale-up or official-role use needs separate authority and a prospective contract."]}
    outcome = {"execution_status": "complete_no_training", "artifact_status": "aligned_finite_hash_verified_predictions",
        "comparison_status": "same_cohort_retained_checkpoint_equal_blend", "scientific_status": "NO_TRAIN",
        "transfer_status": "internal_development_nomination" if report["nomination_passed"] else "nomination_gate_failed",
        "budget_decision": "no_training_release", "full_handoff_status": "not_applicable"}
    acceptance_ref = f"{REL}/acceptance.json"
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-" + TID + "-measured-inference",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "local-attempt-001",
        "category": "inference", "platform": "local-windows", "hardware": report["hardware"], "evidence_ref": acceptance_ref,
        "measurement": {"device_hours": {"value": report["allocation_seconds"]/3600, "status": "measured"},
                        "cpu_hours": {"value": report["process_cpu_seconds"]/3600, "status": "measured"},
                        "wall_hours": {"value": report["wall_seconds"]/3600, "status": "measured"},
                        "queue_hours": {"value": None, "status": "not_applicable"}}}
    roles = []
    for cohort in ("calibration", "development"):
        for kind in ("prediction_input", "labels_read", "metric_computed", "selection_used"):
            roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-{TID}-{cohort}-{kind}",
                "trajectory_id": TID, "action_id": "A001", "run_id": RUN,
                "dataset_identity": inputs["bindings"]["manifest"]["sha256"],
                "row_manifest_hash": report["cohort_source_idx_sha256"][cohort],
                "role_name": "internal_development", "access_kind": kind, "selection_used": True,
                "evidence_ref": acceptance_ref})
    role_use = {"internal_development": "selection_used", "official_validation": "untouched",
                "test_dev": "untouched", "test_challenge": "untouched"}
    atomic_json(HERE / "acceptance.json", {"format": "molgap-k1-fusion-transfer-acceptance-v1", "evidence_id": EID,
        "run_id": RUN, "outcome": outcome, "trajectory_decision": decision, "role_use": role_use,
        "checks": {"strict_state_load": True, "finite_aligned_50k_outputs": True,
                   "accepted_prediction_reconstruction_passed": True, "no_training": True, "protected_roles_untouched": True},
        "measurement_ref": f"{REL}/results/measurement.json", "cost_scope": report["cost_scope"],
        "roles": roles, "costs": [cost], "limitations": report["limitations"]})
    refs = [f"{REL}/{f}" for f in ("acceptance.json", "terminal_decision.md", "results/measurement.json", "inputs.json", "protocol.md", "prepare.py", "close.py", "run.py", "verification.json")]
    refs += list(inputs["source_hashes"]) + list(report["artifact_hashes"]) + [x["path"] for x in inputs["bindings"].values()]
    hashes = {pointer: sha256_file(ROOT / pointer) for pointer in sorted(set(refs))}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "evidence_id": EID, "track": "B", "contract": "MOLGAP-COMMON-V5-FINAL",
        "legacy_contract": "pcqm-k1-consistency-fusion-transfer-v1", "scope": "desktop_frozen_k1_equal_blend_internal_development_inference",
        "authority": {"pointers": [decision_ref, acceptance_ref, f"{REL}/protocol.md", f"{REL}/results/measurement.json"]},
        "artifacts": [{"name": p, "locator": p, "sha256": h, "availability": "locally_retained_hash_verified"} for p, h in hashes.items()],
        "outcome": outcome, "role_use": role_use,
        "migration": {"migrated_at": "2026-10-04", "training_executed": False, "inference_executed": True,
                      "scientific_reinterpretation": False, "verification_scope": "Prospective clean inference; no training replay claim or model promotion."}}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": TID, "run_id": RUN, "action_id": "A001",
        "finalized_at": datetime.now(timezone.utc).isoformat(), "acceptance_ref": acceptance_ref, "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles}
    atomic_json(HERE / "terminal.json", terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json"), indent=2))


if __name__ == "__main__":
    main()
