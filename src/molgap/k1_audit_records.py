"""Separate failed audit attempts from recovered physical jobs in V5/RML."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .research_memory.trace import file_digest
from .training_reproducibility import atomic_json


def prepare_audit_records(repo: Path, experiment: str, audit_record: Path, *, finalized_at: str):
    root = repo / experiment
    frozen_ref = f"{experiment}/audit_plan/trajectory.json"
    frozen = json.loads((repo / frozen_ref).read_text())
    recovery = json.loads((audit_record / "pcqm_k1_linear_attention_audit/recovery_execution.json").read_text())
    terminal = json.loads((audit_record / "pcqm_k1_linear_attention_audit/post100k_audit/terminal.json").read_text())
    receipt_ref = f"{experiment}/results/audit_recovery_receipt.json"
    receipt = json.loads((repo / receipt_ref).read_text())
    if (recovery.get("complete") is not True or recovery.get("training_executed") is not False
        or recovery.get("optimizer_steps") != 0 or recovery.get("cublas_workspace_config") != ":4096:8"
        or terminal.get("complete") is not True or terminal.get("training_executed_in_audit_stage") is not False
        or any(terminal.get(k) is not False for k in (
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))):
        raise ValueError("Recovery identity or NO_TRAIN/role truth failed")
    results = root / "results"
    atomic_json(results / "audit_terminal.json", terminal)
    atomic_json(results / "audit_recovery_execution.json", recovery)
    atomic_json(results / "audit_acceptance_summary.json", {
        "accepted": True, "model_inference_executed_by_acceptance": False,
        "source_acceptance_ref": (audit_record / "controller_acceptance.json").relative_to(repo).as_posix(),
        "source_acceptance_sha256": file_digest(audit_record / "controller_acceptance.json"),
        "checkpoint_sha256": terminal["checkpoint_sha256"],
        "reproduction": terminal["reproduction"],
        "500k_mae_eV": {k: v["mae_eV"] for k, v in terminal["unseen_500k"].items()},
        "protected_roles_read": False,
    })
    protected = {k: "untouched" for k in ("official_validation", "test_dev", "test_challenge")}
    original_run = frozen["actions"][0]["run_ids"][0]
    resolved_run = f"{receipt['resolved_kernel']}:v{receipt['version']}"
    for complete in (False, True):
        directory = "audit_recovery" if complete else "audit_failure"
        prefix = f"{experiment}/{directory}"
        tid = frozen["trajectory_id"] + ("-recovery" if complete else "")
        eid = tid.replace("TC-", "pcqm-", 1)
        run_id = resolved_run if complete else original_run
        source_ref = f"{prefix}/results/terminal_acceptance.json"
        outcome = {
            "execution_status": "complete" if complete else "failed",
            "artifact_status": "accepted" if complete else "failure_logs_preserved",
            "comparison_status": "paired_endpoint_diagnostic" if complete else "no_comparison",
            "scientific_status": "no_train_negative_transfer" if complete else "not_evaluated",
            "transfer_status": "frozen_portability_regressed" if complete else "unavailable",
            "budget_decision": "stop_under_contract" if complete else "recovery_recorded_separately",
            "full_handoff_status": "not_authorized",
        }
        decision = {"decision_ref": f"{experiment}/decision.md", "outcome": "NO_TRAIN" if complete else "INFRASTRUCTURE_ONLY",
            "next_allowed_actions": [], "reopen_conditions": ["new authority required"], "final": True}
        roles = []
        role_use = dict(protected)
        if complete:
            for role, dataset, digest in (
                ("internal_development_100000_150000", "pcqm4mv2-ogb-fixed-100k-v1", terminal["manifest_sha256"]["original_100k"]),
                ("internal_development_500000_550000", "pcqm4mv2-ogb-fixed-500k-scnet-v1", terminal["manifest_sha256"]["unseen_500k"]),
            ):
                role_use[role] = "used"
                for kind in ("prediction_input", "labels_read", "metric_computed"):
                    roles.append({"schema": "molgap-role-event-v1",
                        "role_event_id": f"role-{tid}-{role}-{kind}", "trajectory_id": tid,
                        "action_id": "A001", "run_id": run_id, "dataset_identity": dataset,
                        "row_manifest_hash": digest, "role_name": role, "access_kind": kind,
                        "selection_used": False, "evidence_ref": source_ref})
        else:
            role_use["internal_development_100000_150000"] = "unknown"
        missing = {"status": "measurement_missing", "value": None}
        cost = {"schema": "molgap-cost-event-v1",
            "cost_event_id": f"cost-{tid}" if complete else frozen["actions"][0]["cost_event_ids"][0],
            "trajectory_id": tid, "action_id": "A001", "run_id": run_id,
            "attempt_id": "frozen-audit-recovery-v1" if complete else "original-audit-cublas-failure",
            "category": "audit" if complete else "infrastructure_failure",
            "platform": "kaggle1", "hardware": "Tesla_T4_16GB",
            "measurement": {"device_hours": missing, "cpu_hours": missing, "queue_hours": missing,
                "wall_hours": {"status": "measured", "value": recovery["elapsed_seconds"] / 3600} if complete else missing},
            "evidence_ref": source_ref}
        artifact_refs = [f"{experiment}/results/audit_failure_v1.md", receipt_ref]
        if complete:
            artifact_refs += [f"{experiment}/results/audit_terminal.json",
                f"{experiment}/results/audit_recovery_execution.json",
                f"{experiment}/results/audit_acceptance_summary.json",
                f"{experiment}/results/portability_analysis.json"]
        authority = [f"{experiment}/protocol.md", f"{experiment}/training_contract.json",
            f"{experiment}/audit_role_plan.json", f"{experiment}/decision.md", frozen_ref,
            f"{experiment}/audit_recovery.json", receipt_ref]
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "legacy_contract": "none-v5-audit-physical-run-binding",
            "evidence_id": eid, "track": "C", "scope": "recovered_frozen_NO_TRAIN_audit" if complete else "original_audit_execution_failure",
            "outcome": outcome, "role_use": role_use, "authority": {"pointers": authority},
            "artifacts": [{"name": Path(p).stem, "locator": p, "availability": "repository_retained",
                           "sha256": file_digest(repo / p)} for p in artifact_refs],
            "migration": {"training_executed": False, "inference_executed": False,
                "scientific_reinterpretation": False, "migrated_at": finalized_at,
                "verification_scope": "saved metadata/chunks; distinct physical run IDs, no local model execution"}}
        accepted = {"format": "molgap-rml-terminal-acceptance-adapter-v1", "evidence_id": eid,
            "run_id": run_id, "outcome": outcome, "trajectory_decision": decision,
            "role_use": role_use, "costs": [cost], "roles": roles,
            "cost_caveat": "Audit wall interval only; device/setup/queue costs unavailable. Not account-billed hours."}
        atomic_json(repo / source_ref, accepted)
        if not complete:
            bound = set(artifact_refs + authority + [source_ref, decision["decision_ref"]])
            atomic_json(repo / prefix / "results/terminal.json", {
                "format": "molgap-rml-terminal-package-v1", "trajectory_id": tid,
                "run_id": run_id, "action_id": "A001", "finalized_at": finalized_at,
                "acceptance_ref": source_ref,
                "artifact_hashes": {p: file_digest(repo / p) for p in sorted(bound)},
                "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles,
                "role_use": role_use})
            continue
        # Do not retrofit a physical recovery job into a different frozen run ID.
        trajectory = copy.deepcopy(frozen)
        trajectory.pop("decision_state", None)
        trajectory.update(trajectory_id=tid, record_mode="retrospective_partial")
        trajectory["hypothesis"]["hypothesis_id"] = f"H-{tid}"
        trajectory["hypothesis"]["historical_unknowns"] = [
            "Recovery spec/receipt preceded execution, but no native RML plan bound this physical recovery ID",
            "No device allocation/account-billed cost available for either audit attempt"]
        trajectory["hypothesis"]["expected_native_cost_ref"] = cost["cost_event_id"]
        trajectory["state_at_start"]["parent_trajectory_ids"] = [frozen["trajectory_id"]]
        trajectory["actions"] = [{"action_id": "A001", "type": "frozen_NO_TRAIN_infrastructure_recovery",
            "run_ids": [run_id], "attempt_ids": ["v1"], "source_commit": recovery["spec"]["launcher_commit"],
            "evidence_refs": [receipt_ref, f"{experiment}/audit_recovery.json"],
            "cost_event_ids": [cost["cost_event_id"]]}]
        trajectory["result"] = {"evidence_ids": [eid], "evidence_refs": [f"{prefix}/v5_evidence.json"]}
        trajectory["decision"] = decision
        atomic_json(repo / prefix / "trajectory.json", trajectory)
        atomic_json(repo / prefix / "v5_evidence.json", evidence)
        atomic_json(repo / prefix / "costs" / f"{cost['cost_event_id']}.json", cost)
        for role in roles:
            atomic_json(repo / prefix / "roles" / f"{role['role_event_id']}.json", role)
