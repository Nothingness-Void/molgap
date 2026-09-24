"""Bind the accepted dual training arms and NO_TRAIN audit to separate RML terminals.

This adapter reads saved metadata and payload hashes. It never instantiates a
model, executes inference, opens a graph dataset, or dispatches remote work.
"""
from __future__ import annotations

import importlib.util

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import file_digest


ROOT_REL = "experiments/pcqm_k1_portability_dual_100k"
ROOT = REPO_ROOT / ROOT_REL
RECORD_REL = "platforms/_records/kaggle/training/pcqm_k1_topology_portability_dual_s42_v1"
RECORD = REPO_ROOT / RECORD_REL
REMOTE_ROOT = RECORD / "pcqm_k1_topology_portability_dual"
RUN_ID = "nothingnessvoid/molgap-k1-topology-portability-dual-s42:v1-planned"
SOURCE_COMMIT = "e58d0ccfa89f33e7eee57712a0c19deb97ea62d1"
ARCHIVE_SHA256 = "31145d13f5e780e299343e491ae2db6a486c918db754c7170bcd0f20b7566f2d"
ACCEPTANCE_SHA256 = "304652a130bb70984a264eae0fc15003b5b10e0aecc18bb27be063e9f5235c9f"
AUDIT_TERMINAL_SHA256 = "a02a2ab432dfee5cefbe232da6aa7e93c4937936b7bbd91a4c15178fc7f031a4"
FINALIZED_AT = "2026-09-25T05:30:58+09:00"
MODES = {
    "rwse_refresh": "neural_atom_k1_rwse_refresh",
    "degree_balance": "neural_atom_k1_degree_balance",
}


def shared_adapter():
    path = REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py"
    spec = importlib.util.spec_from_file_location("shared_rml_terminal_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Shared RML terminal adapter unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_acceptance(shared):
    source = RECORD / "acceptance.json"
    if file_digest(source) != ACCEPTANCE_SHA256:
        raise RuntimeError("Downloaded acceptance changed")
    accepted = shared.load(source)
    training = accepted["training"]
    if (
        accepted.get("accepted") is not True
        or training.get("accepted") is not True
        or accepted.get("model_inference_executed_by_acceptance") is not False
        or accepted.get("audit_inference_executed_in_remote_job") is not True
        or accepted.get("source_commit") != SOURCE_COMMIT
        or accepted.get("source_archive_sha256") != ARCHIVE_SHA256
        or training.get("selected_candidate") is not None
        or any(accepted.get(flag) is not False for flag in (
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read",
        ))
        or set(training["candidates"]) != set(MODES.values())
    ):
        raise RuntimeError("Terminal acceptance or role identity changed")
    terminal_source = REMOTE_ROOT / "post100k_audit/terminal.json"
    if file_digest(terminal_source) != AUDIT_TERMINAL_SHA256:
        raise RuntimeError("Frozen audit terminal changed")
    return accepted, shared.load(terminal_source)


def prepare_arms(accepted, shared):
    training = accepted["training"]
    source = {
        **training,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": ARCHIVE_SHA256,
        "model_inference_executed": False,
    }
    for short_name, mode in MODES.items():
        arm_rel = f"{ROOT_REL}/arms/{short_name}"
        arm = REPO_ROOT / arm_rel
        trajectory = shared.load(arm / "trajectory.json")
        action = trajectory["actions"][0]
        if (
            trajectory["trajectory_id"] != f"TC-k1-{short_name.replace('_', '-')}-100k-s42"
            or action["action_id"] != "A001"
            or action["run_ids"] != [RUN_ID]
            or action["source_commit"] != SOURCE_COMMIT
            or training["candidates"][mode]["gate"]["passed"] is not False
        ):
            raise RuntimeError(f"Frozen arm or gate changed: {mode}")
        adapter = shared_adapter()
        adapter.ROOT = arm
        adapter.ROOT_REL = arm_rel
        adapter.SHARED_ROOT = ROOT
        adapter.SHARED_REL = ROOT_REL
        adapter.RESULTS = arm / "results"
        adapter.RECORD_ROOT = RECORD
        adapter.CANDIDATE_ROOT = REMOTE_ROOT / mode
        adapter.RAW_ACCEPTANCE = adapter.RESULTS / "acceptance_v1.json"
        adapter.SOURCE_ACCEPTANCE_REL = "results/acceptance_v1.json"
        adapter.write(adapter.RAW_ACCEPTANCE, source)
        adapter.TRAJECTORY_ID = trajectory["trajectory_id"]
        adapter.RUN_ID = RUN_ID
        adapter.MODE = mode
        adapter.ACTION_ID = "A001"
        adapter.EVIDENCE_ID = trajectory["result"]["evidence_ids"][0]
        adapter.LOCAL_RECORD_URI = f"external://local-platform-record/{RECORD_REL}/pcqm_k1_topology_portability_dual/{mode}"
        adapter.DECISION_OUTCOME = "POSITIVE_BELOW_GATE"
        adapter.SCIENTIFIC_STATUS = "positive_below_gate"
        adapter.NEXT_ALLOWED_ACTIONS = ["close this exact arm without scale or protected-role release"]
        adapter.REOPEN_CONDITIONS = ["a distinct mechanism requires a new prospective contract and budget"]
        adapter.FINALIZED_AT = FINALIZED_AT
        adapter.MIGRATED_AT = "2026-09-25"
        adapter.PLATFORM = "kaggle1"
        adapter.HARDWARE = "Tesla_T4_16GB"
        adapter.EXTRA_ARTIFACTS = (
            ("submission_receipt", adapter.shared_rel("results/submission_receipt.json")),
            ("frozen_portability_analysis", adapter.shared_rel("results/portability_analysis.json")),
        )
        adapter.main()
        print(f"Prepared training RML terminal: {short_name}", flush=True)


def prepare_audit(accepted, remote_terminal, shared):
    audit_rel = f"{ROOT_REL}/audit"
    audit = ROOT / "audit"
    results = audit / "results"
    trajectory = shared.load(audit / "trajectory.json")
    action = trajectory["actions"][0]
    trajectory_id = trajectory["trajectory_id"]
    evidence_id = trajectory["result"]["evidence_ids"][0]
    if (
        trajectory_id != "TC-k1-topology-portability-post100k-audit-s42"
        or action["action_id"] != "A001"
        or action["run_ids"] != [RUN_ID]
        or action["source_commit"] != SOURCE_COMMIT
        or remote_terminal.get("complete") is not True
        or remote_terminal.get("experiment_purpose") != "NO_TRAIN"
        or remote_terminal.get("training_executed_in_audit_stage") is not False
        or remote_terminal.get("model_inference_executed") is not True
    ):
        raise RuntimeError("Frozen NO_TRAIN action or terminal changed")
    shared.write(results / "audit_terminal.json", remote_terminal)
    summary = {
        "format": "molgap-k1-portability-audit-terminal-acceptance-v1",
        "accepted": True,
        "evidence_id": evidence_id,
        "run_id": RUN_ID,
        "source_acceptance_sha256": ACCEPTANCE_SHA256,
        "audit_terminal_sha256": AUDIT_TERMINAL_SHA256,
        "model_inference_executed_during_acceptance": False,
        "model_inference_executed_in_remote_job": True,
        "training_executed_in_remote_audit_stage": False,
        "role_source": "fixed100k-development-100000-150000 and fixed500k-development-500000-550000",
        "protected_roles_read": False,
        "mae_eV": {name: row["mae_eV"] for name, row in accepted["audit"].items()},
    }
    shared.write(results / "acceptance_summary.json", summary)

    role_use = {
        "internal_development_100000_150000": "used",
        "internal_development_500000_550000": "used",
        "official_validation": "untouched",
        "test_dev": "untouched",
        "test_challenge": "untouched",
    }
    roles = []
    for role_name, dataset, manifest, suffix in (
        ("internal_development_100000_150000", "pcqm4mv2-ogb-fixed-100k-v1",
         "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d", "original-100k"),
        ("internal_development_500000_550000", "pcqm4mv2-ogb-fixed-500k-scnet-v1",
         "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751", "frozen-500k"),
    ):
        for kind in ("prediction_input", "labels_read", "metric_computed"):
            roles.append({
                "schema": "molgap-role-event-v1",
                "role_event_id": f"role-{trajectory_id}-{suffix}-{kind}",
                "trajectory_id": trajectory_id, "action_id": "A001", "run_id": RUN_ID,
                "dataset_identity": dataset, "row_manifest_hash": manifest,
                "role_name": role_name, "access_kind": kind, "selection_used": False,
                "evidence_ref": f"{audit_rel}/results/terminal_acceptance.json",
            })
    shared.write(results / "role_history.json", {
        "format": "molgap-role-history-index-v1", "trajectory_id": trajectory_id,
        "protected_roles_read": False, "events": roles,
    })
    missing = {"value": None, "status": "measurement_missing"}
    cost = {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": f"cost-{trajectory_id}-inference",
        "trajectory_id": trajectory_id, "action_id": "A001", "run_id": RUN_ID,
        "attempt_id": "post100k-audit-v1", "category": "audit",
        "platform": "kaggle1", "hardware": "Tesla_T4_16GB",
        "measurement": {
            "device_hours": missing, "cpu_hours": missing,
            "wall_hours": {"value": float(remote_terminal["elapsed_seconds"]) / 3600.0,
                           "status": "measured"},
            "queue_hours": missing,
        },
        "evidence_ref": f"{audit_rel}/results/terminal_acceptance.json",
    }
    shared.write(results / "cost_records.json", {"format": "molgap-cost-records-v1", "costs": [cost]})
    outcome = {
        "execution_status": "complete", "artifact_status": "accepted",
        "comparison_status": "paired_endpoint_diagnostic",
        "scientific_status": "no_train_negative_transfer",
        "transfer_status": "frozen_portability_regressed",
        "budget_decision": "stop_under_contract",
        "full_handoff_status": "not_authorized",
    }
    decision = {
        "decision_ref": f"{audit_rel}/decision.md", "outcome": "NO_TRAIN",
        "next_allowed_actions": [],
        "reopen_conditions": ["a new independent-role or training question needs a new prospective authority"],
        "final": True,
    }
    artifacts = (
        ("acceptance_summary", f"{audit_rel}/results/acceptance_summary.json"),
        ("audit_terminal", f"{audit_rel}/results/audit_terminal.json"),
        ("portability_analysis", f"{ROOT_REL}/results/portability_analysis.json"),
        ("submission_receipt", f"{ROOT_REL}/results/submission_receipt.json"),
        ("role_history", f"{audit_rel}/results/role_history.json"),
        ("cost_records", f"{audit_rel}/results/cost_records.json"),
    )
    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL", "legacy_contract": "none-prospective-v5",
        "evidence_id": evidence_id, "track": "C",
        "scope": "terminal_fixed500k_internal_development_no_train_portability_audit",
        "outcome": outcome, "role_use": role_use,
        "artifacts": [
            {"name": name, "locator": path, "sha256": file_digest(REPO_ROOT / path),
             "availability": "repository_retained"}
            for name, path in artifacts
        ],
        "authority": {"pointers": [
            f"{ROOT_REL}/protocol.md", f"{ROOT_REL}/training_contract.json",
            f"{ROOT_REL}/source_config.json", f"{audit_rel}/role_plan.json",
            f"{audit_rel}/decision.md", f"{audit_rel}/trajectory.json",
        ]},
        "migration": {
            "training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False, "migrated_at": "2026-09-25",
            "verification_scope": "remote frozen-checkpoint inference; local no-inference acceptance and terminal binding",
        },
    }
    shared.write(results / "terminal_evidence.json", evidence)
    terminal_acceptance = {
        "format": "molgap-rml-terminal-acceptance-adapter-v1",
        "evidence_id": evidence_id, "run_id": RUN_ID,
        "outcome": outcome, "trajectory_decision": decision,
        "role_use": role_use, "costs": [cost], "roles": roles,
        "source_acceptance_ref": f"{audit_rel}/results/acceptance_summary.json",
        "source_acceptance_sha256": file_digest(results / "acceptance_summary.json"),
    }
    shared.write(results / "terminal_acceptance.json", terminal_acceptance)
    bound = {path for _, path in artifacts}
    bound.update(evidence["authority"]["pointers"])
    bound.add(f"{audit_rel}/results/terminal_acceptance.json")
    bound.add(decision["decision_ref"])
    terminal = {
        "format": "molgap-rml-terminal-package-v1",
        "trajectory_id": trajectory_id, "run_id": RUN_ID, "action_id": "A001",
        "finalized_at": FINALIZED_AT,
        "acceptance_ref": f"{audit_rel}/results/terminal_acceptance.json",
        "artifact_hashes": {path: file_digest(REPO_ROOT / path) for path in sorted(bound)},
        "evidence": evidence, "decision": decision,
        "costs": [cost], "roles": roles, "role_use": role_use,
    }
    shared.write(results / "terminal.json", terminal)
    print("Prepared separate NO_TRAIN audit RML terminal", flush=True)


def main() -> None:
    shared = shared_adapter()
    accepted, remote_terminal = checked_acceptance(shared)
    prepare_arms(accepted, shared)
    prepare_audit(accepted, remote_terminal, shared)


if __name__ == "__main__":
    main()
