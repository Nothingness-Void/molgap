"""Translate inspected BN diagnostic artifacts through the existing finalizer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = ROOT / "experiments/pcqm_k1_500k_bn_calibration"
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-500k-bn-calibration-20261007"
RUN = "local-k1-500k-bn-calibration-20261007"
EID = "pcqm-k1-500k-bn-calibration-20261007"


def main():
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    result = json.loads((HERE / "results/analysis.json").read_text(encoding="utf-8"))
    required = {"strict_state_loading", "source_and_cache_hashes", "selected_prediction_reconstruction",
        "exact_rows_targets", "finite_predictions", "training_only_calibration", "parameters_unchanged",
        "non_bn_buffers_unchanged", "buffers_restored", "restored_inference_exact",
        "no_optimization", "protected_roles_untouched"}
    if result["status"] != "complete" or not required <= result["checks"].keys() or not all(value is True for value in result["checks"].values()):
        raise ValueError("Diagnostic checks incomplete")
    if result["inputs_sha256"] != sha256_file(HERE / "inputs.json") or result["prospective_sha256"] != sha256_file(HERE / "rml/trajectory.json"):
        raise ValueError("Prospective/result binding differs")
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Executed diagnostic source differs: {name}")
    if result["cost"]["worker_wall_seconds"] > inputs["worker_wall_ceiling_seconds"]:
        raise ValueError("Diagnostic exceeded frozen wall ceiling")
    outcome = {"execution_status": "complete_no_training", "artifact_status": "local_hash_verified",
        "comparison_status": "consumed_role_paired_bn_state_diagnostic", "scientific_status": "NO_TRAIN",
        "transfer_status": "not_evaluated", "budget_decision": "bounded_local_diagnostic_complete",
        "full_handoff_status": "not_applicable"}
    decision = {"outcome": "NO_TRAIN", "decision_ref": f"{REL}/terminal_decision.md",
        "next_allowed_actions": [], "reopen_conditions": [
            "Separately frozen and authorized independent-role validation or training-time BN intervention"]}
    role_use = {"train_prefix": "prediction_input", "internal_development": "selection_used",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-k1-500k-bn-calibration-observed",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001",
        "platform": "local-windows", "hardware": "CPU; four intra-op threads; no accelerator",
        "category": "inference", "evidence_ref": f"{REL}/acceptance.json", "measurement": {
            "wall_hours": {"value": result["cost"]["worker_wall_seconds"] / 3600, "status": "measured"},
            "cpu_hours": {"value": result["cost"]["worker_process_cpu_seconds"] / 3600, "status": "measured"},
            "device_hours": {"value": None, "status": "not_applicable"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}
    roles = []
    decoded = {role: hashlib.sha256(np.arange(*bounds, dtype="<i8").tobytes()).hexdigest()
        for role, bounds in [("train_prefix", inputs["train_range"]),
                             ("internal_development", inputs["development_range"]) ]}
    if decoded != result["decoded_role_manifest_hashes"] or result["sample_source_idx_sha256"]["development"] != decoded["internal_development"]:
        raise ValueError("Observed decoded/development membership differs")
    if result["max_selected_prediction_reconstruction_eV"] > inputs["prediction_tolerance_eV"]:
        raise ValueError("Selected predictor reconstruction exceeds tolerance")
    for role, sample_key, access_kinds in [
        ("train_prefix", "calibration", ["labels_read", "prediction_input"]),
        ("internal_development", "development", ["labels_read", "prediction_input", "metric_computed", "selection_used"]),
    ]:
        for access in access_kinds:
            roles.append({"schema": "molgap-role-event-v1",
                "role_event_id": f"role-k1-500k-bn-calibration-{role}-{access}",
                "trajectory_id": TID, "action_id": "A001", "run_id": RUN,
                "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
                "row_manifest_hash": decoded[role] if access == "labels_read" else result["sample_source_idx_sha256"][sample_key],
                "role_name": role, "access_kind": access, "selection_used": role == "internal_development",
                "evidence_ref": f"{REL}/acceptance.json"})
    acceptance = {"format": "molgap-k1-bn-calibration-acceptance-v1", "evidence_id": EID,
        "run_id": RUN, "outcome": outcome, "trajectory_decision": decision, "role_use": role_use,
        "costs": [cost], "roles": roles, "checks": result["checks"],
        "max_prediction_reconstruction_eV": result["max_selected_prediction_reconstruction_eV"],
        "metrics": result["metrics"],
        "bootstrap_probability_semantics": "gain_eV uses original_error-calibrated_error. Existing helper probability_better is P(input_delta<0), hence here P(calibration_worse), not P(calibration_improves). The gate uses delta and ci95 only.",
        "comparison_class": "CONTEXT_ONLY", "analysis_ref": f"{REL}/results/analysis.json",
        "role_scope": "Loader decoded all500K training and50K development backing label tensors. Calibration uses only16384 sampled training-member features; no label objective or gradients. Full consumed development metrics inform diagnostic interpretation. No protected role.",
        "decoded_role_manifest_hashes": decoded,
        "calibration_scope": "Parameter-free BN running-state adaptation; learned parameters fixed, dropout disabled, cumulative updates. This is not supervised training or training replay readiness.",
        "cost_scope": "Worker startup/bootstrap, worker hashing, loading, BN calibration and inference included; preparation hashing, tests, metadata publication and Git overhead excluded.",
        "limitations": result["limits"]}
    atomic_json(HERE / "acceptance.json", acceptance)
    retained = ["results/analysis.json", "results/original.pt", "results/calibrated.pt",
        "results/calibrated_buffers.pt", "acceptance.json", "terminal_decision.md", "attribution.md"]
    paths = [p for p in HERE.rglob("*") if p.is_file() and "rml" not in p.relative_to(HERE).parts
        and "__pycache__" not in p.parts and p.name != "terminal.json"]
    paths += [ROOT / "src/molgap/k1_frozen_inference.py", ROOT / "src/molgap/k1_bn_calibration.py",
        ROOT / "research_memory/policies/pcqm-k1-500k-bn-calibration.1.json"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EID, "track": "B", "scope": "desktop_k1_500k_bn_calibration_diagnostic",
        "legacy_contract": "pcqm-k1-500k-bn-calibration-v1", "outcome": outcome,
        "authority": {"pointers": [f"{REL}/{name}" for name in [
            "protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json", "results/analysis.json"]]},
        "role_use": role_use,
        "migration": {"migrated_at": "2026-10-07", "training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False,
            "verification_scope": "Metadata publication of separately executed prospective BN calibration/inference; no model execution by finalizer"},
        "observed_execution": {"training_executed": False, "inference_executed": True,
            "execution_ref": f"{REL}/results/analysis.json"},
        "artifacts": [{"name": name, "locator": f"{REL}/{name}", "sha256": hashes[f"{REL}/{name}"],
            "availability": "locally_retained_hash_verified"} for name in retained]}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": TID,
        "run_id": RUN, "action_id": "A001", "finalized_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_ref": f"{REL}/acceptance.json", "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles}
    atomic_json(HERE / "terminal.json", terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json")))


if __name__ == "__main__":
    main()
