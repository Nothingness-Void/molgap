"""Publish inspected local evidence through the existing RML finalizer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-500k-bottleneck-diagnostic-20261007"
RUN = "local-k1-500k-bottleneck-20261007"
EID = "pcqm-k1-500k-bottleneck-diagnostic-20261007"


def main():
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    result = json.loads((HERE / "results/analysis.json").read_text(encoding="utf-8"))
    if result["status"] != "complete" or not all(result["checks"].values()):
        raise ValueError("Diagnostic checks incomplete")
    if result["inputs_sha256"] != sha256_file(HERE / "inputs.json") or result["prospective_sha256"] != sha256_file(HERE / "rml/trajectory.json"):
        raise ValueError("Prospective/result binding differs")
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Executed diagnostic source differs")
    if result["cost"]["worker_wall_seconds"] > inputs["worker_wall_ceiling_seconds"]:
        raise ValueError("Diagnostic exceeded frozen wall ceiling")
    outcome = {"execution_status": "complete_no_training", "artifact_status": "local_hash_verified",
        "comparison_status": "consumed_role_frozen_intervention_and_clean_state_comparison",
        "scientific_status": "NO_TRAIN", "transfer_status": "not_evaluated",
        "budget_decision": "bounded_local_diagnostic_complete", "full_handoff_status": "not_applicable"}
    decision = {"outcome": "NO_TRAIN", "decision_ref": f"{REL}/terminal_decision.md",
        "next_allowed_actions": [], "reopen_conditions": [
            "Separately frozen and authorized training intervention to identify independent-slot capacity or pretraining-scale efficacy"]}
    role_use = {"train_prefix": "selection_used", "internal_development": "selection_used",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-k1-500k-bottleneck-observed",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001",
        "platform": "local-windows", "hardware": "CPU; four intra-op threads; no accelerator",
        "category": "inference", "evidence_ref": f"{REL}/acceptance.json", "measurement": {
            "wall_hours": {"value": result["cost"]["worker_wall_seconds"] / 3600, "status": "measured"},
            "cpu_hours": {"value": result["cost"]["worker_process_cpu_seconds"] / 3600, "status": "measured"},
            "device_hours": {"value": None, "status": "not_applicable"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}
    roles = []
    manifest = json.loads((Path(inputs["cache_root"]) / "manifest.json").read_text(encoding="utf-8"))
    decoded_hashes = {}
    for role_name, group in [("train_prefix", "train_combined"), ("internal_development", "development")]:
        role_range = manifest["roles"]["train" if role_name == "train_prefix" else "development"]
        decoded_hashes[role_name] = hashlib.sha256(np.arange(
            role_range["source_idx_start"], role_range["source_idx_stop"], dtype="<i8").tobytes()).hexdigest()
        for access in ["prediction_input", "labels_read", "metric_computed", "selection_used"]:
            roles.append({"schema": "molgap-role-event-v1",
                "role_event_id": f"role-k1-500k-bottleneck-{role_name}-{access}",
                "trajectory_id": TID, "action_id": "A001", "run_id": RUN,
                "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
                "row_manifest_hash": decoded_hashes[role_name] if access == "labels_read" else result["sample_source_idx_sha256"][group],
                "role_name": role_name, "access_kind": access, "selection_used": True,
                "evidence_ref": f"{REL}/acceptance.json"})
    acceptance = {"format": "molgap-k1-frozen-bottleneck-acceptance-v1", "evidence_id": EID,
        "run_id": RUN, "outcome": outcome, "trajectory_decision": decision, "role_use": role_use,
        "costs": [cost], "roles": roles, "checks": result["checks"],
        "max_prediction_reconstruction_eV": result["max_selected_prediction_reconstruction_eV"],
        "comparison_class": "CONTEXT_ONLY", "analysis_ref": f"{REL}/results/analysis.json",
        "role_scope": "Whole accepted train/development shards were decoded, including backing label tensors; only the fixed4096 sample rows were evaluated. No protected role.",
        "decoded_role_manifest_hashes": decoded_hashes,
        "sample_scope": "The2048 train diagnostic is stratified1024/1024, not population-proportional. Report both strata or use20/80 weights for a population point estimate.",
        "cost_scope": "Worker startup, loading and inference only. Preparation/hashing, tests, metadata publication and Git overhead excluded.",
        "timestamp_note": "analysis.started_at was captured at report publication; it is not the worker start timestamp. Elapsed cost comes from monotonic timers.",
        "limitations": result["limits"]}
    atomic_json(HERE / "acceptance.json", acceptance)
    retained = ["results/analysis.json", "results/selected.pt", "results/final.pt",
                "results/last_slot_off.pt", "results/all_slots_off.pt", "acceptance.json",
                "terminal_decision.md", "attribution.md", "pretraining_lineage.json"]
    paths = [p for p in HERE.rglob("*") if p.is_file() and "rml" not in p.relative_to(HERE).parts
             and "__pycache__" not in p.parts and p.name != "terminal.json"]
    paths += [ROOT / "src/molgap/k1_frozen_inference.py",
              ROOT / "research_memory/policies/pcqm-k1-500k-bottleneck-diagnostic.1.json"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EID, "track": "B", "scope": "desktop_k1_500k_frozen_bottleneck_diagnostic",
        "legacy_contract": "pcqm-k1-500k-bottleneck-diagnostic-v1", "outcome": outcome,
        "authority": {"pointers": [f"{REL}/{name}" for name in ["protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json", "results/analysis.json"]]},
        "role_use": role_use,
        "migration": {"migrated_at": "2026-10-07", "training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False, "verification_scope": "Metadata publication of separately executed prospective local inference; no execution by finalizer"},
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
