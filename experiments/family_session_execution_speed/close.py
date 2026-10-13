"""Accept bounded synthetic timings using the existing comparison/finalize APIs."""
from datetime import datetime, timezone
import json
from pathlib import Path

from molgap.family_execution_diagnostic import compare_trials
from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file

REL = "experiments/family_session_execution_speed"
TID = "TB-family-session-speed-20261014"
RUN = "family-session-speed-20261014"


def main():
    root = Path(__file__).resolve().parents[2]
    here = root / REL
    config = json.loads((here / "config.json").read_text())
    result = json.loads((here / "results/result.json").read_text())
    completion = json.loads((here / "results/completion.json").read_text())
    if (completion["result_sha256"] != sha256_file(here / "results/result.json") or result["status"] != "COMPLETE"
            or result["config_sha256"] != sha256_file(here / "config.json")):
        raise ValueError("Completed prospective result binding differs")
    for source in config["sources"].values():
        if sha256_file(root / source["archive"]) != source["archive_sha256"]:
            raise ValueError("Retained source snapshot changed")
    checks = compare_trials(result["trials"], config)
    if checks != result["checks"] or result["formal_training_executed"] is not False or result["molecular_roles_consumed"] is not False:
        raise ValueError("Synthetic equality/scope differs")
    if not 0 < result["process_wall_seconds"] <= config["maximum_process_wall_seconds"]:
        raise ValueError("Native process budget exceeded")
    for name in ("terminal_decision.md", "attribution.md"):
        if not (here / name).is_file():
            raise ValueError("Write disposition and attribution before RML closure")
    atomic_json(here / "analysis.json", {**checks, "quality": "not_evaluated", "full_epoch_speed": "not_measured"})
    outcome = {"execution_status": "complete_synthetic_session_diagnostic", "artifact_status": "local_hash_verified",
        "comparison_status": "exact_mathematical_endpoints_execution_only", "scientific_status": "NO_TRAIN",
        "transfer_status": "not_evaluated", "budget_decision": "bounded_diagnostic_complete", "full_handoff_status": "not_applicable"}
    decision = {"outcome": "NO_TRAIN", "decision_ref": REL + "/terminal_decision.md", "next_allowed_actions": [],
        "reopen_conditions": ["Separately authorized representative data/runtime/platform test"]}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-family-session-speed-observed-20261014",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "local-001",
        "platform": "local-windows", "hardware": "RTX5060", "category": "audit",
        "evidence_ref": REL + "/acceptance.json", "measurement": {
            "wall_hours": {"value": result["process_wall_seconds"] / 3600, "status": "measured"},
            "device_hours": {"value": None, "status": "measurement_missing"},
            "cpu_hours": {"value": sum(r["process_cpu_seconds"] for r in result["trials"]) / 3600, "status": "measured"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}
    role_use = dict.fromkeys(("train", "development", "official_validation", "test_dev", "test_challenge"), "untouched")
    acceptance = {"evidence_id": RUN, "run_id": RUN, "outcome": outcome, "trajectory_decision": decision,
        "roles": [], "costs": [cost], "role_use": role_use, "checks": checks,
        "scope": "Scratch optimizer updates on generated synthetic chains; no formal/molecular training",
        "cost_scope": "Parent child-process wall includes startup and GPU/CPU/IO; CPU time excludes imports; device allocation/busy time unknown"}
    atomic_json(here / "acceptance.json", acceptance)
    retained = ["acceptance.json", "analysis.json", "terminal_decision.md", "attribution.md", "config.json",
        "protocol.md", "role_plan.json", "results/result.json", "results/completion.json",
        "source_snapshots/reference.zip", "source_snapshots/candidate.zip"]
    hashes = {REL + "/" + name: sha256_file(here / name) for name in retained}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": RUN, "track": "B", "scope": "synthetic_session_execution",
        "legacy_contract": "synthetic-shared-session-speed-v1", "outcome": outcome,
        "authority": {"pointers": [REL + "/" + name for name in ("protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json")]},
        "role_use": role_use, "migration": {"migrated_at": datetime.now(timezone.utc).isoformat(),
            "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "Synthetic scratch session/optimizer diagnostic; not formal training"},
        "observed_execution": {"training_executed": False, "inference_executed": False,
            "scratch_optimizer_updates": True, "execution_ref": REL + "/results/result.json"},
        "artifacts": [{"name": name, "locator": REL + "/" + name, "sha256": hashes[REL + "/" + name],
            "availability": "locally_retained_hash_verified"} for name in retained]}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": TID, "run_id": RUN,
        "action_id": "A001", "finalized_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_ref": REL + "/acceptance.json", "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "costs": [cost], "roles": []}
    atomic_json(here / "terminal.json", terminal)
    print(json.dumps(finalize(root, REL + "/rml", REL + "/terminal.json")))


if __name__ == "__main__":
    main()
