"""Validate diagnostic measurements, then reuse canonical V5/RML finalization."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file

REL = "experiments/family_checkpoint_efficiency"
TID = "TB-family-checkpoint-efficiency-20261013"
RUN = "family-checkpoint-efficiency-20261013"
EID = "family-checkpoint-efficiency-20261013"


def main():
    root = Path(__file__).resolve().parents[2]
    here = root / REL
    if (here / "rml/rml_finalized/finalization.json").is_file():
        print(json.dumps(finalize(root, f"{REL}/rml", f"{REL}/terminal.json")))
        return
    read = lambda name: json.loads((here / name).read_text(encoding="utf-8"))
    result, config, trajectory = read("results/result.json"), read("config.json"), read("rml/trajectory.json")
    if (read("results/completion.json")["result_sha256"] != sha256_file(here / "results/result.json")
            or result["config_sha256"] != sha256_file(here / "config.json")
            or trajectory["state_at_start"]["source_config_identity"] != canonical_fingerprint(config)):
        raise ValueError("Completed prospective result/config identity differs")
    if (result["status"] != "COMPLETE" or any(result[name] is not False for name in
            ("training_executed", "inference_executed", "molecular_roles_consumed"))
            or not all(result["checks"].values())):
        raise ValueError("Diagnostic scope or mechanical checks failed")
    expected = {(family, device, count, operation) for family in config["families"] for device in config["devices"]
        for count in config["cursor_lengths"] for operation in ("validate", "validate_publish")
        if operation == "validate" or count in config["publish_cursor_lengths"]}
    identities = [(r["family"], r["device"], r["cursor_length"], r["operation"]) for r in result["cases"]]
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError("Missing or duplicated measured cases")
    analysis = {"cases": [], "training_speedup": "not_measured", "quality": "not_evaluated"}
    for row in result["cases"]:
        values = {}
        for name in ("original", "batched"):
            times = row[name + "_seconds"]
            if len(times) != config["repeats"] or any(not math.isfinite(x) or x <= 0 for x in times):
                raise ValueError("Invalid repeat/timing")
            values[name] = {"median_seconds": statistics.median(times), "range_seconds": [min(times), max(times)]}
        if row["operation"] == "validate_publish" and len(row["identical_checkpoint_sha256"]) != 64:
            raise ValueError("Missing byte-identical checkpoint binding")
        values.update(family=row["family"], device=row["device"], cursor_length=row["cursor_length"],
            operation=row["operation"], time_reduction_fraction=1 - values["batched"]["median_seconds"] / values["original"]["median_seconds"])
        analysis["cases"].append(values)
    if not 0 < result["wall_seconds"] < config["total_wall_seconds"] or not 0 < result["gpu_phase_seconds"] < config["gpu_wall_seconds"]:
        raise ValueError("Measured diagnostic exceeded its native budget")
    if result["runtime"]["accelerator"] != config["accelerator"]:
        raise ValueError("Accelerator allocation differs")
    atomic_json(here / "analysis.json", analysis)
    decision = {"outcome": "NO_TRAIN", "decision_ref": f"{REL}/terminal_decision.md",
        "next_allowed_actions": [], "reopen_conditions": ["Separately frozen native training/platform performance contract"]}
    outcome = {"execution_status": "complete_synthetic_checkpoint_diagnostic", "artifact_status": "local_hash_verified",
        "comparison_status": "matched_state_bytes_execution_only", "scientific_status": "NO_TRAIN",
        "transfer_status": "not_evaluated", "budget_decision": "bounded_diagnostic_complete", "full_handoff_status": "not_applicable"}
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-family-checkpoint-observed-20261013",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "local-001",
        "platform": "local-windows", "hardware": config["accelerator"], "category": "audit",
        "evidence_ref": f"{REL}/acceptance.json", "measurement": {
            "wall_hours": {"value": result["wall_seconds"] / 3600, "status": "measured"},
            "device_hours": {"value": result["gpu_phase_seconds"] / 3600, "status": "measured"},
            "cpu_hours": {"value": result["process_cpu_seconds"] / 3600, "status": "measured"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}
    role_use = dict.fromkeys(("train", "development", "official_validation", "test_dev", "test_challenge"), "untouched")
    acceptance = {"evidence_id": EID, "run_id": RUN, "outcome": outcome, "trajectory_decision": decision,
        "checks": result["checks"], "roles": [], "costs": [cost], "role_use": role_use,
        "analysis_ref": f"{REL}/analysis.json", "execution_scope": "Synthetic checkpoint validation/atomic serialization only, no model work",
        "cost_scope": "Worker/process CPU time excludes imports; GPU-phase wall includes setup/IO, not exclusive GPU busy time"}
    atomic_json(here / "acceptance.json", acceptance)
    retained = ["acceptance.json", "analysis.json", "terminal_decision.md", "attribution.md",
        "results/result.json", "results/completion.json", "config.json", "protocol.md", "role_plan.json"]
    hashes = {f"{REL}/{name}": sha256_file(here / name) for name in retained}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EID, "track": "B", "scope": "synthetic_checkpoint_execution",
        "legacy_contract": "family-checkpoint-efficiency-v1", "outcome": outcome,
        "authority": {"pointers": [f"{REL}/{name}" for name in ("protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json")]},
        "role_use": role_use, "migration": {"migrated_at": datetime.now(timezone.utc).date().isoformat(),
            "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "Metadata acceptance of separately recorded synthetic checkpoint timing"},
        "observed_execution": {"training_executed": False, "inference_executed": False, "execution_ref": f"{REL}/results/result.json"},
        "artifacts": [{"name": name, "locator": f"{REL}/{name}", "sha256": hashes[f"{REL}/{name}"],
            "availability": "locally_retained_hash_verified"} for name in retained]}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": TID, "run_id": RUN,
        "action_id": "A001", "finalized_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_ref": f"{REL}/acceptance.json", "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "costs": [cost], "roles": []}
    atomic_json(here / "terminal.json", terminal)
    print(json.dumps(finalize(root, f"{REL}/rml", f"{REL}/terminal.json")))


if __name__ == "__main__":
    main()
