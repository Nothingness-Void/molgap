"""Accept measured diagnostic metadata through the existing V5/RML finalizer."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from molgap.input_pipeline_diagnostic import index_digest
from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file

REL = "experiments/pcqm_input_pipeline"
TID = "TB-input-pipeline-local-20261013"
RUN = "input-pipeline-local-20261013"
EID = "pcqm-input-pipeline-local-20261013"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path(__file__).resolve().parents[2]
    here = root / REL
    result = read(here / "results/result.json")
    config = read(here / "config.json")
    prospective = read(here / "rml/trajectory.json")
    if read(here / "results/completion.json")["result_sha256"] != sha256_file(here / "results/result.json"):
        raise ValueError("Completed result SHA differs")
    if result["config_sha256"] != sha256_file(here / "config.json") or (
            prospective["state_at_start"]["source_config_identity"] != canonical_fingerprint(config)):
        raise ValueError("Prospective/config identity differs")
    if result["status"] != "COMPLETE" or result["quality_evaluated"] or (
            result["development_rows_consumed"] or result["protected_roles_consumed"]):
        raise ValueError("Scope or completion differs")
    if result["runtime"]["gpu"] != config["accelerator"]:
        raise ValueError("Observed accelerator differs")
    if len(result["cpu"]) != 3 * len(config["cpu_workers"]) or len(result["gpu"]) != (
            3 * len(config["gpu_families"]) * config["gpu_repeats"]):
        raise ValueError("Incomplete diagnostic cases")
    memberships = {row["membership_sha256"] for row in result["cpu"]}
    if len(memberships) != 1:
        raise ValueError("CPU membership changed across strategies")
    expected_cpu = {(workers, strategy) for workers in config["cpu_workers"]
                    for strategy in ("global", "block", "bucket")}
    if {(row["workers"], row["strategy"]) for row in result["cpu"]} != expected_cpu:
        raise ValueError("CPU case identities differ")
    expected_gpu = {(family, strategy, repeat) for family in config["gpu_families"]
                    for strategy in ("global", "block", "bucket")
                    for repeat in range(config["gpu_repeats"])}
    if {(row["family"], row["strategy"], row["repeat"]) for row in result["gpu"]} != expected_gpu:
        raise ValueError("GPU case identities differ")
    for row in result["cpu"]:
        if row["samples"] != config["train_rows"] // 128 * 128 or len(row["traversals"]) != config["cpu_repeats"]:
            raise ValueError("CPU sample/pass count differs")
        if any(not math.isfinite(item["epoch_seconds"]) or item["epoch_seconds"] <= 0
               or item["samples"] != row["samples"] for item in row["traversals"]):
            raise ValueError("Invalid CPU traversal")
    for family in config["gpu_families"]:
        if len({row["initialization_sha256"] for row in result["gpu"] if row["family"] == family}) != 1:
            raise ValueError("GPU initialization changed within a family")
    for row in result["gpu"]:
        if row["samples"] != config["gpu_rows"] or row["steps"] != config["gpu_rows"] // 128:
            raise ValueError("Measured GPU exposure differs")
        if not math.isfinite(row["pipeline_seconds"]) or row["pipeline_seconds"] <= 0:
            raise ValueError("Invalid timing")
    analysis = {"cpu": result["cpu"], "gpu": {}, "quality_evaluated": False,
                "accuracy_causality": "insufficient_evidence"}
    for family in config["gpu_families"]:
        values = {}
        for strategy in ("global", "block", "bucket"):
            rows = [row for row in result["gpu"] if row["family"] == family and row["strategy"] == strategy]
            seconds = [row["pipeline_seconds"] for row in rows]
            values[strategy] = {"median_pipeline_seconds": statistics.median(seconds),
                "range_seconds": [min(seconds), max(seconds)],
                "median_molecules_per_second": statistics.median(row["molecules_per_second"] for row in rows),
                "median_loader_wait_ms": statistics.median(row["loader_wait_mean_ms"] for row in rows),
                "median_cuda_step_span_ms": statistics.median(row["cuda_step_span_mean_ms"] for row in rows),
                "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows)}
        baseline = values["global"]["median_pipeline_seconds"]
        for strategy, observation in values.items():
            observation["median_time_reduction_fraction"] = 1 - observation["median_pipeline_seconds"] / baseline
        analysis["gpu"][family] = values
    atomic_json(here / "analysis.json", analysis)
    decision = {"outcome": "NO_TRAIN", "decision_ref": f"{REL}/terminal_decision.md",
        "next_allowed_actions": [], "reopen_conditions": ["Separately authorized paired quality test or native platform profile"]}
    outcome = {"execution_status": "complete_execution_diagnostic", "artifact_status": "local_hash_verified",
        "comparison_status": "matched_membership_scratch_execution", "scientific_status": "NO_TRAIN",
        "transfer_status": "not_evaluated", "budget_decision": "bounded_local_diagnostic_complete",
        "full_handoff_status": "not_applicable"}
    roles = []
    for name, digest, kinds in (
            ("train_decoded", index_digest(range(config["train_rows"])), ("labels_read",)),
            ("train_probe", result["gpu_source_idx_sha256"], ("labels_read", "prediction_input"))):
        for access in kinds:
            roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-input-pipeline-{name}-{access}-20261013",
                "trajectory_id": TID, "action_id": "A001", "run_id": RUN,
                "dataset_identity": config["manifest_sha256"], "row_manifest_hash": digest,
                "role_name": name, "access_kind": access, "selection_used": False,
                "evidence_ref": f"{REL}/acceptance.json"})
    costs = [{"schema": "molgap-cost-event-v1", "cost_event_id": "cost-input-pipeline-worker-20261013",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "local-001",
        "platform": "local-windows", "hardware": config["accelerator"], "category": "audit",
        "evidence_ref": f"{REL}/acceptance.json", "measurement": {
            "wall_hours": {"value": result["cost"]["worker_wall_seconds"] / 3600, "status": "measured"},
            "device_hours": {"value": result["gpu_allocation_worker_seconds"] / 3600, "status": "measured"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "not_applicable"}}}]
    role_use = {"train_decoded": "labels_read", "train_probe": "prediction_input", "development": "untouched",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    acceptance = {"evidence_id": EID, "run_id": RUN, "outcome": outcome, "trajectory_decision": decision,
        "checks": {"completed_result_sha": True, "prospective_config": True, "case_count": True,
            "matched_cpu_membership": True, "matched_gpu_exposure": True, "finite_timing": True,
            "no_protected_roles": True}, "roles": roles, "costs": costs, "role_use": role_use,
        "analysis_ref": f"{REL}/analysis.json", "limitations": result["limits"],
        "cost_scope": "Worker wall excludes Python imports; GPU phase wall includes setup/cleanup, not exclusive busy time. Parent CPU is retained in result; child CPU and combined native CPU hours unknown.",
        "execution_scope": "Discarded scratch updates only, no scientific training endpoint, development MAE or retained predictor"}
    atomic_json(here / "acceptance.json", acceptance)
    retained = ["acceptance.json", "analysis.json", "terminal_decision.md", "attribution.md",
                "results/result.json", "results/completion.json", "config.json", "protocol.md", "role_plan.json"]
    hashes = {f"{REL}/{name}": sha256_file(here / name) for name in retained}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EID, "track": "B", "scope": "desktop_input_pipeline_execution_diagnostic",
        "legacy_contract": "pcqm-input-pipeline-local-v1", "outcome": outcome,
        "authority": {"pointers": [f"{REL}/{name}" for name in ("protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json")]},
        "role_use": role_use, "migration": {"migrated_at": "2026-10-13", "training_executed": False,
            "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "Accept separately executed prospective scratch profiling, no model execution in finalizer"},
        "observed_execution": {"training_executed": False, "inference_executed": True,
            "execution_ref": f"{REL}/results/result.json"},
        "artifacts": [{"name": name, "locator": f"{REL}/{name}", "sha256": hashes[f"{REL}/{name}"],
            "availability": "locally_retained_hash_verified"} for name in retained]}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": TID, "run_id": RUN,
        "action_id": "A001", "finalized_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_ref": f"{REL}/acceptance.json", "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "costs": costs, "roles": roles}
    atomic_json(here / "terminal.json", terminal)
    print(json.dumps(finalize(root, f"{REL}/rml", f"{REL}/terminal.json")))


if __name__ == "__main__":
    main()
