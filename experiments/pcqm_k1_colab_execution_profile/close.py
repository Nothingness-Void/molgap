"""Accept retained execution artifacts and publish through the existing RML finalizer."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-colab-execution-profile-20261007"
RUN = "k1-profile-a100-20261007"
EID = "pcqm-k1-colab-execution-profile-20261007"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    result_dir = HERE / "results/attempt-002"
    result, runtime = read(result_dir / "result.json"), read(result_dir / "runtime.json")
    manifest = read(result_dir / "payload_manifest.json")
    if sha256_file(result_dir / "payload_manifest.json") != runtime["payload_sha256"]:
        raise ValueError("Executed payload binding differs")
    if manifest != read(HERE / "payload_manifest_attempt002.json"):
        raise ValueError("Remote/local frozen manifest differs")
    if manifest["files"]["prospective/trajectory.json"] != sha256_file(HERE / "rml/trajectory.json"):
        raise ValueError("Prospective binding differs")
    inputs = read(HERE / "inputs.json")
    if manifest["checkpoint"]["sha256"] != inputs["selected"]["sha256"]:
        raise ValueError("Selected checkpoint changed")
    completion = read(result_dir / "completion.json")
    for name, digest in completion["artifacts"].items():
        if Path(name).name != name or sha256_file(result_dir / name) != digest:
            raise ValueError("Finalized remote artifact differs")
    rows = result["training_rows"]
    if rows != inputs["sample_source_idx"] or len(set(rows)) != 4096 or not all(0 <= x < 500000 for x in rows):
        raise ValueError("Training sample differs")
    if result["status"] != "complete" or result["development_rows"] or result["training_replay_ready"]:
        raise ValueError("Diagnostic scope differs")
    if runtime["torch"] != "2.4.1+cu121" or runtime["gpu"] != "NVIDIA A100-SXM4-40GB":
        raise ValueError("Observed native runtime differs")
    settings = runtime["determinism"]
    if settings["precision"] != "fp32" or settings["tf32_enabled"] or not settings["deterministic_algorithms"]:
        raise ValueError("Frozen FP32 settings differ")
    expected = {"single_w2": (1, 2), "double_w0": (2, 0), "double_w2": (2, 2), "double_w4": (2, 4)}
    cases = {}
    for row in result["cases"]:
        detail = read(result_dir / (row["case"] + ".json"))
        if (row["passes"], row["workers"]) != expected[row["case"]] or len(detail["steps"]) != 24:
            raise ValueError("Measured execution case differs")
        if not all(math.isfinite(s["step_s"]) and s["step_s"] > 0 for s in detail["steps"]):
            raise ValueError("Invalid measured timing")
        if statistics.median(s["step_s"] for s in detail["steps"]) != row["median_step_s"]:
            raise ValueError("Median differs from observations")
        cases[row["case"]] = row["median_step_s"]
    if set(cases) != set(expected) or len(result["phase_samples"]) != 6:
        raise ValueError("Incomplete profile")
    gradients = result["gradient_relation"]["batches"]
    if len(gradients) != 4 or result["gradient_relation"]["optimizer_updates"] != 0:
        raise ValueError("Gradient diagnostic differs")
    if not all(math.isfinite(r[k]) for r in gradients for k in ("norm_ratio", "cosine")):
        raise ValueError("Nonfinite gradient diagnostic")
    process = read(result_dir / "worker_process_observation.json")
    if process["returncode"] != 0 or process["worker_wall_seconds_including_imports"] > 1200:
        raise ValueError("Worker completion/bound differs")
    phases = {k: statistics.mean(r[k] for r in result["phase_samples"]) for k in result["phase_samples"][0]}
    phase_sum = sum(phases.values())
    analysis = {"cases_seconds": cases, "phase_mean_seconds": phases,
        "phase_share": {k: v / phase_sum for k, v in phases.items()},
        "single_vs_double_reduction": 1 - cases["single_w2"] / cases["double_w2"],
        "double_over_single_overhead": cases["double_w2"] / cases["single_w2"] - 1,
        "workers0_vs2_overhead": cases["double_w0"] / cases["double_w2"] - 1,
        "gradient_norm_ratio_range": [min(r["norm_ratio"] for r in gradients), max(r["norm_ratio"] for r in gradients)],
        "cosines": [r["cosine"] for r in gradients], "accuracy_causality": "insufficient_evidence"}
    atomic_json(HERE / "analysis.json", analysis)
    outcome = {"execution_status": "complete_execution_diagnostic", "artifact_status": "local_hash_verified",
        "comparison_status": "matched_scratch_execution_counterfactual", "scientific_status": "NO_TRAIN",
        "transfer_status": "not_evaluated", "budget_decision": "bounded_a100_diagnostic_complete",
        "full_handoff_status": "not_applicable"}
    decision = {"outcome": "NO_TRAIN", "decision_ref": f"{REL}/terminal_decision.md",
        "next_allowed_actions": [], "reopen_conditions": ["Separately frozen native T4 execution or single-pass quality qualification"]}
    roles = []
    for role, row_ids, kinds in [("train_decoded", range(500000), ["labels_read"]),
                                ("train_probe", rows, ["labels_read", "prediction_input"])]:
        row_hash = hashlib.sha256(np.asarray(list(row_ids), dtype="<i8").tobytes()).hexdigest()
        for access in kinds:
            roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-k1-a100-profile-{role}-{access}",
                "trajectory_id": TID, "action_id": "A001", "run_id": RUN,
                "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1", "row_manifest_hash": row_hash,
                "role_name": role, "access_kind": access, "selection_used": False,
                "evidence_ref": f"{REL}/acceptance.json"})
    costs = []
    def cost(label, attempt, category, wall, device, cpu=None, hardware=None):
        values = {"wall_hours": wall, "device_hours": device, "cpu_hours": cpu, "queue_hours": None}
        costs.append({"schema": "molgap-cost-event-v1", "cost_event_id": "cost-k1-a100-profile-" + label,
            "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": attempt,
            "platform": "colab" if hardware is None else "local-windows",
            "hardware": hardware or runtime["gpu"], "category": category,
            "evidence_ref": f"{REL}/acceptance.json", "measurement": {k: {
                "value": v / 3600 if v is not None else None,
                "status": "measured" if v is not None else "measurement_missing"} for k, v in values.items()}})
    previous = HERE / "results/attempt-001"
    failed_seconds = read(previous / "worker_process_observation.json")["worker_wall_seconds_including_imports"]
    cost("failed-worker", "attempt-001", "infrastructure_failure", failed_seconds, failed_seconds)
    cost("worker", "attempt-002", "other", process["worker_wall_seconds_including_imports"],
         process["worker_wall_seconds_including_imports"], result["cost"]["parent_cpu_seconds"])
    for attempt, directory in [("attempt-001", previous), ("attempt-002", result_dir)]:
        cost(attempt + "-setup", attempt, "preflight", read(directory / "setup_observation.json")["setup_wall_seconds"], None)
    cost("idle-allocation", "shared-session", "other", None, None)
    cost("local-staging", "local-staging", "cache_build", None, None, hardware="CPU; no accelerator")
    role_use = {"train_decoded": "labels_read", "train_probe": "prediction_input", "development": "untouched",
                "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    acceptance = {"evidence_id": EID, "run_id": RUN, "outcome": outcome, "trajectory_decision": decision,
        "role_use": role_use, "roles": roles, "costs": costs, "analysis_ref": f"{REL}/analysis.json",
        "checks": {"frozen_payload": True, "prospective": True, "selected_checkpoint": True,
            "train_membership": True, "native_fp32_a100": True, "all_cases": True,
            "finite_diagnostics": True, "artifact_hashes": True, "runtime_released": True},
        "execution_scope": "127 discarded scratch optimizer steps; four selected-state gradient batches; no scientific training/selection",
        "cost_scope": "Worker device hours are allocated A100 wall time, not GPU busy or full session billing. Setup allocation, idle interval and staging costs unknown. Parent CPU excludes children.",
        "role_scope": "Local staging decoded ten train shards/500K backing labels; worker loaded4096 training members only. No development/protected role.",
        "limitations": result["limits"]}
    atomic_json(HERE / "acceptance.json", acceptance)
    paths = [p for p in HERE.rglob('*') if p.is_file() and not any(x in p.parts for x in ('__pycache__', 'rml', 'rml_finalized')) and p.name != 'terminal.json']
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    retained = ["acceptance.json", "analysis.json", "terminal_decision.md", "attribution.md",
                "results/attempt-002/result.json", "results/attempt-002/completion.json"]
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EID, "track": "B", "scope": "desktop_k1_a100_execution_diagnostic",
        "legacy_contract": "pcqm-k1-colab-execution-profile-v1", "outcome": outcome,
        "authority": {"pointers": [f"{REL}/{n}" for n in ["protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json"]]},
        "role_use": role_use, "migration": {"migrated_at": "2026-10-08", "training_executed": False,
            "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "Metadata acceptance of separately executed prospective scratch profiling; finalizer executes no model"},
        "observed_execution": {"training_executed": False, "inference_executed": True,
            "execution_ref": f"{REL}/results/attempt-002/result.json"},
        "artifacts": [{"name": n, "locator": f"{REL}/{n}", "sha256": hashes[f"{REL}/{n}"],
            "availability": "locally_retained_hash_verified"} for n in retained]}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": TID, "run_id": RUN,
        "action_id": "A001", "finalized_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_ref": f"{REL}/acceptance.json", "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "costs": costs, "roles": roles}
    atomic_json(HERE / "terminal.json", terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json")))


if __name__ == '__main__':
    main()
