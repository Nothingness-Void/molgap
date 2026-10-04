"""Saved-metadata qualification adapter over the shared NO_TRAIN RML closure."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .training_reproducibility import atomic_json, sha256_file
from .research_memory.terminal_wiring import close_terminal_arm

BASE = "experiments/pcqm_gptrans_scale_qualification"


def accept_and_close(root: Path, records: Path, package: Path):
    root = root.resolve()
    read = lambda p: json.loads(p.read_text())
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    folder, target = records / "gptrans_scale_qualification", root / BASE / "results"
    manifest, result, cost = (read(folder / n) for n in ("output_manifest.json", "qualification.json", "cost.json"))
    local = read(package / "staging.json")["package"]
    receipt = read(root / BASE / "submission_v1.json")
    contract = read(root / BASE / "contract.json")
    if receipt.get("kernel") != "kaseichou/molgap-gptrans-g1-scale-qualification" or receipt.get("version_number") != 1 or receipt.get("status") != "SUBMITTED":
        raise ValueError("Exact returned Kaggle identity required")
    for key in ("package_identity", "spec_identity", "source_commit"):
        if receipt["release_binding"][key] != local[key]:
            raise ValueError("Physical receipt release binding differs")
    if receipt["release_binding"]["source_archive_sha256"] != local["archive_sha256"]:
        raise ValueError("Physical receipt archive binding differs")
    if manifest["source_identity"] != local or result["source_identity"] != local:
        raise ValueError("Remote/local source binding differs")
    if manifest["status"] != "COMPLETE" or cost["status"] != "COMPLETE":
        raise ValueError("Only complete saved qualification output is accepted")
    for name, digest in manifest["files"].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or sha256_file(p) != digest:
            raise ValueError("Qualification artifact changed")
    for flag in ("development_role_read", "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read", "scientific_training_executed"):
        if result.get(flag) is not False or manifest.get(flag) is not False:
            raise ValueError("Qualification scope exceeded")
    for key in ("parameters", "physical_batch", "precision", "disposable_optimizer_updates"):
        if result[key] != contract[key]:
            raise ValueError(f"Qualification contract mismatch: {key}")
    if not result["model_inference_executed"] or result["deterministic_repeat"][0] != result["deterministic_repeat"][1]:
        raise ValueError("Optimizer-inclusive repeat proof missing")
    from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
    if result["manifest_sha256"] != FIXED_500K_MANIFEST_SHA256:
        raise ValueError("Dataset binding differs")
    for key in ("optimizer_graphs_per_second", "estimated_equal_exposure_training_hours", "memory_reserve_fraction", "worker_wall_seconds"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("Invalid measured qualification")
    expected = result["memory_reserve_fraction"] >= .15 and result["estimated_equal_exposure_training_hours"] <= contract["training_estimate_cap_hours"]
    if result["qualification_passed"] != expected or cost["allocated_devices"] not in (1, 2):
        raise ValueError("Qualification decision or allocation differs")
    if cost["allocation_wall_seconds"] > contract["maximum_wall_seconds"] + 30:
        raise ValueError("Qualification exceeded wall cap")
    if abs(cost["allocated_device_hours"] - cost["allocated_devices"] * cost["allocation_wall_seconds"] / 3600) > 1e-6:
        raise ValueError("Full allocated cost is inconsistent")
    target.mkdir(parents=True, exist_ok=True)
    acceptance = target / "acceptance.json"
    atomic_json(acceptance, {"accepted": True, "qualification_passed": expected, "model_inference_executed_locally": False,
        "scientific_training_executed": False, "remote_identity": receipt, "result": result, "native_cost": cost})
    decision = target / "decision.md"
    decision.write_text(f"# G1 scale execution qualification\n\nSaved artifacts passed mechanical acceptance; budget/runtime eligibility={expected}. "
        "This disposable train-role profile is CONTEXT_ONLY, not a scientific training or Replay pair. "
        "The evaluation proxy excludes actual dev distributions and checkpoint I/O. No long training was released.\n")
    trajectory_path = root / BASE / "rml_plan/trajectory.json"
    trajectory = read(trajectory_path)
    tid = trajectory["trajectory_id"]
    paths = [trajectory_path, root / BASE / "contract.json", root / BASE / "protocol.md",
        root / BASE / "release_binding.json", root / BASE / "submission_v1.json", acceptance, decision,
        *[folder / n for n in manifest["files"]], folder / "output_manifest.json"]
    timestamp = datetime.now(timezone.utc).isoformat()
    role_use = {"train": "profiling_only", "internal_development": "untouched", "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": "pcqm-gptrans-g1-scale500k-qualification-s42", "track": "C", "scope": "train_only_disposable_GPU_profile",
        "legacy_contract": "none-v5-prospective", "outcome": {"execution_status": "complete", "artifact_status": "accepted",
        "comparison_status": "context_only", "scientific_status": "not_evaluated", "transfer_status": "not_evaluated",
        "budget_decision": "stop_under_contract", "full_handoff_status": "not_authorized"}, "role_use": role_use,
        "authority": {"pointers": [rel(p) for p in paths]}, "artifacts": [{"name": p.name, "locator": rel(p), "sha256": sha256_file(p), "availability": "retained_metadata"} for p in paths],
        "migration": {"migrated_at": timestamp, "training_executed": False, "inference_executed": False,
        "scientific_reinterpretation": False, "verification_scope": "Saved metadata only locally; remote profiling included disposable optimizer updates and train-role inference"}}
    roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{kind}", "trajectory_id": tid,
        "action_id": "A001", "run_id": "gptrans-g1-scale-qualification-s42", "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
        "row_manifest_hash": FIXED_500K_MANIFEST_SHA256, "role_name": "train", "access_kind": kind,
        "selection_used": False, "evidence_ref": rel(acceptance)} for kind in ("training_membership", "labels_read", "metric_computed")]
    observed_cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-" + tid + "-observed", "trajectory_id": tid,
        "action_id": "A001", "run_id": "gptrans-g1-scale-qualification-s42", "attempt_id": "v1", "category": "preflight",
        "platform": "kaggle2", "hardware": "Tesla_T4", "evidence_ref": rel(acceptance), "measurement": {
        "wall_hours": {"status": "measured", "value": cost["allocation_wall_seconds"] / 3600},
        "device_hours": {"status": "measured", "value": cost["allocated_device_hours"]},
        "queue_hours": {"status": "measurement_missing", "value": None}, "cpu_hours": {"status": "measurement_missing", "value": None}}}
    terminal = target / "terminal.json"
    atomic_json(terminal, {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid,
        "run_id": "gptrans-g1-scale-qualification-s42", "action_id": "A001", "finalized_at": timestamp,
        "acceptance_ref": rel(acceptance), "artifact_hashes": {rel(p): sha256_file(p) for p in paths}, "evidence": evidence,
        "decision": {"final": True, "outcome": "NO_TRAIN", "decision_ref": rel(decision), "next_allowed_actions": ["Controller analysis; no automatic long training"], "reopen_conditions": ["Explicit separate scale release"]},
        "costs": [observed_cost], "roles": roles, "role_use": role_use})
    return close_terminal_arm(root, trajectory_path, terminal)
