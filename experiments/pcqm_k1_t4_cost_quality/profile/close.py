"""Metadata-only native T4 acceptance; publication requires explicit --execute."""
import argparse
import hashlib
import math
from pathlib import Path
import statistics
import struct
import json

from molgap.evidence_pointers import load_json_object, resolve_repo_pointer, verify_bound_artifact
from molgap.constants import REPO_ROOT
from molgap.k1_execution_profile import T4_CASES, verify_t4_payload
from molgap.research_memory.finalize import finalize
from molgap.research_memory.schemas import validate_cost_event, validate_role_event, validate_trajectory
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v5_common import validate_v5_evidence_envelope

ROOT = REPO_ROOT
REL = "experiments/pcqm_k1_t4_cost_quality/profile"
STAGING = "platforms/_records/kaggle/staging/k1-native-t4-profile-20261009"
RESULTS = "platforms/_records/kaggle/training/k1_native_t4_profile_s42_v1/profile"
MANIFEST_SHA = "d06f5ecdb4e399a9ed55bae081d98160cf90bef7bcecc6b4a984404d1dfd75be"
TID = "TB-k1-native-t4-cost-profile-20261009"
RUN = "molgap-k1-native-t4-profile-s42-v1"
KERNEL = "nvoid912/" + RUN
EID = "pcqm-k1-native-t4-cost-profile-20261009"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def accept(root=ROOT):
    """Verify retained bytes without deserializing a checkpoint or writing files."""
    root = Path(root).resolve()
    bindings = {}

    def path(pointer):
        resolved = resolve_repo_pointer(root, pointer)
        require(resolved is not None, "Local retained evidence required")
        bindings[pointer] = sha256_file(resolved)
        return resolved

    def read(pointer):
        return load_json_object(path(pointer))

    plan = read(REL + "/payload_plan.json")
    publication = read(REL + "/publication_binding.json")
    require(publication["payload_manifest_sha256"] == MANIFEST_SHA, "Publication manifest differs")
    require(publication["files"]["source_dataset/payload_manifest.json"] == MANIFEST_SHA,
            "Published manifest differs")
    require(publication["requested_kernel"] == KERNEL, "Publication kernel differs")
    for name, digest in publication["files"].items():
        pointer = STAGING + "/publication/" + name
        verify_bound_artifact(root, pointer, digest)
        path(pointer)
    manifest_path = path(STAGING + "/payload/payload_manifest.json")
    manifest = verify_t4_payload(manifest_path.parent, MANIFEST_SHA)
    for name, digest in manifest["files"].items():
        pointer = STAGING + "/payload/" + name
        verify_bound_artifact(root, pointer, digest)
        path(pointer)
    require(manifest["checkpoint"] == {k: plan["checkpoint"][k] for k in ("sha256", "source_sha256")},
            "Selected checkpoint differs")
    require(manifest["files"]["train_probe.pt"] == plan["train_probe"]["sha256"], "Sample bytes differ")
    require(manifest["sample_source_idx"] == plan["sample_source_idx"], "Sample order differs")
    require(plan["role"] == "official-train-derived-500k" and plan["geometry_used"] is False,
            "Preparation role differs")
    require(manifest["accepted_profile_manifest_sha256"] == plan["accepted_profile_manifest"]["sha256"],
            "Accepted sample provenance differs")
    require(manifest["source_input_commit"] == plan["source_input_commit"] == publication["source_commit"],
            "Source commit differs")
    require(set(manifest["source_files"]) == set(plan["source_files"]), "Source allowlist differs")
    require(all(manifest["files"][n] == d for n, d in plan["source_files"].items()), "Frozen sources differ")
    for name, pin in plan["packaging_files"].items():
        require(manifest["files"][name] == pin["sha256"], "Packaging pin differs")
    trajectory_pointer = REL + "/rml/trajectory.json"
    trajectory = validate_trajectory(read(trajectory_pointer))
    verify_bound_artifact(root, trajectory_pointer, plan["trajectory"]["sha256"])
    # The finalizer owns the live trajectory; terminal bindings retain its
    # immutable payload snapshot instead of a mutable canonical input pointer.
    del bindings[trajectory_pointer]
    require(manifest["files"]["prospective/trajectory.json"] == plan["trajectory"]["sha256"],
            "Prospective payload differs")
    receipt = read(REL + "/plan_receipt.json")
    verify_bound_artifact(root, REL + "/plan_receipt.json", plan["plan_receipt"]["sha256"])
    require(manifest["files"]["prospective/plan_receipt.json"] == plan["plan_receipt"]["sha256"],
            "Plan receipt payload differs")
    require(receipt["status"] == "PLANNED" and receipt["trajectory_id"] == TID
            and trajectory["trajectory_id"] == TID and trajectory["owner"] == "desktop"
            and trajectory["record_mode"] == "prospective" and trajectory["decision"]["outcome"] == "ACTIVE",
            "Prospective identity differs")
    submission = read(REL + "/submission_response.json")
    require(submission["status"] == "submitted" and submission["kernel"] == KERNEL
            and submission["kernel_id"] == 137849502 and submission["version_number"] == 1
            and not submission["identity_conflicts"], "Physical job binding differs")
    scheduler = read(REL + "/scheduler_reconciliation_complete.json")
    require(scheduler["kernel"] == KERNEL and scheduler["version_number"] == 1
            and scheduler["state"]["status"] == "COMPLETE"
            and scheduler["state"].get("failureMessage") is None, "Scheduler completion/identity differs")
    completion = read(RESULTS + "/completion.json")
    expected = {"single_w2.json", "double_mean2_w2.json", "runtime.json", "result.json",
                "phase_timings.json", "sample_timings.json", "scratch_publication_probe.pt"}
    require(completion["status"] == "complete" and set(completion["artifacts"]) == expected,
            "Completion inventory differs")
    for name, digest in completion["artifacts"].items():
        verify_bound_artifact(root, RESULTS + "/" + name, digest)
        path(RESULTS + "/" + name)
    result, runtime = read(RESULTS + "/result.json"), read(RESULTS + "/runtime.json")
    entry, process = read(RESULTS + "/entry_observation.json"), read(RESULTS + "/worker_process_observation.json")
    require(result["format"] == "molgap-k1-native-t4-profile-v1" and result["status"] == "complete"
            and result["scientific_outcome"] == "NO_TRAIN" and result["development_rows"] == []
            and result["protected_roles"] == "untouched" and result["accuracy_acceptance"] is False
            and result["training_replay_ready"] is False and result["training_rows"] == plan["sample_source_idx"],
            "Diagnostic scope differs")
    require(runtime["payload_sha256"] == MANIFEST_SHA and runtime["torch"] == "2.11.0+cu128"
            and runtime["cuda"] == "12.8" and runtime["visible_gpu_count"] == 2
            and runtime["gpu"] == "Tesla T4" and runtime["gpu_names"] == ["Tesla T4"] * 2
            and runtime["active_gpu_indices"] == [0] and runtime["scientific_training"] is False,
            "Native runtime differs")
    require(runtime["determinism"] == {"seed": 42, "precision": "fp32", "tf32_enabled": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8", "float32_matmul_precision": "highest"},
            "Determinism differs")
    pins = publication["pins"]
    for field, pin in (("payload_manifest_sha256", "EXPECTED_PROFILE_MANIFEST_SHA256"),
                       ("setup_sha256", "EXPECTED_SETUP_SHA256"),
                       ("source_archive_sha256", "EXPECTED_PROFILE_ARCHIVE_SHA256"),
                       ("unpack_sha256", "EXPECTED_UNPACK_SHA256")):
        require(entry[field] == pins[pin], "Executed publication pin differs")
    require(pins["EXPECTED_PROFILE_MANIFEST_SHA256"] == MANIFEST_SHA
            and pins["EXPECTED_SETUP_SHA256"] == manifest["files"]["setup.sh"]
            and pins["EXPECTED_PROFILE_ARCHIVE_SHA256"] == publication["files"]["source_dataset/source_payload.bin"]
            and pins["EXPECTED_UNPACK_SHA256"] == publication["files"]["source_dataset/unpack.py"],
            "Publication pins inconsistent")
    wall = entry["entry_wall_seconds_including_verify_setup_bootstrap"]
    require(entry["status"] == process["status"] == "complete" and entry["scientific_outcome"] == "NO_TRAIN"
            and entry["observed_allocated_gpu_count"] == 2 and entry["observed_gpu_names"] == ["Tesla T4"] * 2
            and entry["allocation_ceiling_seconds"] == 1200 and positive(wall) and wall <= 1200
            and math.isclose(entry["allocated_T4_device_hours"], wall * 2 / 3600), "Entry allocation differs")
    worker = result["cost"]["worker_wall_seconds"]
    child = process["child_wall_seconds_including_imports_load_teardown"]
    require(process["worker_ceiling_seconds"] == 900 and process["visible_gpu_count"] == 2
            and positive(worker) and worker <= child <= 900
            and child <= entry["bootstrap_wall_seconds_including_setup"] <= wall
            and math.isclose(process["allocated_T4_device_hours"], child * 2 / 3600)
            and result["cost"]["visible_gpu_count"] == 2
            and math.isclose(result["cost"]["allocated_T4_device_hours"], worker * 2 / 3600),
            "Worker cost/bound differs")
    require(all(x["gpu_busy_seconds"] is None for x in (entry, process, result["cost"]))
            and result["cost"]["global_cost_extrapolation"] is False, "Unmeasured cost claim")
    require(len(result["cases"]) == 2, "Case inventory differs")
    medians = {}
    for row, (name, passes, workers) in zip(result["cases"], T4_CASES):
        require(row == read(RESULTS + "/" + name + ".json") and row["case"] == name
                and (row["passes"], row["workers"]) == (passes, workers) and len(row["steps"]) == 24,
                "Measured case differs")
        for step in row["steps"]:
            require(all(positive(step[k]) for k in ("step_s", "loader_wait_s", "transfer_compute_s"))
                    and math.isclose(step["step_s"], step["loader_wait_s"] + step["transfer_compute_s"]),
                    "Invalid measured timing")
        medians[name] = statistics.median(s["step_s"] for s in row["steps"])
        require(medians[name] == row["median_step_s"], "Median differs")
    phase = read(RESULTS + "/phase_timings.json")
    require(phase["samples"] == result["phase_samples"] and len(phase["samples"]) == 6
            and phase["synchronized_instrumentation"] is True and phase["warmup_steps"] == 2
            and phase["optimizer_steps"] == result["phase_optimizer_steps"] == 8
            and phase["consistency_coefficient"] == result["consistency_coefficient"] == 0
            and result["scratch_optimizer_steps_per_case"] == 29 and result["scratch_optimizer_steps_total"] == 66
            and result["gradient_relation_probe_performed"] is False and result["gradient_norm_probe"] is None,
            "Scratch phase scope differs")
    keys = {"loader", "h2d", "forward_loss", "backward", "clip", "optimizer"}
    require(all(set(s) == keys and all(positive(v) for v in s.values()) for s in phase["samples"]),
            "Invalid phase timing")
    sample = read(RESULTS + "/sample_timings.json")
    require(sample == result["sample_timings"] and sample["eval_role"] == "train_members_only"
            and sample["eval_rows"] == 512 and sample["eval_optimizer_steps"] == 0
            and len(sample["eval_step_seconds"]) == 4 and all(positive(v) for v in sample["eval_step_seconds"])
            and positive(sample["localFS_atomic_checkpoint_write_seconds"])
            and sample["localFS_checkpoint_bytes"] == path(RESULTS + "/scratch_publication_probe.pt").stat().st_size
            and sample["full50k_development_seconds"] is None and sample["remote_upload_seconds"] is None,
            "Train evaluation/publication scope differs")
    for name in ("protocol.md", "inputs.json", "role_plan.json", "plan_input.json",
                 "terminal_decision.md", "attribution.md"):
        path(REL + "/" + name)
    analysis = {"median_step_seconds": medians,
                "single_step_saving_fraction": 1 - medians["single_w2"] / medians["double_mean2_w2"],
                "phase_mean_seconds": {k: statistics.mean(s[k] for s in phase["samples"]) for k in sorted(keys)},
                "sample_timings": sample, "worker_subset_seconds": worker,
                "full_entry_wall_seconds": wall, "allocated_T4_device_hours": wall * 2 / 3600,
                "scheduler_snapshot_status": scheduler["state"]["status"],
                "accuracy_causality": "insufficient_evidence", "global_cost_extrapolation": False}
    return {"analysis": analysis, "artifact_hashes": bindings, "rows": result["training_rows"]}


def close(root=ROOT, *, execute=False, finalized_at=None):
    report = accept(root)
    if not execute:
        return report["analysis"]
    require(bool(finalized_at), "Explicit immutable finalization timestamp required")
    root = Path(root).resolve()
    acceptance_ref = REL + "/acceptance.json"
    decision = {"outcome": "NO_TRAIN", "decision_ref": REL + "/terminal_decision.md",
                "next_allowed_actions": [], "reopen_conditions": ["Separate prospective cost-quality authority"]}
    outcome = {"execution_status": "complete_execution_diagnostic", "artifact_status": "local_hash_verified",
               "comparison_status": "matched_scratch_execution_counterfactual", "scientific_status": "NO_TRAIN",
               "transfer_status": "not_evaluated", "budget_decision": "bounded_native_t4_diagnostic_complete",
               "full_handoff_status": "not_applicable"}
    row_hash = hashlib.sha256(b"".join(struct.pack("<q", i) for i in report["rows"])).hexdigest()
    roles = [validate_role_event({"schema": "molgap-role-event-v1", "role_event_id": "role-k1-t4-profile-" + kind,
             "trajectory_id": TID, "action_id": "A001", "run_id": RUN,
             "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1", "row_manifest_hash": row_hash,
             "role_name": "train_probe", "access_kind": kind, "selection_used": False,
             "evidence_ref": acceptance_ref}) for kind in ("labels_read", "prediction_input")]
    analysis = report["analysis"]
    # Worker and supervisor are nested subsets, never additional cost events.
    cost = validate_cost_event({"schema": "molgap-cost-event-v1", "cost_event_id": "cost-k1-native-t4-profile-entry",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "attempt-001",
        "platform": "kaggle", "hardware": "Tesla T4 (2 allocated; cuda0 active)", "category": "other",
        "evidence_ref": acceptance_ref, "measurement": {
            "device_hours": {"value": analysis["allocated_T4_device_hours"], "status": "measured"},
            "wall_hours": {"value": analysis["full_entry_wall_seconds"] / 3600, "status": "measured"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "measurement_missing"}}})
    role_use = {"train_probe": "labels_read", "development": "untouched", "official_validation": "untouched",
                "test_dev": "untouched", "test_challenge": "untouched"}
    acceptance = {"evidence_id": EID, "run_id": RUN, "outcome": outcome, "trajectory_decision": decision,
        "role_use": role_use, "roles": roles, "costs": [cost], "analysis": analysis,
        "execution_scope": "66 discarded scratch updates;512 train-member eval rows; no scientific training/selection",
        "role_scope": "Prebuilt accepted sample reused; only4096 training labels decoded in this attempt, not500K",
        "cost_scope": "Full entry allocation once; worker subset not additive; GPU busy/queue/upload/full50K unknown",
        "scheduler_scope": "Parent-retained exact version COMPLETE snapshot; acceptance makes no remote query"}
    atomic_json(root / acceptance_ref, acceptance)
    hashes = report["artifact_hashes"] | {acceptance_ref: sha256_file(root / acceptance_ref)}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EID, "track": "B", "scope": "desktop_k1_native_t4_execution_diagnostic",
        "legacy_contract": "pcqm-k1-native-t4-cost-profile-v1", "outcome": outcome,
        "authority": {"pointers": [REL + "/" + n for n in ("protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json")]},
        "role_use": role_use, "migration": {"migrated_at": finalized_at, "training_executed": False,
            "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "Metadata/hash acceptance only; no model executed by finalizer"},
        "observed_execution": {"training_executed": False, "inference_executed": True,
                               "execution_ref": RESULTS + "/result.json"},
        "artifacts": [{"name": p, "locator": p, "sha256": d, "availability": "locally_retained_hash_verified"}
                      for p, d in sorted(hashes.items())]}
    validate_v5_evidence_envelope(evidence, repo_root=root)
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": TID, "run_id": RUN,
        "action_id": "A001", "finalized_at": finalized_at, "acceptance_ref": acceptance_ref,
        "artifact_hashes": hashes, "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles}
    atomic_json(root / REL / "terminal.json", terminal)
    return finalize(root, REL + "/rml", REL + "/terminal.json")


if __name__ == "__main__":
    require(Path(__file__).resolve() == (ROOT / REL / "close.py").resolve(),
            "Active wrapper/source checkout differs; set PYTHONPATH to this checkout/src")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalized-at", help="Parent-frozen ISO timestamp; required with --execute")
    args = parser.parse_args()
    print(json.dumps(close(execute=args.execute, finalized_at=args.finalized_at), sort_keys=True))
