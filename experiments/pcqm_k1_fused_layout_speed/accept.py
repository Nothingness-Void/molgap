"""Accept saved speed metadata using the existing canonical trace/finalizer."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from molgap.research_memory.finalize import finalize
from molgap.research_memory.trace import atomic_write, canonicalize_trace, json_bytes
from molgap.training_reproducibility import atomic_json, sha256_file, assert_finite_state_dict
from molgap.v4_runtime import torch_load_compat

REL = "experiments/pcqm_k1_fused_layout_speed"
RUN = "k1-fused-layout-speed-100k-20261011"
ARMS = ("reference", "fused_layout")


def inspect_rows(rows):
    if len(rows) != 15620 or {r["arm"] for r in rows} != set(ARMS):
        raise ValueError("Expected exactly ten complete matched epochs")
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        if [(r["epoch"], r["batch"]) for r in selected] != [(e, b) for e in range(10) for b in range(781)]:
            raise ValueError("Missing, duplicate or reordered speed observations")
        if any(not all(math.isfinite(r[k]) for k in ("step_seconds", "pipeline_seconds", "learning_rate"))
               or r["samples"] != 128 or r["step_seconds"] <= 0
               or r["pipeline_seconds"] < r["step_seconds"] for r in selected):
            raise ValueError("Invalid exposure/timing")
        for row in selected:
            lr = 1e-6 + (4e-4 - 1e-6) * (1 + math.cos(math.pi * row["epoch"] / 40)) / 2
            if abs(row["learning_rate"] - lr) > 1e-15:
                raise ValueError("Compressed or changed learning-rate schedule")
    return True


def accept(root):
    folder = root / REL
    release = json.loads((folder / "release.json").read_text())
    stage = Path(release["config"]).parent
    output = Path(release["output"])
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] != "COMPLETE_SPEED_PREFIX" or summary["completed_epochs"] != 10 or summary["batch_offset"] != 0:
        raise ValueError("This acceptance requires the completed10ep prefix, not a cost-stop substitute")
    if sha256_file(Path(release["config"])) != release["config_sha256"] or summary["config_sha256"] != release["config_sha256"]:
        raise ValueError("Config identity changed")
    if sha256_file(output / "last.pt") != summary["checkpoint_sha256"]:
        raise ValueError("Atomic terminal checkpoint changed")
    saved = torch_load_compat(output / "last.pt", map_location="cpu", weights_only=False)
    if saved["epoch"] != 10 or saved["offset"] != 0 or saved["config_sha256"] != release["config_sha256"]:
        raise ValueError("Checkpoint does not bind the completed prefix")
    for arm in ARMS:
        assert_finite_state_dict(saved["arms"][arm]["model"], label=arm)
        if saved["arms"][arm]["scheduler"]["last_epoch"] != 10:
            raise ValueError("Scheduler exposure changed")
        if any(int(s["step"]) != 7810 for s in saved["arms"][arm]["optimizer"]["state"].values()):
            raise ValueError("Optimizer step exposure changed")
    rows = json.loads((output / "timings.json").read_text())["rows"]
    inspect_rows(rows)
    from molgap.k1_local_speed import summarize
    if summary["results"] != summarize(rows):
        raise ValueError("Summary does not reproduce raw timings")
    runtime = json.loads((output / "runtime.json").read_text())
    if (runtime["determinism"]["tf32_enabled"] or runtime["determinism"]["precision"] != "fp32"
            or "RTX 5060" not in runtime["accelerator"]["name"]):
        raise ValueError("Runtime differs from the authorized local FP32 release")
    cpu = json.loads((output / "cpu_acceptance.json").read_text())
    if cpu["development_decoded"] or cpu["cuda_initialized"] or cpu["train_rows"] != 100000:
        raise ValueError("TRAIN-only CPU release failed")
    allocation = json.loads((output / "allocation_terminal.json").read_text())
    if allocation["wall_seconds"] >= 3600:
        raise ValueError("Total allocation ceiling exceeded")
    results = folder / "results"
    results.mkdir(exist_ok=True)
    for name in ("summary.json", "timings.json", "cpu_acceptance.json", "runtime.json", "allocation.json", "allocation_terminal.json"):
        shutil.copyfile(output / name, results / name)
    shutil.copyfile(stage / "config.json", results / "executed_config.json")
    shutil.copyfile(stage / "SOURCE_FILES.json", results / "SOURCE_FILES.json")
    # Keep the actual executable archive and checkpoint in ignored custody;
    # committed receipts retain exact identities and locations, not a fake clone copy.
    atomic_json(results / "retained_artifacts.json", {
        "checkpoint": {"path": str(output / "last.pt"), "sha256": summary["checkpoint_sha256"]},
        "source_archive": release["source"], "ignored_local_custody": True,
        "checkpoint_resume_qualification": "not_performed_for_this_adapter",
    })
    receipt = json.loads((folder / "process_receipt.json").read_text(encoding="utf-8-sig"))
    native = f"local:{receipt['computer_name']}:pid{receipt['process_id']}:{receipt['created_at_utc']}"
    now = datetime.now(timezone.utc).isoformat()
    evidence_ref = f"{REL}/terminal_decision.md"
    receipts = []
    for arm in ARMS:
        arm_folder = folder / arm
        frozen = json.loads((arm_folder / "rml/trajectory.json").read_text())
        binding = frozen["state_at_start"]["same_run_replay"]
        tid = frozen["trajectory_id"]
        eid = f"pcqm-k1-local-speed-10ep-{arm}-20261011"
        trace = {"trajectory_id": tid, "run_id": RUN, "observations": [],
            "provenance": {"source_ref": f"{REL}/results/timings.json",
                "wall_scope": "Sum of synchronized arm pipeline windows; excludes shared loader/checkpoint",
                "development_evaluated": False}}
        cumulative = 0.0
        for epoch in range(10):
            selected = [r for r in rows if r["arm"] == arm and r["epoch"] == epoch]
            interval = sum(r["pipeline_seconds"] for r in selected)
            cumulative += interval
            trace["observations"].append({"event": "terminal" if epoch == 9 else "observation",
                "optimizer_step": (epoch + 1) * 781, "sample_presentations": (epoch + 1) * 99968,
                "epoch_or_pass": epoch + 1, "learning_rate": selected[0]["learning_rate"],
                "wall_time_seconds": interval, "cumulative_wall_time_seconds": cumulative,
                "checkpoint_identity": summary["checkpoint_sha256"] if epoch == 9 else None})
        trace_path = arm_folder / "trace.json"
        atomic_write(trace_path, json_bytes(canonicalize_trace(trace)))
        role_use = {"train": "training_membership", "development": "untouched",
            "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
        outcome = {"execution_status": "complete", "artifact_status": "local_hash_verified",
            "comparison_status": "matched_speed_prefix_only", "scientific_status": "INCONCLUSIVE",
            "transfer_status": "not_evaluated", "budget_decision": "authorized10ep_complete",
            "full_handoff_status": "not_applicable"}
        decision = {"outcome": "INFRASTRUCTURE_ONLY", "decision_ref": evidence_ref,
            "next_allowed_actions": [], "reopen_conditions": ["Separately authorized quality/native qualification"]}
        acceptance_ref = f"{REL}/{arm}/acceptance.json"
        cost = {"schema": "molgap-cost-event-v1", "cost_event_id": f"cost-k1-speed-{arm}-measured-20261011",
            "trajectory_id": tid, "action_id": "A001", "run_id": RUN, "attempt_id": "local-attempt001",
            "platform": "local", "hardware": "NVIDIA GeForce RTX5060", "category": "training",
            "evidence_ref": acceptance_ref,
            "measurement": {"device_hours": {"value": None, "status": "measurement_missing" if arm == "reference" else "not_applicable"},
                "wall_hours": {"value": allocation["wall_seconds"] / 3600 if arm == "reference" else None,
                    "status": "measured" if arm == "reference" else "not_applicable"},
                "cpu_hours": {"value": None, "status": "measurement_missing" if arm == "reference" else "not_applicable"},
                "queue_hours": {"value": None, "status": "not_applicable"}}}
        roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-k1-speed-{arm}-{kind}-20261011",
            "trajectory_id": tid, "action_id": "A001", "run_id": RUN,
            "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1:TRAIN[0,100000)",
            "row_manifest_hash": cpu["manifest_sha256"], "role_name": "train", "access_kind": kind,
            "selection_used": False, "evidence_ref": acceptance_ref}
            for kind in ("training_membership", "labels_read", "metric_computed")]
        atomic_json(root / acceptance_ref, {"evidence_id": eid, "run_id": RUN, "outcome": outcome,
            "trajectory_decision": decision, "role_use": role_use, "roles": roles, "costs": [cost],
            "cost_scope": "Reference wall event owns the shared job window once; candidate adds no second allocation. Per-arm pipeline timings are analysis windows, not additive allocation costs",
            "exact_device_release_time": "unknown", "quality_evaluated": False})
        pointers = [f"{REL}/protocol.md", f"{REL}/inputs.json", f"{REL}/role_plan.json",
            f"{REL}/release.json", f"{REL}/process_receipt.json", evidence_ref,
            f"{REL}/attribution.md", acceptance_ref, f"{REL}/{arm}/trace.json"]
        pointers += [p.relative_to(root).as_posix() for p in results.glob("*.json")]
        hashes = {p: sha256_file(root / p) for p in pointers}
        identity = {"scientific_contract": "local10ep-speed-only", "dataset_identity": "fixed100k-v1",
            "row_split_identity": "TRAIN[0,100000)", "architecture_identity": "K1-ES-GPS9-192-RWSE16",
            "optimizer_identity": "AdamW-foreachFalse-fused" + str(arm != "reference"),
            "lr_schedule_identity": "cosine40-first10", "target_transform_identity": "frozen-100k-mean-std",
            "precision_identity": "FP32-TF32off", "ema_semantics": "none",
            "evaluation_role_identity": "not_evaluated", "selection_role_identity": "none",
            "x_axis_semantics": "optimizer_steps", "terminal_endpoint_identity": "10ep-speed-prefix",
            "matched_architecture_required": True}
        manifest = {"schema": "molgap-trace-manifest-v1", "trajectory_id": tid, "run_id": RUN,
            "reference_id": "pcqm-k1-local-speed-10ep-reference-20261011", "comparison_role": binding["comparison_role"],
            "contract_ref": f"{REL}/protocol.md", "model_identity": "K1-ES-GPS9-192",
            "x_axis": "optimizer_steps", "presentation_semantics_ref": f"{REL}/protocol.md",
            "weight_semantics": "live", "metric_semantics": "No retained quality metrics; speed-only TRAIN loss guards",
            "evaluation_role_identity": "not_evaluated", "selection_semantics": "none",
            "trace_artifact_ref": f"{REL}/{arm}/trace.json", "terminal_evidence_ref": f"{REL}/{arm}/rml/rml_finalized/v5_evidence.json",
            "exposure": {"optimizer_steps": 7810, "sample_presentations": 999680}, "comparability_identity": identity,
            "backtest_eligibility": {"eligible": False, "exclusion_reasons": ["No scientific endpoint/development evaluation", "Fused rounding differs", "Local adapter resume equivalence not qualified"]}}
        terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid,
            "run_id": RUN, "action_id": "A001", "finalized_at": now, "acceptance_ref": acceptance_ref,
            "artifact_hashes": hashes, "decision": decision, "costs": [cost], "roles": roles,
            "trace_manifest": manifest, "same_run_observation": {
                "schema": "molgap-same-run-observation-v1", "spec_identity": binding["spec_identity"],
                "logical_run_id": RUN, "platform_name": "local", "platform_run_reference": native,
                "attempt_id": "local-attempt001", "source_commit": summary["source_commit"],
                "source_package_sha256": release["source"]["archive_sha256"]},
            "evidence": {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
                "evidence_id": eid, "track": "B", "scope": "local_speed_prefix", "legacy_contract": "local10ep-speed-only",
                "outcome": outcome, "role_use": role_use, "authority": {"pointers": [evidence_ref, acceptance_ref, f"{REL}/protocol.md"]},
                "artifacts": [{"name": p.split("/")[-1], "locator": p, "sha256": h,
                    "availability": "locally_retained_hash_verified"} for p, h in hashes.items()],
                "migration": {"migrated_at": now, "training_executed": False, "inference_executed": False,
                    "scientific_reinterpretation": False, "verification_scope": "Metadata acceptance of separately executed TRAIN speed prefix"},
                "observed_execution": {"training_executed": True, "inference_executed": False,
                    "training_replay_ready": False, "execution_ref": f"{REL}/results/summary.json"}}}
        path = arm_folder / "terminal.json"
        atomic_json(path, terminal)
        receipts.append(finalize(root, arm_folder / "rml/trajectory.json", path, trace_path))
    atomic_json(folder / "closure_receipts.json", {"results": receipts})
    print(json.dumps({"status": "finalized_speed_only", "results": receipts}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    accept(parser.parse_args().repo_root.resolve())
