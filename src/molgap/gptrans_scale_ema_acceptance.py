"""Saved-tensor acceptance for one500K stream; no model construction/inference.

The two EMA endpoints are aligned observations, not an invented accepted500K
reference. Terminal qualification remains noncausal under the frozen purpose.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .experiment_package import verify_experiment_source_package
from .gptrans_author_acceptance import _require
from .research_memory.trace import load_canonical_trace
from .training_reproducibility import atomic_json, sha256_file


BASE = "experiments/pcqm_gptrans_capacity_relations_100k"
FILTERS = {"ema9999": .9999, "ema999": .999}


def _retained_artifacts(paths, primary_trace, relative):
    """Retain both EMA views but bind one canonical physical-run trace."""
    artifacts = []
    for path in paths:
        kind = "supporting_evidence"
        if path.resolve() == primary_trace.resolve():
            kind = "training_trace"
        elif "trace" in path.name:
            # The auxiliary view and raw history are retained, not a second
            # canonical trace for the same one-optimizer RML transaction.
            kind = "analysis"
        artifacts.append({"name": path.name, "locator": relative(path),
            "sha256": sha256_file(path), "availability": "local_verified",
            "artifact_type": kind})
    return artifacts


def accept_outputs(root, records, package):
    import tarfile
    import torch
    from .k1_terminal_analysis import paired_saved_errors
    from .pcqm_gptrans_v4 import FrozenEpochScheduler
    from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
    root, records = Path(root).resolve(), Path(records).resolve()
    read = lambda path: json.loads(path.read_bytes())
    base = root / BASE / "gpu"
    config, receipt = read(base / "screen_config.json"), read(base / "submission_v1.json")
    frozen = verify_experiment_source_package(package)
    _require(receipt["status"] == "submitted" and not receipt["reconciliation_required"], "Physical submission unresolved")
    _require(receipt["requested_kernel"] == config["requested_kernel"] and receipt["version_number"] == 1, "Physical identity")
    for key in ("source_commit", "spec_identity", "package_identity"):
        _require(frozen[key] == receipt["release_binding"][key], "Source receipt: " + key)
    _require(frozen["archive_sha256"] == receipt["release_binding"]["source_archive_sha256"], "Source archive")
    with tarfile.open(Path(package) / "source.tar.gz", "r:gz") as archive:
        _require(json.loads(archive.extractfile(BASE + "/gpu/screen_config.json").read()) == config, "Frozen config changed")
    screen = records / config["output_subdirectory"]
    folder = screen / "scale_ema/training"
    startup, cost, summary = (read(screen / name) for name in ("startup.json", "native_cost.json", "job_summary.json"))
    result = read(folder / "completion_manifest.json")
    _require(startup["spec_identity"] == frozen["spec_identity"] and startup["source_archive_sha256"] == frozen["archive_sha256"], "Startup binding")
    _require(startup["arms"] == list(config["arms"]), "Declared physical arms")
    outcomes = [r for r in summary["outcomes"] if r["variant"] == "scale_ema"]
    _require(len(outcomes) == 1 and outcomes[0]["complete"] is True, "Scale arm not complete")
    _require(result["source_identity"] == frozen and result["configuration"] == config["scale_study"], "Scale source/config")
    _require(result["complete"] is True and result["live_optimizer_streams"] == 1 and result["parameters"] == 5246817,
        "Single unchanged live model")
    _require(result["manifest_sha256"] == FIXED_500K_MANIFEST_SHA256 and result["optimizer_steps"] == 46860
        and result["sample_presentations"] == 5998080, "Fixed data/exposure")
    _require(result["training_executed"] is True and result["development_role_read"] is True, "Observed role use")
    for key in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        _require(result[key] is False and startup[key] is False, "Protected role: " + key)
    for name, digest in result["files"].items():
        path = (folder / name).resolve()
        _require(path.is_relative_to(folder.resolve()) and path.is_file() and sha256_file(path) == digest,
            "Scale file binding: " + name)
    qualification = read(folder / "qualification.json")
    _require(qualification["qualification_passed"] is True and qualification["source_identity"] == frozen
        and qualification["deterministic_repeat"][0] == qualification["deterministic_repeat"][1]
        and qualification["precision"] == "fp32" and qualification["tf32_enabled"] is False
        and qualification["physical_batch"] == 128 and qualification["memory_reserve_fraction"] >= .15,
        "Runtime qualification")
    _require(cost["allocated_gpu_count"] == 2 and cost["used_gpu_count"] == 2
        and len(cost["allocated_gpu_inventory"]) == 2 and all("T4" in n for n in cost["allocated_gpu_inventory"]), "T4x2 allocation")
    _require(math.isfinite(cost["wall_seconds"]) and 0 < cost["wall_seconds"] <= config["maximum_wall_seconds"]
        and cost["wall_seconds"] == summary["elapsed_seconds"]
        and abs(cost["allocated_device_hours"] - cost["wall_seconds"] * 2 / 3600) < 1e-10, "Native allocation cost")
    tensors, traces = {}, {}
    for view in FILTERS:
        trace = load_canonical_trace(folder / view / "canonical_trace.json")
        rows = trace["observations"]
        _require(len(rows) == 60 and trace["trajectory_id"] == config["arms"]["scale_ema"]["trajectory_id"]
            and trace["run_id"] == config["scale_study"]["logical_run_id"], "Canonical physical trace")
        _require(all(r["optimizer_step"] == (i + 1) * 781 and r["sample_presentations"] == (i + 1) * 781 * 128
            and r["learning_rate"] == FrozenEpochScheduler.learning_rate(i) for i, r in enumerate(rows)), "Trace exposure/LR")
        best = min(range(60), key=lambda i: rows[i]["ema_dev_metric"])
        _require(best == result["best"][view]["rung"], "Frozen per-view selector")
        payload = torch.load(folder / view / "development_predictions.pt", map_location="cpu", weights_only=True)
        for flag in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
            _require(payload[flag] is False, "Prediction protected role")
        _require(torch.equal(payload["source_idx"], torch.arange(500000, 550000)), "Prediction row order")
        _require(all(len(payload[k]) == 50000 and bool(torch.isfinite(payload[k]).all()) for k in ("prediction_eV", "target_eV")), "Finite endpoints")
        mae = float((payload["prediction_eV"].double() - payload["target_eV"].double()).abs().mean())
        _require(abs(mae - result["best"][view]["mae_eV"]) < 1e-7
            and abs(mae - rows[best]["ema_dev_metric"]) < 1e-7, "Endpoint recomputation")
        tensors[view], traces[view] = payload, trace
    common = ("optimizer_step", "sample_presentations", "learning_rate", "live_train_metric", "live_dev_metric", "checkpoint_identity")
    _require(all(all(a[k] == b[k] for k in common) for a, b in zip(traces["ema9999"]["observations"], traces["ema999"]["observations"])),
        "EMA views must share the exact live trajectory")
    analysis = paired_saved_errors(tensors["ema9999"], tensors["ema999"])
    return {"format": "molgap-scale-ema-acceptance-v1", "accepted": True, "comparison_class": "PAIRED_ENDPOINT",
        "experiment_purpose": "transfer_study", "strict_ready": False, "replay_ready": False,
        "replay_exclusion": "no qualified500K causal reference; noncausal purpose is not STRICT_CAUSAL",
        "source_identity": frozen, "configuration": config["scale_study"], "result": result,
        "paired_analysis": analysis, "native_cost": cost, "model_inference_executed": False, "local_training_executed": False}


def close_outputs(root, records, acceptance):
    """Use the existing RML transaction; preserve noncausal qualification."""
    from .research_memory.terminal_wiring import close_terminal_arm, build_default_trace_manifest
    root, records, acceptance = Path(root).resolve(), Path(records).resolve(), Path(acceptance).resolve()
    read = lambda p: json.loads(p.read_bytes())
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    base = root / BASE / "gpu"
    config, result = read(base / "screen_config.json"), read(acceptance)
    _require(result["accepted"] is True and result["strict_ready"] is False and result["experiment_purpose"] == "transfer_study", "Noncausal acceptance required")
    plan = base / "scale_ema/rml_plan/trajectory.json"
    target = base / "scale_ema/results"
    terminal = target / "terminal.json"
    if (plan.parent / "rml_finalized/finalization.json").exists():
        return close_terminal_arm(root, plan, terminal)
    trajectory = read(plan)
    tid, run = trajectory["trajectory_id"], config["scale_study"]["logical_run_id"]
    folder = records / config["output_subdirectory"] / "scale_ema/training"
    target.mkdir(parents=True, exist_ok=True)
    from .research_memory.trace import atomic_write
    decision_path = target / "decision.md"
    atomic_write(decision_path, ("# Equal-update500K EMA observations\n\n"
        f"Saved artifacts were accepted on {datetime.now(timezone.utc).date()}. Fast-minus-slow selected EMA MAE: "
        f"{result['paired_analysis']['candidate_minus_reference_eV']:.10f} eV. "
        "One live optimizer supplied both views. This is PAIRED_ENDPOINT under transfer_study, not STRICT_CAUSAL, "
        "a full-scale ranking, multi-seed stability, or a Replay-ready causal pair.100K evidence remains context only. "
        "No successor, seed, protected evaluation or full training was authorized.\n").encode())
    role_use = {"internal_train": "consumed", "internal_development": "consumed", "official_validation": "untouched",
        "test_dev": "untouched", "test_challenge": "untouched"}
    roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{role}-{kind}", "trajectory_id": tid,
        "action_id": "A001", "run_id": run, "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
        "row_manifest_hash": result["result"]["manifest_sha256"], "role_name": role, "access_kind": kind,
        "selection_used": kind == "selection_used", "evidence_ref": rel(acceptance)}
        for role, kinds in (("internal_train", ("training_membership", "labels_read", "metric_computed")),
            ("internal_development", ("prediction_input", "labels_read", "metric_computed", "selection_used"))) for kind in kinds]
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-" + tid + "-observed-v1", "trajectory_id": tid,
        "action_id": "A001", "run_id": run, "attempt_id": "v1", "category": "training", "platform": "kaggle2",
        "hardware": "Tesla_T4", "evidence_ref": rel(acceptance), "measurement": {
            "device_hours": {"status": "measured", "value": result["native_cost"]["allocated_device_hours"] / 2},
            "wall_hours": {"status": "measured", "value": result["native_cost"]["wall_seconds"] / 3600},
            "cpu_hours": {"status": "measurement_missing", "value": None}, "queue_hours": {"status": "measurement_missing", "value": None}}}
    outcome = {"execution_status": "complete", "artifact_status": "accepted", "comparison_status": "paired_endpoint",
        "scientific_status": "not_evaluated", "transfer_status": "partial_evidence", "budget_decision": "stop_under_contract",
        "full_handoff_status": "not_authorized"}
    decision = {"final": True, "outcome": "CONTEXT_ONLY", "decision_ref": rel(decision_path),
        "next_allowed_actions": ["Controller interpretation of aligned EMA views only"], "reopen_conditions": ["Separately authorized causal500K comparison"]}
    result.update(evidence_id="pcqm-gptrans-g1-scale-ema-equal-updates-500k-s42", run_id=run,
        outcome=outcome, trajectory_decision=decision, role_use=role_use, roles=roles, costs=[cost])
    atomic_json(acceptance, result)
    trace = load_canonical_trace(folder / "ema999/canonical_trace.json")
    trace_manifest = build_default_trace_manifest(root, trajectory, {"run_id": run}, trace)
    trace_manifest.update(trace_artifact_ref=rel(folder / "ema999/canonical_trace.json"),
        terminal_evidence_ref=rel(terminal), evaluation_role_identity="fixed500k:development-500000-550000",
        selection_semantics="minimum EMA999 internal50K MAE over60equal-update rungs",
        backtest_eligibility={"eligible": False, "exclusion_reasons": [result["replay_exclusion"]]})
    paths = [plan, base / "scale_ema/contract.json", base / "scale_ema/comparison_readiness_prelaunch.json",
        root / BASE / "protocol.md", base / "submission_v1.json", acceptance, decision_path,
        folder / "completion_manifest.json", folder / "last_checkpoint.pt", folder / "qualification.json",
        *[folder / name for name in result["result"]["files"]], records / config["output_subdirectory"] / "native_cost.json"]
    paths = sorted(set(paths))
    timestamp = datetime.now(timezone.utc).isoformat()
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": result["evidence_id"], "track": "C", "scope": "equal-update500K shared-live EMA transfer study",
        "legacy_contract": "none-prospective-v5", "outcome": outcome, "role_use": role_use,
        "authority": {"pointers": [rel(plan), rel(root / BASE / "protocol.md"), rel(base / "scale_ema/contract.json"), rel(decision_path)]},
        "artifacts": _retained_artifacts(paths, folder / "ema999/canonical_trace.json", rel),
        "migration": {"migrated_at": timestamp, "training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False, "verification_scope": "Saved tensors and canonical traces; no local model execution"}}
    atomic_json(terminal, {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid, "run_id": run,
        "action_id": "A001", "finalized_at": timestamp, "acceptance_ref": rel(acceptance),
        "artifact_hashes": {rel(p): sha256_file(p) for p in paths}, "evidence": evidence, "decision": decision,
        "costs": [cost], "roles": roles, "role_use": role_use, "trace_manifest": trace_manifest})
    return close_terminal_arm(root, plan, terminal, trace=folder / "ema999/canonical_trace.json", arm_identifier="ema999")
