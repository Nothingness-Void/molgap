"""500K saved-output translation into the existing strict/RML terminal owners."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .comparison_readiness import (REQUIRED_OBSERVED_BINDINGS, assess_comparison_readiness,
    reference_bundle_digest, validate_comparison_readiness)
from .evidence_pointers import load_json_object
from .gptrans_scale_ema_acceptance import accept_outputs, _retained_artifacts
from .gptrans_scale_reference import BASE
from .k1_terminal_analysis import paired_saved_errors
from .research_memory.paths import verify_bound_artifact
from .research_memory.reference_qualification import verify_reference_qualification
from .research_memory.terminal_wiring import close_terminal_arm
from .research_memory.trace import atomic_write, file_digest, json_bytes, load_canonical_trace


def directional_signal(analysis, matched, threshold):
    gain = -analysis["candidate_minus_reference_eV"]
    final_gain = matched[-1]["gain_eV"]
    tail_gain = sum(row["gain_eV"] for row in matched[-10:]) / 10
    return gain >= threshold and analysis["paired_row_bootstrap"]["ci95"][1] < 0 and final_gain > 0 and tail_gain > 0


def _require_actual_replay_pair(root, tid, reference_id):
    pool = load_json_object(root / "research_memory/derived/replay_pool.json")
    candidate = [row for row in pool["entries"] if row["trajectory_id"] == tid and row["comparison_role"] == "candidate"]
    if len(candidate) != 1 or candidate[0]["capability"] != "complete":
        raise ValueError("Actual complete candidate Replay admission missing")
    key = candidate[0]["comparability_key"]
    reference = [row for row in pool["entries"] if row["reference_id"] == reference_id
        and row["comparability_key"] == key and row["comparison_role"] == "reference" and row["capability"] == "complete"]
    if len(reference) != 1:
        raise ValueError("Actual complete reference Replay admission missing")
    return {"actual_complete_replay_pair": True, "candidate_trajectory_id": tid,
        "reference_trajectory_id": reference[0]["trajectory_id"], "comparability_key": key}


def accept_local_outputs(root, records, package):
    import torch
    root, records = Path(root).resolve(), Path(records).resolve()
    native = accept_outputs(root, records, package, experiment_ref=BASE)
    config = load_json_object(root / BASE / "gpu/screen_config.json")
    bundle = load_json_object(root / config["reference_bundle_ref"])
    verify_reference_qualification(root, bundle["qualification_ref"], expected_bundle=bundle,
        expected_bundle_path=config["reference_bundle_ref"])
    folder = records / config["output_subdirectory"] / "scale_ema/training"
    payload = torch.load(folder / "ema999/development_predictions.pt", map_location="cpu", weights_only=True)
    reference = torch.load(root / bundle["prediction_manifest"]["artifact_locator"], map_location="cpu", weights_only=True)
    analysis = paired_saved_errors(reference, payload)
    trace = load_canonical_trace(folder / "ema999/canonical_trace.json")
    manifest = load_json_object(root / bundle["trace_manifest_ref"])
    ref_trace = load_canonical_trace(root / manifest["trace_artifact_ref"])
    matched = []
    for row, ref in zip(trace["observations"], ref_trace["observations"]):
        if any(row[k] != ref[k] for k in ("optimizer_step", "sample_presentations")):
            raise ValueError("Reference/candidate clocks differ")
        matched.append({"optimizer_step": row["optimizer_step"], "sample_presentations": row["sample_presentations"],
            "candidate_live_eV": row["live_dev_metric"], "reference_live_eV": ref["live_dev_metric"],
            "candidate_ema_eV": row["ema_dev_metric"], "reference_ema_eV": ref["ema_dev_metric"],
            "gain_eV": ref["ema_dev_metric"] - row["ema_dev_metric"]})
    gain = -analysis["candidate_minus_reference_eV"]
    final_gain = matched[-1]["gain_eV"]
    tail_gain = sum(row["gain_eV"] for row in matched[-10:]) / 10
    directional = directional_signal(analysis, matched, config["directional_transfer_gate_eV"])
    prediction = {**bundle["prediction_manifest"], "artifact_locator": (folder / "ema999/development_predictions.pt").relative_to(root).as_posix(),
        "artifact_sha256": file_digest(folder / "ema999/development_predictions.pt")}
    for field, tensor in (("prediction_sha256", "prediction_eV"), ("source_idx_sha256", "source_idx"), ("target_sha256", "target_eV")):
        prediction[field] = hashlib.sha256(payload[tensor].contiguous().numpy().tobytes()).hexdigest()
    if any(prediction[key] != bundle["prediction_manifest"][key] for key in ("source_idx_sha256", "target_sha256")):
        raise ValueError("Reference row/target bytes differ")
    return {"accepted": True, "experiment_purpose": "architecture_comparison", "native_acceptance": native,
        "parameters": native["result"]["parameters"], "prediction_manifest": prediction, "paired_analysis": analysis,
        "matched_trajectory": matched, "selected_gain_eV": gain, "final_gain_eV": final_gain,
        "final_ten_mean_gain_eV": tail_gain, "directional_transfer_signal": directional,
        "historical_material_gate_passed": directional and gain >= config["material_gate_eV"],
        "model_inference_executed": False, "local_training_executed": False, "replay_ready": False,
        "replay_note": "Actual admission is checked after independent terminal closure, not asserted by tensor acceptance"}


def close_local_outputs(root, records, acceptance):
    root, records, acceptance = Path(root).resolve(), Path(records).resolve(), Path(acceptance).resolve()
    load = lambda p: load_json_object(p)
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    save = lambda p, value: atomic_write(p, json_bytes(value))
    gpu = root / BASE / "gpu"
    config, result = load(gpu / "screen_config.json"), load(acceptance)
    if result["accepted"] is not True or result["experiment_purpose"] != "architecture_comparison":
        raise ValueError("Independent local500K acceptance is required")
    bundle = load(root / config["reference_bundle_ref"])
    verify_reference_qualification(root, bundle["qualification_ref"], expected_bundle=bundle,
        expected_bundle_path=config["reference_bundle_ref"])
    target = gpu / "scale_ema/results"
    plan = gpu / "scale_ema/rml_plan/trajectory.json"
    screen = records / config["output_subdirectory"]
    folder = screen / "scale_ema/training"
    trace_path = folder / "ema999/canonical_trace.json"
    tid, run = config["arms"]["scale_ema"]["trajectory_id"], config["scale_study"]["logical_run_id"]
    if (plan.parent / "rml_finalized/finalization.json").is_file():
        closed = close_terminal_arm(root, plan, target / "terminal.json", trace=trace_path, arm_identifier="ema999")
        return {"closure": closed, **_require_actual_replay_pair(root, tid, bundle["reference_id"])}
    eid = "pcqm-gptrans-g1-local-transfer-equal-updates-500k-s42"
    timestamp = datetime.now(timezone.utc).isoformat()
    ref_manifest = load(root / bundle["trace_manifest_ref"])
    manifest = deepcopy(ref_manifest)
    manifest.update(trajectory_id=tid, run_id=run, reference_id=bundle["reference_id"], comparison_role="candidate",
        model_identity=config["arms"]["scale_ema"]["comparison_identity"]["architecture_config_identity"],
        contract_ref=rel(gpu / "scale_ema/contract.json"), trace_artifact_ref=rel(trace_path), trace_artifact_sha256=file_digest(trace_path),
        terminal_evidence_ref=rel(target / "terminal_evidence.json"), backtest_eligibility={"eligible": True, "exclusion_reasons": []})
    manifest["comparability_identity"]["architecture_identity"] = manifest["model_identity"]
    manifest["comparability_identity"]["matched_architecture_required"] = False
    save(target / "candidate_trace_manifest.json", manifest)
    save(target / "prediction_manifest.json", result["prediction_manifest"])
    save(target / "paired_analysis.json", result["paired_analysis"])
    save(target / "row_manifest.json", {key: result["prediction_manifest"][key] for key in ("source_idx_sha256", "row_count", "ordering_semantics")})
    save(target / "target_manifest.json", {"target_identity": bundle["comparison_identity"]["target_identity"], "target_sha256": result["prediction_manifest"]["target_sha256"], "unit": "eV"})
    qualification = load(folder / "qualification.json")
    save(target / "runtime_certificate.json", {"status": "accepted", "scope": bundle["comparison_identity"]["runtime_certificate_scope"],
        "qualification_ref": rel(folder / "qualification.json"), "qualification_sha256": file_digest(folder / "qualification.json"),
        "runtime": qualification["runtime"], "deterministic_repeat": qualification["deterministic_repeat"]})
    role_use = {"internal_train": "consumed", "internal_development": "consumed", "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{role}-{kind}", "trajectory_id": tid,
        "action_id": "A001", "run_id": run, "dataset_identity": bundle["comparison_identity"]["dataset_identity"],
        "row_manifest_hash": config["dataset_manifest_sha256"], "role_name": role, "access_kind": kind,
        "selection_used": kind == "selection_used", "evidence_ref": rel(acceptance)}
        for role, kinds in (("internal_train", ("training_membership", "labels_read", "metric_computed")),
            ("internal_development", ("prediction_input", "labels_read", "metric_computed", "selection_used"))) for kind in kinds]
    native = result["native_acceptance"]["native_cost"]
    costs = [{"schema": "molgap-cost-event-v1", "cost_event_id": "cost-" + tid + "-observed-v1", "trajectory_id": tid,
        "action_id": "A001", "run_id": run, "attempt_id": "v1", "category": "training", "platform": "kaggle2",
        "hardware": "Tesla_T4", "evidence_ref": rel(screen / "native_cost.json"), "measurement": {
            "device_hours": {"status": "measured", "value": native["allocated_device_hours"]},
            "wall_hours": {"status": "measured", "value": native["wall_seconds"] / 3600},
            "cpu_hours": {"status": "measurement_missing", "value": None}, "queue_hours": {"status": "measurement_missing", "value": None}}}]
    save(target / "role_history.json", {"roles": roles})
    save(target / "cost_records.json", {"costs": costs, "idle_allocation_counted": True})
    label = "POSITIVE_UNDER_CONTRACT" if result["historical_material_gate_passed"] else "INCONCLUSIVE"
    scientific = label if result["historical_material_gate_passed"] else "POSITIVE_BELOW_GATE" if result["directional_transfer_signal"] else "NEGATIVE_UNDER_CONTRACT"
    outcome = {"execution_status": "complete", "artifact_status": "accepted", "comparison_status": "strict_causal",
        "scientific_status": scientific, "transfer_status": "partial_evidence", "budget_decision": "stop_under_contract", "full_handoff_status": "not_authorized"}
    decision_path = target / "decision.md"
    atomic_write(decision_path, (f"# Equal-update500K local bridge\n\nOn {timestamp[:10]} independently accepted endpoints gave selected gain {result['selected_gain_eV']:.10f} eV, final gain {result['final_gain_eV']:.10f}, and final-ten mean {result['final_ten_mean_gain_eV']:.10f}. Directional signal={result['directional_transfer_signal']}; historical material gate passed={result['historical_material_gate_passed']}. The original control's noncausal claim remains unchanged. Single-seed equal-update evidence is not full convergence, seed stability, protected evaluation or a full-training release.\n").encode())
    role_plan = load(gpu / "scale_ema/comparison_readiness_prelaunch.json")["role_applicability_plan"]
    bind = lambda paths: {key: {"ref": rel(path), "sha256": file_digest(path)} for key, path in paths.items()}
    def side(identity, paths, events):
        return {"comparison_identity": identity, "artifacts": {key: "complete" for key in paths}, "artifact_bindings": bind(paths),
            "terminal_complete": True, "prediction_status": "complete", "row_alignment_status": "aligned", "runtime_certificate_status": "accepted",
            "role_applicability_plan": role_plan, "observed_role_event_kinds": sorted({event["access_kind"] for event in events}),
            "trace_field_availability": manifest["trace_fields"], "trace_status": "complete", "stochasticity_status": "unavailable"}
    proof = load(root / bundle["qualification_ref"])
    reference_paths = {key: root / bundle[key + "_ref"] for key in REQUIRED_OBSERVED_BINDINGS if key not in {"checkpoint", "prediction_manifest", "source_config"}}
    reference_paths.update(checkpoint=root / proof["checkpoint_ref"], prediction_manifest=root / proof["prediction_manifest_ref"], source_config=root / bundle["contract_ref"])
    ref_events = load(root / bundle["role_history_ref"])["roles"]
    reference = side(bundle["comparison_identity"], reference_paths, ref_events)
    reference.update(reference_bundle_id=bundle["reference_bundle_id"], reference_bundle_sha256=reference_bundle_digest(bundle))
    paths = {"checkpoint": folder / "ema999/best_model.pt", "source_config": gpu / "screen_config.json", "acceptance": acceptance,
        "decision": decision_path, "target_transform_asset": root / bundle["target_transform_asset_ref"], "trace_manifest": target / "candidate_trace_manifest.json"}
    paths.update({key: target / (key + ".json") for key in ("prediction_manifest", "row_manifest", "target_manifest", "runtime_certificate", "role_history", "cost_records", "paired_analysis")})
    readiness = assess_comparison_readiness(candidate_id=eid, candidate=side(config["arms"]["scale_ema"]["comparison_identity"], paths, roles),
        reference_id=bundle["reference_id"], reference=reference, experiment_purpose="architecture_comparison",
        declared_intervention_fields=["architecture_config_identity"], intervention_group_id="gptrans-local-transfer500k", mechanism_id="uncapped-real-bond-local-stream")
    validate_comparison_readiness(readiness, evidence_verifier=lambda ref, digest: verify_bound_artifact(root, ref, digest))
    if not readiness["strict_ready"]:
        raise ValueError("500K strict comparison incomplete: " + str(readiness["blocker_codes"]))
    save(target / "comparison_readiness.json", readiness)
    authority = [root / BASE / "protocol.md", plan, gpu / "scale_ema/contract.json", gpu / "submission_v1.json", decision_path]
    retained = set(paths.values()) | set(authority) | {target / "comparison_readiness.json", folder / "completion_manifest.json", screen / "native_cost.json"}
    retained.update(folder / name for name in result["native_acceptance"]["result"]["files"])
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL", "evidence_id": eid,
        "track": "C", "scope": "same-update500K local architecture comparison", "legacy_contract": "none-prospective-v5", "outcome": outcome,
        "role_use": role_use, "authority": {"pointers": [rel(path) for path in authority]}, "artifacts": _retained_artifacts(sorted(retained), trace_path, rel),
        "migration": {"migrated_at": timestamp, "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False,
            "verification_scope": "saved predictions, native trace, source/config, runtime, roles and complete allocation"}}
    decision = {"final": True, "outcome": label, "decision_ref": rel(decision_path), "next_allowed_actions": ["Controller interpretation and the separately planned bounded NO_TRAIN probes only"], "reopen_conditions": ["explicit new compute decision"]}
    save(target / "terminal_evidence.json", evidence)
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid, "run_id": run, "action_id": "A001", "finalized_at": timestamp,
        "acceptance_ref": rel(acceptance), "artifact_hashes": {rel(path): file_digest(path) for path in retained}, "evidence": evidence,
        "decision": decision, "roles": roles, "costs": costs, "role_use": role_use, "trace_manifest": manifest,
        "comparison_readiness_ref": rel(target / "comparison_readiness.json")}
    save(target / "terminal.json", terminal)
    closed = close_terminal_arm(root, plan, target / "terminal.json", trace=trace_path, arm_identifier="ema999")
    return {"closure": closed, **_require_actual_replay_pair(root, tid, bundle["reference_id"])}
