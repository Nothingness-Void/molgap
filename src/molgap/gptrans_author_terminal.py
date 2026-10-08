"""Native G1/G2 metadata adapter over the shared per-arm RML transaction."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .comparison_readiness import (
    REQUIRED_OBSERVED_BINDINGS, assess_comparison_readiness,
    reference_bundle_digest, validate_comparison_readiness,
)
from .research_memory.paths import verify_bound_artifact
from .research_memory.terminal_wiring import close_terminal_arm
from .research_memory.trace import atomic_write, file_digest, json_bytes


def close_author_outputs(repo_root: Path, records: Path, acceptance: Path,
        *, experiment_ref: str = "experiments/pcqm_gptrans_author_alignment") -> list[dict]:
    root = Path(repo_root).resolve()
    base = root / experiment_ref
    capacity_studies = {"experiments/pcqm_gptrans_capacity_nodes_100k", "experiments/pcqm_gptrans_capacity_relations_100k", "experiments/pcqm_gptrans_pair_transition_100k", "experiments/pcqm_gptrans_local_control_100k", "experiments/pcqm_gptrans_local_relation_100k"}
    followup = experiment_ref in capacity_studies or experiment_ref in {"experiments/pcqm_gptrans_input_ema_100k", "experiments/pcqm_gptrans_recipe_paths_100k", "experiments/pcqm_gptrans_pair_scale_100k", "experiments/pcqm_gptrans_path_ema_combination_100k", "experiments/pcqm_gptrans_readout_100k", "experiments/pcqm_gptrans_decay_clock_100k"}
    if experiment_ref not in capacity_studies and experiment_ref not in {"experiments/pcqm_gptrans_author_alignment", "experiments/pcqm_gptrans_input_ema_100k", "experiments/pcqm_gptrans_recipe_paths_100k", "experiments/pcqm_gptrans_pair_scale_100k", "experiments/pcqm_gptrans_path_ema_combination_100k", "experiments/pcqm_gptrans_readout_100k", "experiments/pcqm_gptrans_decay_clock_100k"}:
        raise ValueError("Unsupported terminal experiment adapter")
    gpu = base / "gpu"
    load = lambda p: json.loads(p.read_bytes())
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    save = lambda p, value: atomic_write(p, json_bytes(value))
    result = load(acceptance)
    if result.get("accepted") is not True:
        raise ValueError("Independent saved-output acceptance is required")
    config = load(gpu / "screen_config.json")
    screen = Path(records).resolve() / config.get("output_subdirectory", "gptrans_input_ema_screen" if followup else "gptrans_author_screen")
    bundle = load(root / config["reference_bundle_ref"])
    reference_manifest = load(root / bundle["trace_manifest_ref"])
    ref_metadata = load(root / bundle["role_history_ref"])
    if followup:
        from .research_memory.candidate_reference import verify_candidate_reference
        verify_candidate_reference(root, bundle["candidate_reference_qualification_ref"], expected_bundle=bundle)
        proof = load(root / bundle["candidate_reference_qualification_ref"])
        ready_ref = proof.get("terminal_readiness_ref")
        if ready_ref is None:
            qualified = load(root / proof["candidate_qualification_ref"])
            ready_ref = qualified["qualified_readiness_ref"]
        ref_bindings = load(root / ready_ref)["candidate_artifact_bindings"]
        ref_prediction = root / ref_bindings["prediction_manifest"]["ref"]
        ref_checkpoint = root / ref_bindings["checkpoint"]["ref"]
    else:
        ref_prediction = root / "experiments/pcqm_gptrans_v5_audit_reference/results/terminal/prediction_manifest.json"
        ref_checkpoint = (root / load(ref_prediction)["artifact_locator"]).parent / "best_model.pt"
    if file_digest(ref_checkpoint) != bundle["checkpoint_identity"]:
        raise ValueError("Reference checkpoint binding changed")
    reference_paths = {name: root / bundle[name + "_ref"] for name in REQUIRED_OBSERVED_BINDINGS
        if name not in {"checkpoint", "prediction_manifest", "source_config"}}
    reference_paths.update(checkpoint=ref_checkpoint, prediction_manifest=ref_prediction,
        source_config=root / bundle["contract_ref"])
    bind = lambda paths: {name: {"ref": rel(p), "sha256": file_digest(p)} for name, p in paths.items()}
    role_plan = load(gpu / next(iter(config["arms"])) / "comparison_readiness_prelaunch.json")["role_applicability_plan"]
    trace_fields = reference_manifest["trace_fields"]
    if not all(trace_fields.values()):
        raise ValueError("Reference strict trace fields are incomplete")
    def side(identity, bindings, events):
        return {"comparison_identity": identity, "artifacts": {k: "complete" for k in bindings},
            "artifact_bindings": bindings, "terminal_complete": True, "prediction_status": "complete",
            "row_alignment_status": "aligned", "runtime_certificate_status": "accepted",
            "role_applicability_plan": role_plan, "observed_role_event_kinds": sorted({r["access_kind"] for r in events}),
            "trace_field_availability": trace_fields, "trace_status": "complete", "stochasticity_status": "unavailable"}
    reference = side(bundle["comparison_identity"], bind(reference_paths), ref_metadata["roles"])
    reference.update(reference_bundle_id=bundle["reference_bundle_id"], reference_bundle_sha256=reference_bundle_digest(bundle))
    closures = []
    for mode, arm in result["arms"].items():
        target, training = gpu / mode / "results", screen / mode / "training"
        plan_path = gpu / mode / "rml_plan/trajectory.json"
        tid, run = arm["trajectory_id"], arm["run_id"]
        evidence_id = ("pcqm-gptrans-g1-" if followup else "pcqm-gptrans-author-") + mode.replace("_", "-") + "-100k-s42"
        timestamp = (load(target / "terminal.json")["finalized_at"] if (target / "terminal.json").exists()
            else datetime.now(timezone.utc).isoformat())
        for name in ("prediction_manifest", "runtime_certificate", "paired_analysis"):
            save(target / (name + ".json"), arm[name])
        # Preserve historical reference qualification rather than force pool admission.
        manifest = copy.deepcopy(reference_manifest)
        manifest.update(trajectory_id=tid, run_id=run, comparison_role="candidate", reference_id=bundle["reference_id"],
            trace_artifact_ref=rel(training / "canonical_trace.json"),
            trace_artifact_sha256=file_digest(training / "canonical_trace.json"),
            terminal_evidence_ref=rel(target / "terminal_evidence.json"))
        manifest["comparability_identity"]["architecture_identity"] = arm["comparison_identity"]["architecture_config_identity"]
        manifest["comparability_identity"]["matched_architecture_required"] = False
        manifest["contract_ref"] = rel(gpu / mode / "contract.json") if followup else manifest["contract_ref"]
        manifest["model_identity"] = arm["comparison_identity"]["architecture_config_identity"]
        if followup and arm["comparison_identity"]["ema_decay"] == .999:
            manifest["comparability_identity"]["ema_semantics"] = "ema999-each-step-selection"
        manifest["comparability_identity"]["optimizer_identity"] = arm["comparison_identity"]["optimizer_identity"]
        manifest["backtest_eligibility"] = ({"eligible": True, "exclusion_reasons": []} if followup else
            {"eligible": False, "exclusion_reasons": ["canonical_reference_not_replay_enrolled; historical eligibility left unchanged"]})
        save(target / "candidate_trace_manifest.json", manifest)
        role_use = {"internal_train": "consumed", "internal_development": "consumed",
            "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
        metadata_ref = rel(target / "observed_metadata.json")
        roles = []
        for role, kinds in (("internal_train", ("training_membership", "labels_read", "metric_computed")),
                ("internal_development", ("prediction_input", "labels_read", "metric_computed", "selection_used"))):
            for kind in kinds:
                roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{role}-{kind}",
                    "trajectory_id": tid, "action_id": "A001", "run_id": run, "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1",
                    "row_manifest_hash": file_digest(root / bundle["row_manifest_ref"]),
                    "role_name": role, "access_kind": kind, "selection_used": kind == "selection_used", "evidence_ref": metadata_ref})
        native = result["native_cost"]
        cost = {"schema": "molgap-cost-event-v1", "cost_event_id": f"cost-{tid}-observed-v1",
            "trajectory_id": tid, "action_id": "A001", "run_id": run, "attempt_id": "v1", "category": "training",
            "platform": config.get("platform_id", "kaggle3" if followup else "kaggle2"), "hardware": "Tesla_T4", "evidence_ref": metadata_ref,
            "measurement": {"device_hours": {"status": "measured", "value": native["allocated_device_hours"] / len(config["arms"])},
                "wall_hours": {"status": "measured", "value": native["wall_seconds"] / 3600},
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "queue_hours": {"status": "measurement_missing", "value": None}}}
        label = "POSITIVE_UNDER_CONTRACT" if arm["material_gate_passed"] else "INCONCLUSIVE"
        outcome = {"execution_status": "complete", "artifact_status": "accepted", "comparison_status": "strict_causal" if followup else "paired_endpoint",
            "scientific_status": label, "transfer_status": "not_evaluated", "budget_decision": "stop_under_contract",
            "full_handoff_status": "not_authorized"}
        decision = {"outcome": label, "final": True, "decision_ref": rel(target / "decision.md"),
            "next_allowed_actions": ["CLOSE"], "reopen_conditions": ["explicit scale/combination compute authorization"]}
        decision_text = (
            f"# {mode}: bounded terminal decision\n\nOn 2026-10-01, saved outputs passed independent acceptance. "
            f"Frozen EMA MAE: {arm['paired_analysis']['candidate_mae_eV']:.10f} eV; "
            f"reference gain: {arm['material_gain_eV']:.10f} eV against this protocol's 0.003 eV gate. "
            f"Parameters: {arm['parameters']}; selected epoch: {arm['best_epoch']}.\n\n"
            "The result is a paired endpoint, not a STRICT_CAUSAL claim: the immutable reference bundles role/cost/acceptance "
            "in one metadata file, incompatible with the distinct-binding gate. Row bootstrap does not measure "
            "training-seed variability. No scale, combination or delivery superiority was established. Late EMA continued "
            "improving and lagged live weights; changing endpoint selection was not authorized.\n\n"
            "RML closure retains the full observed trace. Replay is blocked by the immutable reference's lack of replay enrollment; "
            "the historical qualification was not rewritten. No successor was released.\n"
        )
        if followup:
            decision_text = (f"# {mode}: bounded terminal decision\n\nOn {timestamp[:10]}, independently accepted saved outputs "
                f"gave EMA MAE {arm['paired_analysis']['candidate_mae_eV']:.10f} eV and frozen-reference gain "
                f"{arm['material_gain_eV']:.10f} eV. The prospectively selected material gate was "
                f"{config['material_gate_eV']} eV; passed={arm['material_gate_passed']}.\n\n"
                "This is a single-seed causal comparison of the declared intervention, not seed stability or scale superiority. "
                f"Immutable reference {bundle['reference_id']} endpoint and canonical observations were reused without retraining. "
                "No protected evaluation role was accessed. No successor was authorized by this terminal result.\n")
        atomic_write(target / "decision.md", decision_text.encode())
        save(target / "observed_metadata.json", {"evidence_id": evidence_id, "trajectory_id": tid, "run_id": run,
            "outcome": outcome, "trajectory_decision": decision, "role_use": role_use, "roles": roles, "costs": [cost],
            "native_cost_ref": rel(screen / "native_cost.json"), "cost_attribution": "full allocated T4 cost divided among declared arms; includes idle devices, not utilization",
            "observed_role_sources": [rel(training / "completion_manifest.json"), rel(training / "canonical_trace.json"), rel(acceptance)],
            "local_training_executed": False, "model_inference_executed": False})
        save(target / "role_history.json", {"roles": roles, "source_ref": metadata_ref,
            "source_sha256": file_digest(target / "observed_metadata.json")})
        save(target / "cost_records.json", {"costs": [cost], "source_ref": metadata_ref,
            "source_sha256": file_digest(target / "observed_metadata.json")})
        paths = {name: root / bundle[name + "_ref"] for name in ("row_manifest", "target_manifest", "target_transform_asset")}
        paths.update(checkpoint=training / "best_model.pt", runtime_certificate=target / "runtime_certificate.json",
            prediction_manifest=target / "prediction_manifest.json", trace_manifest=target / "candidate_trace_manifest.json",
            role_history=target / "role_history.json", cost_records=target / "cost_records.json",
            acceptance=acceptance, decision=target / "decision.md", source_config=gpu / "screen_config.json",
            paired_analysis=target / "paired_analysis.json")
        declaration = config["arms"][mode]
        readiness = assess_comparison_readiness(candidate_id=evidence_id, candidate=side(arm["comparison_identity"], bind(paths), roles),
            reference_id=bundle["reference_id"], reference=reference,
            declared_intervention_fields=declaration.get("declared_intervention_fields", ["architecture_config_identity"]),
            experiment_purpose=declaration.get("experiment_purpose", "mechanism_comparison"),
            intervention_group_id="gptrans-g1-followup" if followup else "gptrans-author-input-attribution", mechanism_id=mode)
        validate_comparison_readiness(readiness, evidence_verifier=lambda p, sha: verify_bound_artifact(root, p, sha))
        if ((followup and (readiness["comparison_class"] != "STRICT_CAUSAL" or not readiness["strict_ready"])) or
            (not followup and (readiness["comparison_class"] != "PAIRED_ENDPOINT" or readiness["blocker_codes"] != ["OBSERVED_ARTIFACT_BINDINGS_INCOMPLETE"]))):
            raise ValueError("Unexpected terminal qualification: " + str(readiness["blocker_codes"]))
        # Verify every original reference binding even though it cannot earn the
        # strict distinct-binding claim. Never replace it with a new authority.
        for binding in reference["artifact_bindings"].values():
            verify_bound_artifact(root, binding["ref"], binding["sha256"])
        save(target / "comparison_readiness.json", readiness)
        authorities = [base / ("protocol.md" if followup else "dual_arm_protocol.md"), root / bundle["contract_ref"], plan_path,
            gpu / "submission_v1.json", gpu / mode / "comparison_readiness_prelaunch.json", target / "decision.md"]
        artifacts = sorted(set(paths.values()) | {target / "comparison_readiness.json", screen / "native_cost.json",
            target / "observed_metadata.json", training / "canonical_trace.json", training / "completion_manifest.json", training / "last_checkpoint.pt"})
        if followup:
            artifacts += [root / arm["prediction_manifest"]["artifact_locator"], gpu / mode / "contract.json"]
        if "diagnostic_trace" in arm:
            diagnostic = arm["diagnostic_trace"]
            verify_bound_artifact(root, diagnostic["ref"], diagnostic["sha256"])
            artifacts.append(root / diagnostic["ref"])
        hashes = {rel(p): file_digest(p) for p in artifacts + authorities}
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "evidence_id": evidence_id, "track": "C", "scope": "single_mechanism_screen_100k", "legacy_contract": "none-prospective-v5",
            "outcome": outcome, "role_use": role_use, "authority": {"pointers": [rel(p) for p in authorities]},
            "artifacts": [{"name": p.name, "locator": rel(p), "sha256": hashes[rel(p)], "availability": "local_verified"} for p in artifacts],
            "migration": {"migrated_at": timestamp, "verification_scope": "verified native artifacts and saved prediction tensors",
                "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
        if "diagnostic_trace" in arm:
            # The canonical trace owns replay axes; the redundant native log
            # supplies amplitude telemetry and must not become a second trace.
            for artifact in evidence["artifacts"]:
                if artifact["locator"] == arm["diagnostic_trace"]["ref"]:
                    artifact["purpose"] = "analysis_diagnostics"
        terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid, "run_id": run,
            "action_id": "A001", "finalized_at": timestamp, "acceptance_ref": metadata_ref, "artifact_hashes": hashes,
            "evidence": evidence, "decision": decision, "roles": roles, "costs": [cost], "role_use": role_use,
            "trace_manifest": manifest, "comparison_readiness_ref": rel(target / "comparison_readiness.json")}
        save(target / "terminal_evidence.json", evidence)
        save(target / "terminal.json", terminal)
        closure = close_terminal_arm(root, plan_path, target / "terminal.json", trace=training / "canonical_trace.json", arm_identifier=mode)
        save(target / "closure.json", closure)
        closures.append(closure)
    return closures


def close_author_failed_arm(repo_root: Path, records: Path, acceptance: Path, *, experiment_ref: str, mode: str):
    """Close an observed failed companion without inventing a training trace."""
    from .research_memory.pipeline import finalize_rebuild_backtest
    root = Path(repo_root).resolve()
    base, gpu = root / experiment_ref, root / experiment_ref / "gpu"
    load = lambda p: json.loads(p.read_bytes())
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    result, config = load(acceptance), load(gpu / "screen_config.json")
    if (not result.get("accepted") or result.get("acceptance_scope") != "selected_completed_arms"
            or mode in result["arms"] or mode not in config["arms"]):
        raise ValueError("Failure closure requires independently accepted partial run identity")
    screen = Path(records).resolve() / config["output_subdirectory"]
    outcomes = [r for r in load(screen / "job_summary.json")["outcomes"] if r["variant"] == mode]
    folder, target = screen / mode, gpu / mode / "failure_results"
    if len(outcomes) != 1 or outcomes[0]["complete"] is not False or (folder / "training/completion_manifest.json").exists():
        raise ValueError("Worker is not a retained incomplete attempt")
    preflight = load(folder / "preflight/preflight.json")
    log = (folder / "training.log").read_text()
    if (preflight.get("accepted") is not True or preflight["source_archive_sha256"] != result["source_archive_sha256"]
            or "ValueError: dictionary update sequence element" not in log
            or "canonical_fingerprint(inventory)" not in log):
        raise ValueError("Failure cause/source not established by retained artifacts")
    if any(preflight[k] is not False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
        raise ValueError("Failure attempt protected-role access")
    declaration = config["arms"][mode]
    tid = declaration["trajectory_id"]
    plan_path = gpu / mode / "rml_plan/trajectory.json"
    run = load(plan_path)["actions"][0]["run_ids"][0]
    metadata = target / "acceptance.json"
    timestamp = load(target / "terminal.json")["finalized_at"] if (target / "terminal.json").exists() else datetime.now(timezone.utc).isoformat()
    evidence_id = "pcqm-gptrans-g1-group-decay-infrastructure-failure-100k-s42-v1"
    outcome = {"execution_status": "infrastructure_failed", "artifact_status": "retained_partial",
        "comparison_status": "not_evaluated", "scientific_status": "not_evaluated", "transfer_status": "not_evaluated",
        "budget_decision": "stop_under_contract", "full_handoff_status": "not_authorized"}
    decision = {"outcome": "INFRASTRUCTURE_ONLY", "final": True, "decision_ref": rel(target / "decision.md"),
        "next_allowed_actions": [], "reopen_conditions": ["A separately qualified failed-arm recovery; never rerun the completed companion"]}
    role_use = {"internal_train": "consumed", "internal_development": "consumed",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    # The retained prediction payload and reached epoch-end stack establish role
    # consumption, but do not establish a checkpointed optimizer/exposure cursor.
    if not (folder / "training/development_predictions.pt").is_file():
        raise ValueError("Failed worker development-role evidence missing")
    roles = [{"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-failure-{role}-labels",
        "trajectory_id": tid, "action_id": "A001", "run_id": run, "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1",
        "row_manifest_hash": "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
        "role_name": role, "access_kind": "labels_read", "selection_used": False, "evidence_ref": rel(metadata)}
        for role in ("internal_train", "internal_development")]
    native = result["native_cost"]
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": f"cost-{tid}-failure-observed-v1",
        "trajectory_id": tid, "action_id": "A001", "run_id": run, "attempt_id": "v1", "category": "infrastructure_failure",
        "platform": "kaggle3", "hardware": "Tesla_T4", "evidence_ref": rel(metadata),
        "measurement": {"device_hours": {"status": "measured", "value": native["allocated_device_hours"] / 2},
                        "wall_hours": {"status": "measured", "value": native["wall_seconds"] / 3600},
                        "cpu_hours": {"status": "measurement_missing", "value": None},
                        "queue_hours": {"status": "measurement_missing", "value": None}}}
    atomic_write(target / "decision.md", ("# Failed diagnostic writer, not a scientific loss\n\n"
        f"On {timestamp[:10]}, the grouped arm failed because a list inventory was sent to a mapping-only digest API. "
        "The first epoch reached evaluation but no canonical observation or optimizer/RNG continuation checkpoint was retained. "
        "The selected model alone cannot resume the original contract. No full-epoch exposure or scientific metric was fabricated. "
        "One T4 was reserved throughout the parent wall time; this includes idle allocation, not measured busy time. "
        "The complete companion was accepted independently. No retry or successor was released.\n").encode())
    atomic_write(metadata, json_bytes({"evidence_id": evidence_id, "run_id": run, "outcome": outcome,
        "trajectory_decision": decision, "role_use": role_use, "roles": roles, "costs": [cost],
        "error": outcomes[0]["error"], "local_training_executed": False, "model_inference_executed": False}))
    artifacts = [metadata, target / "decision.md", acceptance, screen / "job_summary.json", screen / "native_cost.json",
        folder / "training.log", folder / "preflight/preflight.json", folder / "training/canonical_trace.json",
        folder / "training/development_predictions.pt", folder / "arm_binding.json"]
    authorities = [plan_path, gpu / "submission_v1.json", base / "protocol.md", gpu / mode / "comparison_readiness_prelaunch.json"]
    hashes = {rel(p): file_digest(p) for p in artifacts + authorities}
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": evidence_id, "track": "C", "scope": "failed_companion_execution", "legacy_contract": "none-prospective-v5",
        "outcome": outcome, "role_use": role_use, "authority": {"pointers": [rel(p) for p in authorities]},
        "artifacts": [{"name": p.name, "locator": rel(p), "sha256": hashes[rel(p)], "availability": "local_verified"} for p in artifacts],
        "migration": {"migrated_at": timestamp, "verification_scope": "Retained failure stack, native cost and partial artifacts only",
                      "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
    terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid, "run_id": run,
        "action_id": "A001", "finalized_at": timestamp, "acceptance_ref": rel(metadata), "artifact_hashes": hashes,
        "evidence": evidence, "decision": decision, "roles": roles, "costs": [cost], "role_use": role_use}
    atomic_write(target / "terminal.json", json_bytes(terminal))
    return finalize_rebuild_backtest(root, plan_path, target / "terminal.json")
