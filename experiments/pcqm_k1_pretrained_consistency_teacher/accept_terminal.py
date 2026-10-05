"""Translate this frozen pair's retained evidence into existing V5/RML owners."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from molgap.comparison_readiness import (
    assess_comparison_readiness, validate_comparison_readiness,
    validate_reference_bundle, reference_bundle_digest,
    REQUIRED_OBSERVED_BINDINGS, REQUIRED_CANDIDATE_OBSERVED_BINDINGS,
)
from molgap.evidence_pointers import verify_bound_artifact
from molgap.experiment_family_workflow import (
    RunContext, TargetIdentityBinding, inspect_output, _check_context,
)
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_screen_training import validate_runtime_preflight
from molgap.research_memory.terminal_wiring import close_terminal_multi_arm
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint
from molgap.v5_common import validate_v5_evidence_envelope

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
RAW = ROOT / "platforms/_records/kaggle/training/k1_pretrained_combo_kaggle1_v1"
OUT = OWNER / "terminal_acceptance"
ARMS = ("pretrained_consistency", "pretrained_consistency_teacher")


def read(path):
    return json.loads(Path(path).read_bytes())


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Immutable terminal input conflict: {path.name}")
    path.write_bytes(payload)
    return relative(path)


def main():
    spec = ExperimentSpec.from_json((OWNER / "experiment_spec.json").read_text())
    prep = read(OWNER / "preparation_reconciliation.json")
    plan = read(OWNER / "family_acceptance_plan.json")
    binding = TargetIdentityBinding.from_acceptance_plan(spec, ROOT, relative(OWNER / "family_acceptance_plan.json"),
        plan_sha256=file_digest(OWNER / "family_acceptance_plan.json"))
    metrics = read(OUT / "scientific_metrics.json")
    assert metrics["teacher_increment_gate_passed"] and not metrics["compression_gate_passed"]
    observation = read(OWNER / "terminal_remote_observation_v1.json")
    assert observation["scheduler"]["status"] == "COMPLETE" and observation["entry_source_verified"]
    pair = read(RAW / "pair_state.json")
    assert pair["status"] == "complete" and pair["spec_identity"] == spec.identity
    stamp_path = OUT / "finalization_timestamp.json"
    stamp = read(stamp_path)["timestamp"] if stamp_path.exists() else datetime.now(timezone.utc).isoformat()
    write(stamp_path, {"timestamp": stamp})
    old_bundle = read(OWNER / "historical_reference_bundle.json") if (OWNER / "historical_reference_bundle.json").exists() else read(ROOT / "experiments/pcqm_k1_dropout_consistency/historical_reference_bundle.json")
    role_use = {"official_train_prefix_0_100000": "used", "fixed50k_development": "selection_used",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    evidence_ids = {a: "pcqm-k1-pretrained-combo-" + a + "-kaggle1-s42-v1-terminal" for a in ARMS}
    infos = {}
    for arm in plan["arms"]:
        a = arm["arm_id"]
        directory = RAW / a
        context = RunContext.from_launch(spec, OWNER / "launch_receipt_v1.json", Path(prep["package"]),
            expected_package_identity=prep["package_identity"], arm_id=a)
        check = inspect_output(directory, context=context, expected=arm["expected"], target_identity=binding)
        assert check["status"] == "MECHANICALLY_VERIFIED", check
        provenance = read(directory / "runtime_provenance.json")
        _check_context(provenance["context"], context)
        certificate = validate_runtime_preflight(directory, provenance)
        assert certificate["platform_id"] == "kaggle1-t4x2"
        receipt = read(directory / "objective_trace_receipt.json")
        assert receipt["sha256"] == file_digest(directory / "objective_trace.json") and receipt["epochs"] == 40
        _check_context(receipt["context"], context)
        # These checkpoint schemas omit transform statistics. The frozen trainer
        # derives them from the pinned training rows; inspect_output binds the
        # retained float32 development targets through the acceptance plan.
        trajectory_path = OWNER / "kaggle1_v1" / a / "trajectory.json"
        trajectory = read(trajectory_path)
        action = trajectory["actions"][0]
        acceptance_path = OUT / a / "acceptance.json"
        outcome_name = "CLOSED" if a == ARMS[0] else "POSITIVE_UNDER_CONTRACT"
        decision = {"decision_ref": relative(OUT / "decision.md"), "next_allowed_actions": [],
            "outcome": outcome_name, "reopen_conditions": ["Separate explicit adoption or justified prospective scale contract; compression gate has not passed."]}
        costs = []
        for category, filename in (("training", "allocation_cost.json"), ("preflight", "diagnostic_cost.json")):
            source = read(directory / filename)
            device = next(c for c in source["costs"] if c["metric"] == "device_seconds")
            wall = next(c for c in source["costs"] if c["metric"] == "wall_seconds")
            costs.append({"schema": "molgap-cost-event-v1", "cost_event_id":
                action["cost_event_ids"][0] if category == "training" else "cost-" + trajectory["trajectory_id"] + "-observed-preflight",
                "trajectory_id": trajectory["trajectory_id"], "action_id": "A001", "run_id": action["run_ids"][0],
                "attempt_id": action["attempt_ids"][0], "platform": "kaggle1", "hardware": "Tesla T4", "category": category,
                "evidence_ref": relative(acceptance_path), "measurement": {
                    "device_hours": {"status": "measured", "value": device["value"] / 3600},
                    "wall_hours": {"status": "measured", "value": wall["value"] / 3600},
                    "cpu_hours": {"status": "measurement_missing", "value": None},
                    "queue_hours": {"status": "measurement_missing", "value": None}}})
        roles = []
        for role, kinds, row_hash in (("official_train_prefix_0_100000", ("training_membership", "labels_read", "prediction_input"), old_bundle["comparison_identity"]["dataset_identity"]),
            ("fixed50k_development", ("labels_read", "prediction_input", "metric_computed", "selection_used"), arm["expected"]["source_idx_sha256"])):
            for kind in kinds:
                roles.append({"schema": "molgap-role-event-v1", "role_event_id": "role-" + trajectory["trajectory_id"] + "-" + role + "-" + kind,
                    "trajectory_id": trajectory["trajectory_id"], "action_id": "A001", "run_id": action["run_ids"][0],
                    "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1", "row_manifest_hash": row_hash, "role_name": role,
                    "access_kind": kind, "selection_used": kind == "selection_used", "evidence_ref": relative(acceptance_path)})
        outcome = {"execution_status": "complete", "artifact_status": "hash_verified_retained",
            "comparison_status": "same_job_teacher_increment_strict_single_seed", "scientific_status": outcome_name,
            "transfer_status": "compression_gate_failed_500k_not_evaluated", "budget_decision": "no_automatic_successor_or_scale_up", "full_handoff_status": "not_applicable"}
        acceptance = {"evidence_id": evidence_ids[a], "run_id": action["run_ids"][0], "outcome": outcome, "trajectory_decision": decision,
            "mechanical": check, "role_use": role_use, "costs": costs, "roles": roles,
            "runtime_certificate_id": canonical_fingerprint(certificate), "metrics_ref": relative(OUT / "scientific_metrics.json"),
            "physical_attempt_mapping": {"logical_attempt_id": action["attempt_ids"][0], "kernel_id": 137071131, "version_number": 1,
                "receipt_ref": relative(OWNER / "launch_receipt_v1.json")}, "native_cost_scope": read(directory / "allocation_cost.json")["scope"]}
        write(acceptance_path, acceptance)
        write(OUT / a / "role_history.json", {"roles": roles, "role_use": role_use,
            "source_acceptance_ref": relative(acceptance_path)})
        write(OUT / a / "cost_records.json", {"costs": costs,
            "source_acceptance_ref": relative(acceptance_path), "scope": acceptance["native_cost_scope"]})
        identity = copy.deepcopy(old_bundle["comparison_identity"])
        recipe = read(ROOT / arm["contract"]["path"])
        identity["loss_identity"] = recipe["loss_identity"]
        identity["evaluation_role_identity"] = recipe["development_role_identity"]
        identity["selection_role_identity"] = recipe["development_role_identity"]
        identity["runtime_certificate_scope"] = canonical_fingerprint({k: certificate[k] for k in (
            "platform_id", "accelerator", "precision", "tf32_enabled", "deterministic_algorithms", "runtime_fingerprint", "software_fingerprint", "determinism_fingerprint", "calibration_fixture_sha256")})
        infos[a] = dict(context=context, directory=directory, trajectory=trajectory, trajectory_path=trajectory_path,
            action=action, acceptance_path=acceptance_path, acceptance=acceptance, identity=identity, costs=costs, roles=roles, outcome=outcome, decision=decision)
    assert infos[ARMS[0]]["identity"]["runtime_certificate_scope"] == infos[ARMS[1]]["identity"]["runtime_certificate_scope"]
    # Bind actual observations through the existing V5 classifier, never assign strict_ready directly.
    maps = {}
    for a, info in infos.items():
        directory = info["directory"]
        paths = {"checkpoint": directory / "selected_model.pt", "runtime_certificate": directory / "runtime_certificate.json",
            "prediction_manifest": directory / "output_manifest.json", "row_manifest": ROOT / old_bundle["row_manifest_ref"],
            "target_manifest": ROOT / old_bundle["target_manifest_ref"], "trace_manifest": directory / "canonical_trace.json.context.json",
            "role_history": OUT / a / "role_history.json", "target_transform_asset": ROOT / old_bundle["target_transform_asset_ref"],
            "cost_records": OUT / a / "cost_records.json", "acceptance": info["acceptance_path"], "decision": OUT / "decision.md",
            "source_config": ROOT / next(x for x in plan["arms"] if x["arm_id"] == a)["contract"]["path"], "paired_analysis": OUT / "scientific_metrics.json"}
        required = REQUIRED_CANDIDATE_OBSERVED_BINDINGS if a == ARMS[1] else REQUIRED_OBSERVED_BINDINGS
        maps[a] = {"comparison_identity": info["identity"], "artifacts": {k: "complete" for k in required},
            "artifact_bindings": {k: {"ref": relative(paths[k]), "sha256": file_digest(paths[k])} for k in required},
            "prediction_status": "complete", "row_alignment_status": "aligned", "terminal_complete": True,
            "runtime_certificate_status": "accepted", "role_status": "complete", "trace_status": "complete",
            "stochasticity_status": "single_seed_not_estimated", "role_applicability_plan": read(OWNER / "comparison_prelaunch_pretrained_consistency_teacher.json")["role_applicability_plan"],
            "observed_role_event_kinds": sorted({r["access_kind"] for r in info["roles"]}),
            "trace_field_availability": read(OWNER / "comparison_prelaunch_pretrained_consistency_teacher.json")["trace_plan"]}
    reference = copy.deepcopy(old_bundle)
    reference.update(reference_id=evidence_ids[ARMS[0]], reference_bundle_id="k1-pretrained-combo-same-job-control-v1",
        comparison_identity=infos[ARMS[0]]["identity"], source_commit_or_archive=prep["source_commit"],
        checkpoint_identity=file_digest(RAW / ARMS[0] / "selected_model.pt"),
        contract_ref=relative(RAW / ARMS[0] / "training_contract.json"), runtime_certificate_ref=relative(RAW / ARMS[0] / "runtime_certificate.json"),
        trace_manifest_ref=relative(RAW / ARMS[0] / "canonical_trace.json.context.json"),
        role_history_ref=relative(OUT / ARMS[0] / "role_history.json"), cost_records_ref=relative(OUT / ARMS[0] / "cost_records.json"),
        acceptance_ref=relative(infos[ARMS[0]]["acceptance_path"]), decision_ref=relative(OUT / "decision.md"))
    reference["prediction_manifest"]["prediction_sha256"] = file_digest(RAW / ARMS[0] / "development_predictions.pt")
    validate_reference_bundle(reference)
    write(OUT / "reference_bundle.json", reference)
    maps[ARMS[0]].update(reference_bundle_id=reference["reference_bundle_id"], reference_bundle_sha256=reference_bundle_digest(reference))
    comparison = assess_comparison_readiness(candidate_id=evidence_ids[ARMS[1]], candidate=maps[ARMS[1]],
        reference_id=evidence_ids[ARMS[0]], reference=maps[ARMS[0]], declared_intervention_fields=["loss_identity"],
        experiment_purpose="training_objective_comparison", intervention_group_id="k1-pretrained-combo-objective")
    validate_comparison_readiness(comparison, evidence_verifier=lambda ref, sha: verify_bound_artifact(ROOT, ref, sha))
    assert comparison["strict_ready"], comparison["blocker_codes"]
    write(OUT / "comparison_readiness.json", comparison)
    jobs = []
    for a, info in infos.items():
        trace = read(info["directory"] / "canonical_trace.json")
        identity = info["identity"]
        manifest = {"schema": "molgap-trace-manifest-v1", "trajectory_id": info["trajectory"]["trajectory_id"],
            "run_id": info["action"]["run_ids"][0], "comparison_role": "reference" if a == ARMS[0] else "candidate",
            "reference_id": evidence_ids[ARMS[0]], "contract_ref": relative(OWNER / "protocol.md"),
            "terminal_evidence_ref": relative(info["acceptance_path"]), "model_identity": identity["architecture_config_identity"],
            "evaluation_role_identity": identity["evaluation_role_identity"], "selection_semantics": identity["checkpoint_selection_identity"],
            "weight_semantics": "live", "metric_semantics": "Gap MAE in eV", "presentation_semantics_ref": relative(OWNER / "protocol.md"),
            "trace_artifact_ref": relative(info["directory"] / "canonical_trace.json"), "trace_artifact_sha256": file_digest(info["directory"] / "canonical_trace.json"),
            "x_axis": "optimizer_steps", "exposure": {"optimizer_steps": 31240, "sample_presentations": 3998720},
            "trace_fields": maps[a]["trace_field_availability"], "backtest_eligibility": {"eligible": True, "exclusion_reasons": []},
            "comparability_identity": {"scientific_contract": relative(OWNER / "protocol.md"), "dataset_identity": identity["dataset_identity"],
                "row_split_identity": identity["row_membership_identity"], "architecture_identity": identity["architecture_config_identity"],
                "optimizer_identity": identity["optimizer_identity"], "lr_schedule_identity": identity["schedule_identity"],
                "target_transform_identity": identity["target_transform_identity"], "precision_identity": "fp32", "ema_semantics": "none",
                "evaluation_role_identity": identity["evaluation_role_identity"], "selection_role_identity": identity["selection_role_identity"],
                "terminal_endpoint_identity": "endpoint_31240", "matched_architecture_required": True, "x_axis_semantics": "optimizer_steps"}}
        source_paths = [p for p in info["directory"].iterdir() if p.is_file()]
        source_paths += [OWNER / "protocol.md", OWNER / "role_plan.json", OWNER / "initialization_provenance.json",
            OWNER / "experiment_spec.json", OWNER / "launch_receipt_v1.json", OWNER / "submission_response_v1.json", OWNER / "terminal_remote_observation_v1.json",
            OUT / "decision.md", OUT / "attribution.md", OUT / "scientific_metrics.json", info["acceptance_path"], OUT / a / "role_history.json", OUT / a / "cost_records.json",
            ROOT / old_bundle["row_manifest_ref"], ROOT / old_bundle["target_manifest_ref"], ROOT / old_bundle["target_transform_asset_ref"]]
        source_paths += [ROOT / x["path"] for x in metrics["teacher_retained_pins"]]
        if a == ARMS[1]:
            source_paths += [OUT / "comparison_readiness.json", OUT / "reference_bundle.json"]
            source_paths += [ROOT / x["ref"] for x in comparison["reference_artifact_bindings"].values()]
        hashes = {relative(p): file_digest(p) for p in source_paths}
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL", "evidence_id": evidence_ids[a],
            "track": "B", "scope": "desktop_k1_pretrained_consistency_teacher_100k_" + a, "legacy_contract": "k1-pretrained-combo-kaggle1-v1",
            "outcome": info["outcome"], "role_use": role_use,
            "authority": {"pointers": [relative(OWNER / "protocol.md"), relative(OUT / "decision.md"), relative(info["acceptance_path"]), relative(OUT / "scientific_metrics.json"), relative(OUT / "attribution.md")]},
            "artifacts": [{"name": p, "locator": p, "sha256": sha, "availability": "locally_retained_hash_verified",
                "artifact_type": "training_trace" if p.endswith("/canonical_trace.json") else
                    "analysis" if "objective_trace" in p else "evidence_metadata"} for p, sha in sorted(hashes.items())],
            "migration": {"migrated_at": stamp, "verification_scope": "Retained artifact acceptance and saved prediction analysis; no new training/inference", "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False},
            "observed_execution": {"training_executed": True, "inference_executed": False, "execution_ref": relative(OWNER / "terminal_remote_observation_v1.json")}}
        validate_v5_evidence_envelope(evidence, repo_root=ROOT)
        terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": info["trajectory"]["trajectory_id"],
            "run_id": info["action"]["run_ids"][0], "action_id": "A001", "finalized_at": stamp,
            "acceptance_ref": relative(info["acceptance_path"]), "artifact_hashes": hashes, "evidence": evidence,
            "decision": info["decision"], "costs": info["costs"], "roles": info["roles"], "trace_manifest": manifest,
            "same_run_observation": {"schema": "molgap-same-run-observation-v1", "spec_identity": spec.identity,
                "logical_run_id": spec.to_dict()["logical_run_id"], "platform_name": "kaggle", "platform_run_reference": info["context"].run_reference,
                "attempt_id": info["action"]["attempt_ids"][0], "source_commit": prep["source_commit"], "source_package_sha256": info["context"].source_archive_sha256}}
        if a == ARMS[1]:
            terminal["comparison_readiness_ref"] = relative(OUT / "comparison_readiness.json")
        terminal_path = OUT / a / "terminal.json"
        write(terminal_path, terminal)
        jobs.append({"arm_identifier": a, "trajectory": relative(info["trajectory_path"]), "terminal": relative(terminal_path), "trace": manifest["trace_artifact_ref"]})
    results = close_terminal_multi_arm(ROOT, jobs)
    write(OUT / "closure_receipt.json", {"results": results})
    print(json.dumps({"strict_ready": comparison["strict_ready"], "results": results}, default=str))


if __name__ == "__main__":
    main()
