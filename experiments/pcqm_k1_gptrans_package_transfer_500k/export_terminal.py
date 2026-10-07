"""Translate accepted native 500K artifacts into the shared RML finalizer inputs."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from molgap.research_memory.finalize import finalize
from molgap.research_memory.recovery import recover_trace
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
ARMS = ("k1_pretrained_consistency", "gptrans_g1_bond_local_ema999")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checked = EXP / "submission_v5/terminal_inspection"
    inspection = load(checked / "inspection_report.json")
    analysis = load(EXP / "terminal_acceptance/analysis.json")
    if not inspection["mechanical_checks_passed"] or not analysis["frozen_complementarity_gate"]["passed"]:
        raise ValueError("This export requires the retained mechanical pass and frozen complementarity nomination")
    raw = ROOT / "platforms/_records/kaggle/staging/pcqm_k1_gptrans_package_transfer_500k/v5-checked-artifacts"
    retained = ROOT / "platforms/_records/kaggle/training/pcqm_k1_gptrans_package_transfer_500k_s42_final"
    finalized_at = datetime.now(timezone.utc).isoformat()
    results = []
    for arm in ARMS:
        prospective = EXP / "kaggle1_v1" / arm / "trajectory.json"
        frozen = load(prospective)
        if frozen["state_at_start"]["reference_ids"]:
            raise ValueError("Do not apply this no-reference closure to another comparison contract")
        trajectory_id = frozen["trajectory_id"]
        run_id = frozen["actions"][0]["run_ids"][0]
        evidence_id = "pcqm-" + arm.replace("_", "-") + "-kaggle1-500k-s42-terminal"
        stage = checked / "evidence/stages" / arm
        contract = load(stage / "scientific_contract.json")
        rows = load(stage / "trace.json")["epochs"]
        destination = EXP / "terminal_acceptance" / arm
        destination.mkdir(parents=True, exist_ok=True)
        acceptance_ref = rel(destination / "acceptance.json")
        decision = {"outcome": "POSITIVE_UNDER_CONTRACT",
                    "decision_ref": rel(EXP / "terminal_acceptance/decision.md"),
                    "next_allowed_actions": [],
                    "reopen_conditions": ["New authorized prospective full-data scale-transfer contract; no automatic execution."]}
        outcome = {"execution_status": "complete",
                   "artifact_status": "complete_selected_and_resume_artifacts_hash_verified",
                   "comparison_status": "fixed50_paired_endpoint_complete_component_strict_reference_missing",
                   "scientific_status": "POSITIVE_UNDER_CONTRACT",
                   "transfer_status": "positive_complementarity_nomination_component_attribution_inconclusive",
                   "budget_decision": "observed_subtotal_below_ceiling_historical_total_unknown",
                   "full_handoff_status": "not_released"}
        role_use = {"train": "consumed", "internal_development": "consumed",
                    "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
        roles = []
        for role, kinds in (("train", ("training_membership", "labels_read")),
                            ("internal_development", ("prediction_input", "labels_read", "metric_computed", "selection_used"))):
            for kind in kinds:
                roles.append({"schema": "molgap-role-event-v1",
                              "role_event_id": f"role-{trajectory_id}-{role}-{kind}",
                              "trajectory_id": trajectory_id, "action_id": "A001", "run_id": run_id,
                              "dataset_identity": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
                              "row_manifest_hash": file_digest(stage / "data_manifest.json"),
                              "role_name": role, "access_kind": kind,
                              "selection_used": role == "internal_development", "evidence_ref": acceptance_ref})
        epoch_seconds = sum(row["seconds"] for row in rows)
        costs = [{"schema": "molgap-cost-event-v1",
                  "cost_event_id": f"cost-{trajectory_id}-measured-epoch-windows",
                  "trajectory_id": trajectory_id, "action_id": "A001", "run_id": run_id,
                  "attempt_id": "kaggle1-package-transfer-001:retained-epoch-windows",
                  "platform": "Kaggle1", "hardware": "Tesla T4", "category": "training",
                  "evidence_ref": acceptance_ref,
                  "measurement": {"device_hours": {"status": "measured", "value": epoch_seconds / 3600},
                                  "wall_hours": {"status": "measured", "value": epoch_seconds / 3600},
                                  "cpu_hours": {"status": "measurement_missing", "value": None},
                                  "queue_hours": {"status": "measurement_missing", "value": None}}}]
        inventory = []
        # Copy only final state/selected output needed for later replay or inference.
        names = ["initial_state.pt", "best_model.pt", "last_checkpoint.pt", "best_predictions.pt"]
        if arm == ARMS[1]:
            names.append("best_live_predictions.pt")
        for name in names:
            source = raw / arm / name
            target = retained / arm / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = file_digest(source)
            if not target.exists():
                shutil.copy2(source, target)
            if file_digest(target) != digest:
                raise ValueError("Durable retained output changed")
            inventory.append({"name": name, "local_path": rel(target), "sha256": digest,
                              "remote_locator": f"kaggle://nothingnessvoid/molgap-k1-gptrans-500k-pair-s42-v1/version/5/output/evidence/stages/{arm}/{name}",
                              "producer_version": 4 if arm == ARMS[1] or name in {"initial_state.pt", "best_model.pt", "best_predictions.pt"} else 5,
                              "retention": "local_exact_bytes_and_retrievable_version5_copy_with_original_producer_lineage"})
        inventory_path = destination / "artifact_inventory.json"
        atomic_write(inventory_path, json_bytes({"format": "molgap-terminal-artifact-inventory-v1", "files": inventory}))
        acceptance = {"format": "molgap-native500k-terminal-acceptance-v1",
                      "evidence_id": evidence_id, "run_id": run_id, "outcome": outcome,
                      "trajectory_decision": decision, "role_use": role_use, "roles": roles, "costs": costs,
                      "mechanical": inspection["arms"][arm],
                      "nomination_scope": "joint_fixed50_complementarity_not_standalone_component_causal_promotion",
                      "component_nomination": "INCONCLUSIVE_missing_qualified_reference",
                      "fixed50_analysis_ref": rel(EXP / "terminal_acceptance/analysis.json"),
                      "inspection_ref": rel(checked / "inspection_report.json"),
                      "artifact_inventory_ref": rel(inventory_path),
                      "cost_semantics": "One assigned native T4 epoch-window elapsed duration (training plus development); excludes qualification/bootstrap/idlepeer. Subset of pair allocation, never add both ledgers.",
                      "pair_allocation_subtotal_T4_hours": inspection["v2_plus_v3_plus_v4_plus_v5_measured_allocated_T4_hours"],
                      "pair_cost_missing": inspection["cost_missing"],
                      "strict_replay_ready": False}
        atomic_write(destination / "acceptance.json", json_bytes(acceptance))
        trace_path = destination / "canonical_trace.json"
        metric = lambda weights, role: {"metric": "MAE", "unit": "eV", "target": "Gap",
                                        "role_identity": role, "weights": weights, "direction": "minimize"}
        recovery_spec = {"trajectory_id": trajectory_id, "run_id": run_id, "rows_key": "epochs",
                         "metric_semantics": {"live_train_metric": metric("live", "training_dropout_supervised_L1"),
                                              "live_dev_metric": metric("live", "internal_development_500000_550000"),
                                              "ema_dev_metric": metric("ema", "internal_development_500000_550000") if arm == ARMS[1] else None},
                         "device_time_semantics": "sum_over_devices",
                         "field_mapping": {"epoch_or_pass": "epoch", "optimizer_step": "global_step",
                                           "sample_presentations": "sample_presentations", "learning_rate": "lr",
                                           "live_train_metric": "train_mae_eV", "live_dev_metric": "live_development_mae_eV",
                                           "ema_dev_metric": "ema_development_mae_eV", "wall_time_seconds": "seconds", "device_time_seconds": "seconds"}}
        if not trace_path.exists():
            recover_trace([stage / "trace.json"], recovery_spec, trace_path, repo_root=ROOT)
        else:
            raise ValueError("Use existing immutable finalization receipt; do not silently regenerate a terminal package")
        identity = {"scientific_contract": contract["benchmark_id"], "dataset_identity": contract["data_role_fingerprint"],
                    "row_split_identity": "train0:500000-internaldevelopment500000:550000",
                    "architecture_identity": f"{arm}:{contract['parameters']}",
                    "optimizer_identity": contract["optimizer_fingerprint"], "lr_schedule_identity": contract["schedule_fingerprint"],
                    "target_transform_identity": contract["target_transform_fingerprint"], "precision_identity": "fp32_noTF32",
                    "ema_semantics": "EMA0.999_each_optimizer_step" if arm == ARMS[1] else "none",
                    "evaluation_role_identity": "internal_development_500000_550000",
                    "selection_role_identity": contract["selection_fingerprint"], "x_axis_semantics": "optimizer_steps",
                    "terminal_endpoint_identity": "complete60epochs_234360steps_29998080presentations", "matched_architecture_required": False}
        trace_manifest = {"schema": "molgap-trace-manifest-v1", "trajectory_id": trajectory_id, "run_id": run_id,
                          "reference_id": evidence_id, "comparison_role": "reference",
                          "contract_ref": rel(EXP / "protocol.md"), "model_identity": identity["architecture_identity"],
                          "x_axis": "optimizer_steps", "presentation_semantics_ref": rel(EXP / "protocol.md"),
                          "weight_semantics": "ema_and_live" if arm == ARMS[1] else "live",
                          "metric_semantics": "Gap MAE in eV; training dropout L1 differs from clean development mode",
                          "evaluation_role_identity": identity["evaluation_role_identity"], "selection_semantics": contract["selection_fingerprint"],
                          "trace_artifact_ref": rel(trace_path), "trace_artifact_sha256": file_digest(trace_path),
                          "terminal_evidence_ref": acceptance_ref, "comparability_identity": identity,
                          "backtest_eligibility": {"eligible": False, "exclusion_reasons": ["missing_frozen_reference", "strict_component_comparison_unqualified", "development_used_for_checkpoint_selection", "historical_total_allocation_incomplete"]},
                          "exposure": {"optimizer_steps": 234360, "sample_presentations": 29998080},
                          "trace_fields": {"epoch_or_pass": True, "optimizer_step": True, "sample_presentations": True,
                                           "learning_rate": True, "live_train_metric": True, "live_dev_metric": True,
                                           "ema_dev_metric": arm == ARMS[1], "checkpoint_identity": False}}
        authority = [rel(EXP / "protocol.md"), decision["decision_ref"], rel(EXP / "terminal_acceptance/analysis_zh.md"),
                     acceptance_ref, rel(checked / "inspection_report.json")]
        artifacts = []
        for name, path in [("accepted_training_trace", stage / "trace.json"),
                           ("stage_manifest", stage / "stage_manifest.json"),
                           ("runtime_certificate", stage / "runtime_certificate.json"),
                           ("scientific_contract", stage / "scientific_contract.json"),
                           ("data_manifest", stage / "data_manifest.json"),
                           ("final_selected_artifact_inventory", inventory_path),
                           ("fixed50_prediction_analysis", EXP / "terminal_acceptance/analysis.json")]:
            artifact = {"name": name, "locator": rel(path), "sha256": file_digest(path), "availability": "committed_metadata_verified"}
            if name == "accepted_training_trace":
                artifact["artifact_type"] = "training_trace"
            artifacts.append(artifact)
        # RML binds portable metadata; inventory binds exact retained/remote binary outputs.
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
                    "evidence_id": evidence_id, "track": "B", "scope": "teacher_free_composed500k_fixed50_complementarity_nomination:" + arm,
                    "legacy_contract": "pcqm-composed500k-dev50k-60pass-v1", "outcome": outcome,
                    "authority": {"pointers": authority}, "artifacts": artifacts, "role_use": role_use,
                    "migration": {"migrated_at": finalized_at, "verification_scope": "new terminal metadata wrapping retained remote execution and fixed-prediction acceptance",
                                  "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False},
                    "observed_execution": {"training_executed": True, "inference_executed": False,
                                           "execution_ref": rel(checked / "inspection_report.json")}}
        binding_paths = set(authority) | {a["locator"] for a in artifacts} | {rel(trace_path),
            rel(checked / "remote_identity_verified.json"), rel(checked / "retrieval_manifest.json"),
            rel(checked / "evidence/invocation_cost.json"), rel(checked / "evidence/pair_state.json"),
            rel(checked / "evidence/cpu_release_report.json")}
        terminal = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": trajectory_id, "run_id": run_id,
                    "action_id": "A001", "finalized_at": finalized_at, "acceptance_ref": acceptance_ref,
                    "artifact_hashes": {pointer: file_digest(ROOT / pointer) for pointer in sorted(binding_paths)},
                    "evidence": evidence, "decision": decision, "costs": costs, "roles": roles, "trace_manifest": trace_manifest}
        terminal_path = destination / "terminal.json"
        atomic_write(terminal_path, json_bytes(terminal))
        results.append(finalize(ROOT, prospective, terminal_path, trace_path))
    atomic_write(EXP / "terminal_acceptance/finalization_summary.json", json_bytes({"finalized_at": finalized_at,
                 "dual_replay_ready": False, "strict_replay_exclusions": ["missing_frozen_reference", "strict_component_comparison_unqualified"],
                 "receipts": results}))
    print(json.dumps({"finalized": [{"trajectory_id": r["trajectory_id"], "outcome": r["outcome"], "trace_status": r["trace_status"]} for r in results]}))


if __name__ == "__main__":
    main()
