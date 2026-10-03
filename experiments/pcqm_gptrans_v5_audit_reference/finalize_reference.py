"""Identity-specific metadata adapter over the shared terminal/RML entry point.

Reads accepted JSON and one development prediction tensor; no model loading,
graph loading, inference, training, platform submission or protected roles.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from molgap.constants import REPO_ROOT

from molgap.research_memory.terminal_wiring import close_terminal_arm
from molgap.research_memory.trace import atomic_write, canonicalize_trace, file_digest, json_bytes
from molgap.comparison_readiness import validate_reference_bundle


def main() -> None:
    root = REPO_ROOT
    directory = Path(__file__).resolve().parent
    results = directory / "results/terminal"
    relative = lambda p: p.relative_to(root).as_posix()
    load = lambda p: json.loads(p.read_bytes())
    save = lambda p, obj: atomic_write(p, json_bytes(obj))
    record_root = root / "platforms/_records/kaggle/training/gptrans_v5_audit_reference"
    retained = record_root / "segment_06/gptrans_v5_audit_reference"
    training = retained / "training"
    final = load(directory / "results/final_acceptance.json")
    content = load(directory / "results/final_prediction_acceptance.json")
    plan_path = directory / "rml_plan/trajectory.json"
    plan = load(plan_path)
    tid = plan["trajectory_id"]
    run = final["kernel_id"]
    assert plan["actions"][0]["run_ids"] == [run]
    assert final["accepted"] and content["accepted"]
    timestamp = (load(results / "terminal.json")["finalized_at"]
                 if (results / "terminal.json").exists() else datetime.now(timezone.utc).isoformat())
    refs = [directory / f"results/segment_{i:02d}_operation.json" for i in range(1, 7)]
    for i, path in enumerate(refs, 1):
        observed = load(path)["accepted_segment"]
        assert (observed["kernel_id"], observed["version"], observed["status"]) == (run, i, "COMPLETE")
        assert observed["source_archive_sha256"] == final["source_archive_sha256"]
        assert observed["fixed_manifest_sha256"] == final["fixed_manifest_sha256"]
    raw_path = training / "canonical_trace.json"
    assert file_digest(raw_path) == final["canonical_trace_sha256"]
    raw = load(raw_path)
    assert raw["trajectory_id"] == tid and raw["run_id"] == "gptrans-t-v5-audit-reference-s42"
    alias = {"format": "molgap-trace-run-alias-v1", "trajectory_id": tid,
             "raw_run_id": raw["run_id"], "prospective_run_id": run,
             "raw_trace_ref": relative(raw_path), "raw_trace_sha256": file_digest(raw_path),
             "operation_receipts": [{"ref": relative(p), "sha256": file_digest(p)} for p in refs],
             "semantics": "Header-only identity reconciliation; all observed rows preserved unchanged"}
    save(results / "run_alias.json", alias)
    trace = copy.deepcopy(raw)
    trace["run_id"] = run
    trace.setdefault("provenance", []).append({"source": relative(raw_path), "sha256": file_digest(raw_path),
                                               "run_alias_ref": relative(results / "run_alias.json"),
                                               "operation": "receipt-bound header-only run alias"})
    trace = canonicalize_trace(trace)
    assert trace["observations"] == raw["observations"] and len(trace["observations"]) == 60
    save(results / "canonical_trace.json", trace)

    # Tensor-only content verification reuses the already frozen 50K development role.
    import torch
    payload_path = training / "development_predictions.pt"
    assert file_digest(payload_path) == content["development_predictions_sha256"]
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    tensor_sha = lambda t: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    prediction, target, rows = (payload[k].view(-1) for k in ("prediction_eV", "target_eV", "source_idx"))
    assert torch.equal(rows, torch.arange(100000, 150000))
    assert prediction.numel() == target.numel() == 50000
    assert torch.isfinite(prediction).all() and torch.isfinite(target).all()
    mae = float((prediction.double() - target.double()).abs().mean())
    assert abs(mae - content["best_development_mae_eV"]) < 1e-12
    for flag in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        assert payload[flag] is False
    old_directory = root / "experiments/pcqm_gptrans_author_alignment/recovered_reference"
    old_bundle = load(old_directory / "reference_bundle.json")
    prediction_manifest = {
        "prediction_sha256": tensor_sha(prediction), "source_idx_sha256": tensor_sha(rows),
        "target_sha256": tensor_sha(target), "ordering_semantics": "source_idx ascending",
        "evaluation_role_identity": old_bundle["prediction_manifest"]["evaluation_role_identity"],
        "row_count": 50000, "unique_source_idx": int(torch.unique(rows).numel()),
    }
    assert prediction_manifest["source_idx_sha256"] == old_bundle["prediction_manifest"]["source_idx_sha256"]
    assert prediction_manifest["target_sha256"] == old_bundle["prediction_manifest"]["target_sha256"]
    save(results / "prediction_manifest.json", {
        **prediction_manifest, "format": "molgap-aligned-prediction-manifest-v1",
        "artifact_locator": relative(payload_path), "artifact_sha256": file_digest(payload_path),
        "development_gap_mae_eV": mae, "tensor_hash_semantics": "sha256-contiguous-c-order-bytes",
    })
    preflight = load(retained / "preflight/preflight.json")
    save(results / "runtime_certificate.json", preflight["runtime_certificate"])
    row_manifest = root / old_bundle["row_manifest_ref"]
    role_use = {"internal_train": "consumed", "internal_development": "consumed",
                "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched"}
    metadata_ref = relative(results / "observed_metadata.json")
    roles = []
    for role, kinds in (("internal_train", ("training_membership", "labels_read", "metric_computed")),
                        ("internal_development", ("prediction_input", "labels_read", "metric_computed", "selection_used"))):
        for kind in kinds:
            roles.append({"schema": "molgap-role-event-v1", "role_event_id": f"role-{tid}-{role}-{kind}",
                          "trajectory_id": tid, "action_id": "A001", "run_id": run,
                          "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1",
                          "row_manifest_hash": file_digest(row_manifest), "role_name": role,
                          "access_kind": kind, "selection_used": kind == "selection_used", "evidence_ref": metadata_ref})
    costs = []
    native_refs = []
    for i in range(1, 7):
        native_path = record_root / f"segment_{i:02d}/gptrans_v5_audit_reference/native_cost.json"
        native = load(native_path)
        acceptance = load(directory / f"results/segment_{i:02d}_acceptance.json")
        hours = native["wall_seconds"] * native["allocated_gpu_count"] / 3600
        assert abs(hours - acceptance["allocated_device_hours"]) < 1e-10
        assert native["source_archive_sha256"] == final["source_archive_sha256"]
        native_refs.append({"ref": relative(native_path), "sha256": file_digest(native_path)})
        costs.append({"schema": "molgap-cost-event-v1", "cost_event_id": f"cost-{tid}-observed-v{i}",
                      "trajectory_id": tid, "action_id": "A001", "run_id": run, "attempt_id": f"v{i}",
                      "category": "training", "platform": "kaggle2", "hardware": "Tesla_T4_x2_allocated_x1_used",
                      "measurement": {"device_hours": {"status": "measured", "value": hours},
                                      "wall_hours": {"status": "measured", "value": native["wall_seconds"] / 3600},
                                      "cpu_hours": {"status": "measurement_missing", "value": None},
                                      "queue_hours": {"status": "measurement_missing", "value": None}},
                      "evidence_ref": metadata_ref})
    assert abs(sum(c["measurement"]["device_hours"]["value"] for c in costs) - final["measured_allocated_t4_device_hours"]) < 1e-10
    outcome = {"execution_status": "complete", "artifact_status": "accepted",
               "comparison_status": "context_only", "scientific_status": "reference_observation_complete",
               "transfer_status": "not_evaluated", "budget_decision": "stop_under_contract", "full_handoff_status": "not_authorized"}
    decision = {"outcome": "CLOSED", "final": True, "decision_ref": relative(results / "decision.md"),
                "next_allowed_actions": ["CLOSE"], "reopen_conditions": []}
    save(results / "observed_metadata.json", {
        "format": "molgap-gptrans-audit-observed-terminal-v1", "trajectory_id": tid, "run_id": run,
        "evidence_id": "pcqm-gptrans-v5-audit-reference-s42", "outcome": outcome, "trajectory_decision": decision,
        "role_use": role_use, "roles": roles, "costs": costs, "native_cost_sources": native_refs,
        "observed_role_sources": [relative(training / "completion_manifest.json"), relative(training / "frozen_reference.json"),
                                  relative(directory / "results/final_prediction_acceptance.json"), relative(raw_path)],
        "infrastructure_attempt": load(refs[0])["infrastructure_attempt"],
        "infrastructure_attempt_cost": "measurement_missing_not_in_successful_training_total",
        "training_executed_remotely": True, "training_executed_locally": False, "model_inference_executed_locally": False,
    })
    analysis = {"format": "molgap-gptrans-live-ema-trajectory-analysis-v1", "scope": "descriptive_single_reference_run",
                "observations": len(trace["observations"]), "final_ema_minus_live_eV": trace["observations"][-1]["ema_dev_metric"] - trace["observations"][-1]["live_dev_metric"],
                "checkpoints": [{k: row[k] for k in ("epoch_or_pass", "optimizer_step", "sample_presentations", "learning_rate", "live_train_metric", "live_dev_metric", "ema_dev_metric")}
                                for row in trace["observations"] if row["epoch_or_pass"] in (10, 20, 30, 40, 50, 60)],
                "final_ten_ema_change_eV": trace["observations"][-1]["ema_dev_metric"] - trace["observations"][-11]["ema_dev_metric"],
                "ema_decay": 0.9999, "ema_half_life_optimizer_steps": 6931.125226233421,
                "ema_half_life_train_passes": 6931.125226233421 / 781,
                "interpretation": "EMA lag is consistent with the live/EMA gap; its causal effect or superiority of an alternative selection policy was not tested",
                "no_successor_release": True}
    save(results / "trajectory_analysis.json", analysis)
    artifacts = [results / "run_alias.json", results / "canonical_trace.json", results / "prediction_manifest.json",
                 results / "runtime_certificate.json", results / "observed_metadata.json", results / "trajectory_analysis.json",
                 directory / "results/final_acceptance.json", directory / "results/final_prediction_acceptance.json",
                 training / "completion_manifest.json", training / "best_model.pt", payload_path, training / "last_checkpoint.pt"]
    authorities = [directory / "protocol.md", directory / "contract.json", directory / "role_plan.json", plan_path,
                   directory / "decision.md", results / "decision.md"]
    bindings = {relative(p): file_digest(p) for p in artifacts + authorities + refs}
    bindings.update({p["ref"]: p["sha256"] for p in native_refs})
    evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
                "evidence_id": "pcqm-gptrans-v5-audit-reference-s42", "track": "C", "scope": "matched_reference_observation_rerun_100k",
                "legacy_contract": "none-prospective-v5", "outcome": outcome, "role_use": role_use,
                "authority": {"pointers": [relative(p) for p in authorities]},
                "artifacts": [{"name": p.name, "locator": relative(p), "sha256": file_digest(p), "availability": "local_verified"} for p in artifacts],
                "migration": {"migrated_at": timestamp, "verification_scope": "metadata/SHA and saved development tensors only; no local model inference",
                              "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
    manifest = copy.deepcopy(load(old_directory / "trace_manifest.json"))
    manifest.update(trajectory_id=tid, run_id=run, contract_ref=relative(directory / "contract.json"),
                    presentation_semantics_ref=relative(directory / "contract.json"), trace_artifact_ref=relative(results / "canonical_trace.json"),
                    terminal_evidence_ref=relative(results / "terminal_evidence.json"),
                    backtest_eligibility={"eligible": False, "exclusion_reasons": ["Reference observation rerun, not an accepted causal candidate/reference experiment"]})
    manifest["comparability_identity"]["scientific_contract"] = "gptrans-v5-observation-reference-fp32-bs128-60epochs"
    manifest["trace_fields"] = {k: True for k in ("optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate", "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity")}
    package = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": tid, "action_id": "A001", "run_id": run,
               "finalized_at": timestamp, "acceptance_ref": metadata_ref, "artifact_hashes": bindings,
               "evidence": evidence, "decision": decision, "roles": roles, "costs": costs, "role_use": role_use, "trace_manifest": manifest}
    save(results / "terminal.json", package)
    closure = close_terminal_arm(repo_root=root, trajectory=plan_path, terminal=results / "terminal.json", trace=results / "canonical_trace.json")
    print(json.dumps(closure, indent=2))
    assert closure["pipeline_status"] == "COMPLETE"

    # A separately identifiable reference bundle; never relabel the historical V4 run.
    finalized = directory / "rml_plan/rml_finalized"
    bundle = copy.deepcopy(old_bundle)
    bundle.update(reference_bundle_id="reference-gptrans-v5-audit-100k-s42", reference_id=evidence["evidence_id"],
                  contract_ref=relative(directory / "contract.json"), source_commit_or_archive=plan["state_at_start"]["source_commit"],
                  checkpoint_identity=final["best_ema_model_sha256"], runtime_certificate_ref=relative(results / "runtime_certificate.json"),
                  prediction_manifest=prediction_manifest, trace_manifest_ref=relative(finalized / "trace_manifest.json"),
                  role_history_ref=metadata_ref, cost_records_ref=metadata_ref, acceptance_ref=metadata_ref, decision_ref=decision["decision_ref"])
    observed_reference = load(training / "frozen_reference.json")
    for observed_key, field in (("data_role_fingerprint", "data_role_identity"), ("row_order_fingerprint", "row_order_identity"),
                                ("feature_fingerprint", "feature_identity"), ("optimizer_fingerprint", "optimizer_identity"),
                                ("schedule_fingerprint", "schedule_identity"), ("target_transform_fingerprint", "target_transform_identity"),
                                ("architecture_fingerprint", "architecture_config_identity"), ("selection_fingerprint", "checkpoint_selection_identity")):
        assert observed_reference[observed_key] == bundle["comparison_identity"][field]
    # The 0.003 policy gate in the runner is not a measured repeatability floor.
    assert bundle["stochasticity"]["training_stochasticity"]["stochasticity_floor_eV"] is None
    validate_reference_bundle(bundle)
    save(results / "reference_bundle.json", bundle)


if __name__ == "__main__":
    main()
