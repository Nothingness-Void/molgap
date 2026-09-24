"""Prepare retained terminal metadata for the generic RML finalizer.

This adapter reads accepted prediction/checkpoint metadata only.  It performs
no model construction, inference, training, or protected-role access.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from molgap.comparison_readiness import (
    REQUIRED_CANDIDATE_OBSERVED_BINDINGS,
    REQUIRED_OBSERVED_BINDINGS,
    assess_comparison_readiness,
    validate_comparison_readiness,
)
from molgap.constants import REPO_ROOT
from molgap.evidence_pointers import verify_bound_artifact
from molgap.research_memory.trace import canonicalize_trace, file_digest, json_bytes


ROOT = REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k"
ROOT_REL = "experiments/pcqm_k1_functional_group_token_100k"
SHARED_ROOT = None
SHARED_REL = None
EXTRA_ARTIFACTS = ()
RESULTS = ROOT / "results"
REFERENCE_ROOT = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference"
REFERENCE_REL = "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference"
RECORD_ROOT = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_functional_group_token_s42_v1"
CANDIDATE_ROOT = RECORD_ROOT / "pcqm_k1_functional_group_token/neural_atom_k1_functional_group_token"
RAW_ACCEPTANCE = RESULTS / "raw_acceptance.json"
SOURCE_ACCEPTANCE_REL = "results/raw_acceptance.json"
TRAJECTORY_ID = "TC-k1-functional-group-token-100k-s42"
RUN_ID = "kaseichou/molgap-k1-functional-group-token-s42:v1"
CONTRACT_RUN_ID = "pcqm-k1-variants-100k-s42-v1-neural_atom_k1_functional_group_token"
MODE = "neural_atom_k1_functional_group_token"
DATASET_ID = "pcqm4mv2-ogb-fixed-100k-v1"
MANIFEST_SHA = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
ACTION_ID = "A002"
EVIDENCE_ID = "pcqm-k1-functional-group-token-100k-s42"
LOCAL_RECORD_URI = (
    "external://local-platform-record/kaggle/pcqm_k1_functional_group_token_s42_v1/"
    "pcqm_k1_functional_group_token/neural_atom_k1_functional_group_token"
)
DECISION_OUTCOME = "POSITIVE_BELOW_GATE"
NEXT_ALLOWED_ACTIONS = ["design one distinct chemistry-conditioned relation-content hypothesis"]
REOPEN_CONDITIONS = [
    "a new candidate must alter relation content rather than add a parallel group-token branch"
]
SCIENTIFIC_STATUS = "positive_below_gate"
FINALIZED_AT = "2026-09-20T19:00:00+09:00"
MIGRATED_AT = "2026-09-20"
PLATFORM = "kaggle2"
HARDWARE = "Tesla_T4_16GB"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    os.replace(temporary, path)


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rel(name: str) -> str:
    return f"{ROOT_REL}/{name}"


def shared_rel(name: str) -> str:
    return f"{SHARED_REL or ROOT_REL}/{name}"


def binding(path: str) -> dict[str, str]:
    return {"ref": path, "sha256": file_digest(REPO_ROOT / path)}


def role_events() -> list[dict[str, Any]]:
    specs = (
        ("training-membership", "official_train_prefix_0_100000", "training_membership", False),
        ("prediction-input", "internal_development_100000_150000", "prediction_input", False),
        ("labels-read", "internal_development_100000_150000", "labels_read", False),
        ("metric-computed", "internal_development_100000_150000", "metric_computed", False),
        ("selection-used", "internal_development_100000_150000", "selection_used", True),
    )
    return [
        {
            "schema": "molgap-role-event-v1",
            "role_event_id": f"role-{TRAJECTORY_ID}-{suffix}",
            "trajectory_id": TRAJECTORY_ID,
            "action_id": ACTION_ID,
            "run_id": RUN_ID,
            "dataset_identity": DATASET_ID,
            "row_manifest_hash": MANIFEST_SHA,
            "role_name": role,
            "access_kind": kind,
            "selection_used": selected,
            "evidence_ref": rel("results/terminal_acceptance.json"),
        }
        for suffix, role, kind, selected in specs
    ]


def make_trace(raw: dict[str, Any], checkpoint_sha: str) -> dict[str, Any]:
    elapsed = 0.0
    rows = []
    for index, epoch in enumerate(raw["epochs"]):
        elapsed += float(epoch["seconds"])
        rows.append(
            {
                "sequence": index,
                "event": "terminal" if index == len(raw["epochs"]) - 1 else "observation",
                "optimizer_step": int(epoch["optimizer_steps"]),
                "sample_presentations": int(epoch["sample_presentations"]),
                "epoch_or_pass": float(epoch["epoch"]),
                "learning_rate": float(epoch["learning_rate"]),
                "live_train_metric": float(epoch["train_normalized_mae"]),
                "live_dev_metric": float(epoch["development_gap_mae_eV"]),
                "ema_dev_metric": None,
                "wall_time_seconds": float(epoch["seconds"]),
                "cumulative_wall_time_seconds": elapsed,
                "device_time_seconds": float(epoch["seconds"]),
                "cumulative_device_time_seconds": elapsed,
                "checkpoint_identity": checkpoint_sha if index == len(raw["epochs"]) - 1 else None,
            }
        )
    return canonicalize_trace(
        {
            "trajectory_id": TRAJECTORY_ID,
            "run_id": RUN_ID,
            "device_time_semantics": "sum_over_devices",
            "metric_semantics": {
                "live_train_metric": {
                    "metric": "mean_absolute_error",
                    "unit": "normalized_target_units",
                    "target": "gap",
                    "role_identity": "official_train_prefix_0_100000",
                    "weights": "live",
                    "direction": "minimize",
                },
                "live_dev_metric": {
                    "metric": "mean_absolute_error",
                    "unit": "eV",
                    "target": "gap",
                    "role_identity": "internal_development_100000_150000",
                    "weights": "live",
                    "direction": "minimize",
                },
                "ema_dev_metric": None,
            },
            "observations": rows,
            "provenance": {
                "source_refs": [
                    f"{LOCAL_RECORD_URI}/trace.json"
                ],
                "recovery": "deterministic-field-map-plus-observed-cumulative-epoch-seconds",
            },
        }
    )


def main() -> None:
    acceptance = load(RAW_ACCEPTANCE)
    candidate = acceptance["candidates"][MODE]
    record = candidate["record"]
    completion = load(CANDIDATE_ROOT / "completion_manifest.json")
    if acceptance.get("accepted") is not True or acceptance.get("model_inference_executed") is not False:
        raise RuntimeError("no-inference acceptance is required")
    for name, expected in completion["artifact_sha256"].items():
        if file_digest(CANDIDATE_ROOT / name) != expected:
            raise RuntimeError(f"terminal artifact changed: {name}")

    payload_path = CANDIDATE_ROOT / "best_development_payload.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    reference_prediction = load(REFERENCE_ROOT / "prediction_manifest.json")
    if tensor_sha(payload["target_eV"]) != reference_prediction["target_sha256"]:
        raise RuntimeError("candidate target bytes differ from reference")
    if tensor_sha(payload["source_idx"]) != reference_prediction["source_idx_sha256"]:
        raise RuntimeError("candidate row order differs from reference")

    prediction_manifest = {
        "format": "molgap-aligned-prediction-manifest-v1",
        "artifact_locator": f"{LOCAL_RECORD_URI}/best_development_payload.pt",
        "artifact_sha256": file_digest(payload_path),
        "development_gap_mae_eV": record["training"]["development_gap_mae_eV"],
        "evaluation_role_identity": "pcqm4mv2-ogb-fixed-100k-v1:internal-development-100000-150000",
        "ordering_semantics": "source_idx ascending",
        "prediction_sha256": tensor_sha(payload["prediction_eV"]),
        "target_sha256": tensor_sha(payload["target_eV"]),
        "source_idx_sha256": tensor_sha(payload["source_idx"]),
        "row_count": int(payload["source_idx"].numel()),
        "unique_source_idx": int(torch.unique(payload["source_idx"]).numel()),
        "tensor_hash_semantics": "sha256-contiguous-c-order-bytes",
    }
    write(RESULTS / "prediction_manifest.json", prediction_manifest)
    write(RESULTS / "row_manifest.json", load(REFERENCE_ROOT / "row_manifest.json"))
    write(RESULTS / "target_manifest.json", load(REFERENCE_ROOT / "target_manifest.json"))
    write(RESULTS / "runtime_certificate.json", load(CANDIDATE_ROOT / "runtime_certificate.json"))
    write(
        RESULTS / "checkpoint_manifest.json",
        {
            "format": "molgap-checkpoint-manifest-v1",
            "run_id": RUN_ID,
            "best_epoch": record["training"]["best_epoch"],
            "best_model_sha256": record["training"]["best_model_sha256"],
            "last_checkpoint_sha256": record["training"]["checkpoint_sha256"],
            "recovery_artifacts": {
                name: digest for name, digest in completion["artifact_sha256"].items()
                if name.startswith("recovery_epoch_")
            },
            "external_root": LOCAL_RECORD_URI,
        },
    )
    paired = {
        "format": "molgap-paired-analysis-v1",
        "candidate_id": MODE,
        "reference_id": "pcqm-k1-v4-100k-reference-s42",
        "rows": 50000,
        "candidate_development_gap_mae_eV": candidate["recomputed_development_gap_mae_eV"],
        "reference_development_gap_mae_eV": acceptance["reference_development_gap_mae_eV"],
        "gain_eV": candidate["gate"]["gain_eV"],
        "paired_error_delta_bootstrap_95_eV": candidate["paired_error_delta_bootstrap_95_eV"],
        "required_gain_eV": candidate["gate"]["required_gain_eV"],
        "row_bootstrap_direction_favorable": candidate["paired_error_delta_bootstrap_95_eV"][1] < 0,
        "material_gate_passed": candidate["gate"]["passed"],
        "training_stochasticity_accounted": candidate["gate"]["training_stochasticity_accounted"],
        "row_bootstrap_is_sufficient_alone": candidate["gate"]["row_bootstrap_is_sufficient_alone"],
    }
    write(RESULTS / "paired_analysis.json", paired)

    raw_trace = load(CANDIDATE_ROOT / "trace.json")
    trace = make_trace(raw_trace, record["training"]["checkpoint_sha256"])
    write(RESULTS / "canonical_trace.json", trace)
    trace_fields = {
        "optimizer_step": True, "sample_presentations": True, "epoch_or_pass": True,
        "learning_rate": True, "live_train_metric": True, "live_dev_metric": True,
        "ema_dev_metric": False, "checkpoint_identity": True,
    }
    comparison_identity = dict(load(ROOT / "comparison_readiness_prelaunch.json")["matched_fields"])
    comparison_identity["architecture_config_identity"] = record["contract"]["architecture_fingerprint"]
    trace_manifest = {
        "schema": "molgap-trace-manifest-v1",
        "trajectory_id": TRAJECTORY_ID,
        "run_id": RUN_ID,
        "contract_ref": shared_rel("training_contract.json"),
        "model_identity": record["contract"]["architecture_fingerprint"],
        "reference_id": "pcqm-k1-v4-100k-reference-s42",
        "comparison_role": "candidate",
        "x_axis": "optimizer_steps",
        "presentation_semantics_ref": shared_rel("training_contract.json"),
        "weight_semantics": "raw_model_no_ema",
        "metric_semantics": "internal_development_direct_gap_mae_eV",
        "evaluation_role_identity": comparison_identity["evaluation_role_identity"],
        "selection_semantics": "best_internal_development_raw_model_over_40_epochs",
        "trace_artifact_ref": rel("results/canonical_trace.json"),
        "trace_artifact_sha256": file_digest(RESULTS / "canonical_trace.json"),
        "terminal_evidence_ref": rel("results/terminal_evidence.json"),
        "comparability_identity": {
            "scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-40epochs",
            "dataset_identity": comparison_identity["dataset_identity"],
            "row_split_identity": "train-0-100000-development-100000-150000@"
            "e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34",
            "architecture_identity": record["contract"]["architecture_fingerprint"],
            "optimizer_identity": comparison_identity["optimizer_identity"],
            "lr_schedule_identity": comparison_identity["schedule_identity"],
            "target_transform_identity": comparison_identity["target_transform_identity"],
            "precision_identity": "fp32-no-tf32-deterministic-bs128-drop-last",
            "ema_semantics": "none",
            "evaluation_role_identity": comparison_identity["evaluation_role_identity"],
            "selection_role_identity": comparison_identity["selection_role_identity"],
            "x_axis_semantics": "optimizer_steps",
            "terminal_endpoint_identity": "31240-optimizer-steps",
            "matched_architecture_required": False,
        },
        "exposure": {"optimizer_steps": 31240, "sample_presentations": 3998720},
        "checkpoint_identity": record["training"]["checkpoint_sha256"],
        "trace_fields": trace_fields,
        "backtest_eligibility": {"eligible": True, "exclusion_reasons": []},
    }
    write(RESULTS / "trace_manifest_terminal.json", trace_manifest)

    roles = role_events()
    write(
        RESULTS / "role_history.json",
        {
            "format": "molgap-role-history-index-v1",
            "trajectory_id": TRAJECTORY_ID,
            "protected_roles_read": False,
            "events": roles,
        },
    )
    elapsed = sum(float(row["seconds"]) for row in raw_trace["epochs"])
    cost = {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": f"cost-{TRAJECTORY_ID}-training",
        "trajectory_id": TRAJECTORY_ID,
        "action_id": ACTION_ID,
        "run_id": RUN_ID,
        "attempt_id": "candidate-v1",
        "category": "training",
        "platform": PLATFORM,
        "hardware": HARDWARE,
        "measurement": {
            "device_hours": {"value": elapsed / 3600.0, "status": "measured"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "wall_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "measurement_missing"},
        },
        "evidence_ref": rel("results/terminal_acceptance.json"),
    }
    write(RESULTS / "cost_records.json", {"format": "molgap-cost-records-v1", "costs": [cost]})

    candidate_paths = {
        "checkpoint": rel("results/checkpoint_manifest.json"),
        "runtime_certificate": rel("results/runtime_certificate.json"),
        "prediction_manifest": rel("results/prediction_manifest.json"),
        "row_manifest": rel("results/row_manifest.json"),
        "target_manifest": rel("results/target_manifest.json"),
        "trace_manifest": rel("results/trace_manifest_terminal.json"),
        "role_history": rel("results/role_history.json"),
        "target_transform_asset": f"{REFERENCE_REL}/target_transform.json",
        "cost_records": rel("results/cost_records.json"),
        "acceptance": rel("results/acceptance_summary.json"),
        "decision": rel("decision.md"),
        "source_config": shared_rel("source_config.json"),
        "paired_analysis": rel("results/paired_analysis.json"),
    }
    reference_bundle = load(REFERENCE_ROOT / "reference_bundle.json")
    reference_paths = {
        "checkpoint": f"{REFERENCE_REL}/checkpoint_manifest.json",
        "runtime_certificate": reference_bundle["runtime_certificate_ref"],
        "prediction_manifest": f"{REFERENCE_REL}/prediction_manifest.json",
        "row_manifest": reference_bundle["row_manifest_ref"],
        "target_manifest": reference_bundle["target_manifest_ref"],
        "trace_manifest": reference_bundle["trace_manifest_ref"],
        "role_history": reference_bundle["role_history_ref"],
        "target_transform_asset": reference_bundle["target_transform_asset_ref"],
        "cost_records": reference_bundle["cost_records_ref"],
        "acceptance": reference_bundle["acceptance_ref"],
        "decision": reference_bundle["decision_ref"],
        "source_config": f"{REFERENCE_REL}/source_config_binding.json",
    }
    if set(candidate_paths) != set(REQUIRED_CANDIDATE_OBSERVED_BINDINGS):
        raise RuntimeError("candidate binding coverage drift")
    if set(reference_paths) != set(REQUIRED_OBSERVED_BINDINGS):
        raise RuntimeError("reference binding coverage drift")

    # Write the compact summary before hashing it into strict readiness.
    summary = {
        "format": "molgap-v5-terminal-acceptance-summary-v1",
        "accepted": True,
        "model_inference_executed": False,
        "source_commit": acceptance["source_commit"],
        "source_archive_sha256": acceptance["source_archive_sha256"],
        "fixed_manifest_sha256": record["fixed_manifest_sha256"],
        "candidate": record["training"],
        "paired_analysis_ref": rel("results/paired_analysis.json"),
        "protected_roles_read": False,
        "artifact_sha256": completion["artifact_sha256"],
    }
    write(RESULTS / "acceptance_summary.json", summary)

    complete_reference = {name: "complete" for name in REQUIRED_OBSERVED_BINDINGS if name != "source_config"}
    complete_candidate = dict(complete_reference)
    complete_candidate["paired_analysis"] = "complete"
    plan = load((SHARED_ROOT or ROOT) / "role_plan.json")
    role_plan = {key: plan[key] for key in (
        "training_membership", "prediction_input", "labels_read", "metric_computed",
        "selection_used", "external_submission",
    )}
    observed = [row["access_kind"] for row in roles]
    candidate_side = {
        "comparison_identity": comparison_identity,
        "artifacts": complete_candidate,
        "artifact_bindings": {name: binding(path) for name, path in candidate_paths.items()},
        "prediction_status": "complete", "row_alignment_status": "aligned",
        "runtime_certificate_status": "accepted", "role_status": "complete",
        "trace_status": "complete", "stochasticity_status": "row_bootstrap_measured_training_floor_policy",
        "terminal_complete": True, "role_applicability_plan": role_plan,
        "observed_role_event_kinds": observed,
        "trace_field_availability": trace_fields,
    }
    reference_side = {
        "comparison_identity": reference_bundle["comparison_identity"],
        "artifacts": complete_reference,
        "artifact_bindings": {name: binding(path) for name, path in reference_paths.items()},
        "prediction_status": "complete", "row_alignment_status": "aligned",
        "runtime_certificate_status": "accepted", "role_status": "complete",
        "trace_status": "complete", "stochasticity_status": "training_stochasticity_unavailable",
        "terminal_complete": True, "role_applicability_plan": role_plan,
        "observed_role_event_kinds": observed,
        "trace_field_availability": trace_fields,
        "reference_bundle_id": reference_bundle["reference_bundle_id"],
        "reference_bundle_sha256": load(ROOT / "comparison_readiness_prelaunch.json")["reference_bundle_sha256"],
    }
    readiness = assess_comparison_readiness(
        candidate_id=MODE, candidate=candidate_side,
        reference_id=reference_bundle["reference_id"], reference=reference_side,
        declared_intervention_fields=["architecture_config_identity"],
        experiment_purpose="architecture_comparison", intervention_group_id="architecture",
    )
    validate_comparison_readiness(
        readiness,
        evidence_verifier=lambda pointer, digest: verify_bound_artifact(REPO_ROOT, pointer, digest),
    )
    if readiness["comparison_class"] != "STRICT_CAUSAL" or not readiness["strict_ready"]:
        raise RuntimeError(f"terminal comparison not strict-ready: {readiness}")
    write(RESULTS / "comparison_readiness.json", readiness)

    decision = {
        "decision_ref": rel("decision.md"),
        "outcome": DECISION_OUTCOME,
        "next_allowed_actions": NEXT_ALLOWED_ACTIONS,
        "reopen_conditions": REOPEN_CONDITIONS,
        "final": True,
    }
    role_use = {
        "official_train_prefix_0_100000": "used",
        "internal_development_100000_150000": "used",
        "official_validation": "untouched",
        "test_dev": "untouched",
        "test_challenge": "untouched",
    }
    outcome = {
        "execution_status": "complete", "artifact_status": "accepted",
        "comparison_status": "strict_causal", "scientific_status": SCIENTIFIC_STATUS,
        "transfer_status": "not_ready", "budget_decision": "stop_under_contract",
        "full_handoff_status": "not_authorized",
    }
    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "evidence_id": EVIDENCE_ID,
        "track": "C", "scope": "terminal_fixed_100k_architecture_screen",
        "contract": "MOLGAP-COMMON-V5-FINAL", "legacy_contract": "none-prospective-v5",
        "outcome": outcome, "role_use": role_use,
        "artifacts": [
            {"name": name, "locator": path, "sha256": file_digest(REPO_ROOT / path), "availability": "repository_retained"}
            for name, path in (
                ("acceptance_summary", rel("results/acceptance_summary.json")),
                ("paired_analysis", rel("results/paired_analysis.json")),
                ("comparison_readiness", rel("results/comparison_readiness.json")),
                ("canonical_trace", rel("results/canonical_trace.json")),
                *EXTRA_ARTIFACTS,
            )
        ],
        "authority": {"pointers": [
            shared_rel("protocol.md"), shared_rel("training_contract.json"), shared_rel("source_config.json"),
            rel("decision.md"), rel("trajectory.json"), rel("results/comparison_readiness.json"),
        ]},
        "migration": {
            "training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False, "migrated_at": MIGRATED_AT,
            "verification_scope": "no-training/no-inference terminal saved-artifact acceptance and paired analysis",
        },
    }
    write(RESULTS / "terminal_evidence.json", evidence)
    terminal_acceptance = {
        "format": "molgap-rml-terminal-acceptance-adapter-v1",
        "evidence_id": evidence["evidence_id"], "run_id": RUN_ID,
        "outcome": outcome, "trajectory_decision": decision, "role_use": role_use,
        "costs": [cost], "roles": roles,
        "source_acceptance_ref": rel(SOURCE_ACCEPTANCE_REL),
        "source_acceptance_sha256": file_digest(RAW_ACCEPTANCE),
    }
    write(RESULTS / "terminal_acceptance.json", terminal_acceptance)

    bound_paths = {rel("results/terminal_acceptance.json")}
    bound_paths.update(artifact["locator"] for artifact in evidence["artifacts"])
    bound_paths.update(evidence["authority"]["pointers"])
    bound_paths.add(rel("decision.md"))
    bound_paths.add(rel("results/comparison_readiness.json"))
    terminal = {
        "format": "molgap-rml-terminal-package-v1",
        "trajectory_id": TRAJECTORY_ID, "run_id": RUN_ID, "action_id": ACTION_ID,
        "finalized_at": FINALIZED_AT,
        "acceptance_ref": rel("results/terminal_acceptance.json"),
        "artifact_hashes": {path: file_digest(REPO_ROOT / path) for path in sorted(bound_paths)},
        "evidence": evidence, "decision": decision, "costs": [cost], "roles": roles,
        "role_use": role_use, "trace_manifest": trace_manifest,
        "comparison_readiness_ref": rel("results/comparison_readiness.json"),
    }
    write(RESULTS / "terminal.json", terminal)


if __name__ == "__main__":
    main()
