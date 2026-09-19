"""Build complete no-inference V5/RML terminal evidence for Kaggle3 v2."""
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


ROOT = REPO_ROOT / "experiments/pcqm_k1_pair_token_node_return_100k"
ROOT_REL = "experiments/pcqm_k1_pair_token_node_return_100k"
REFERENCE_ROOT = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference"
REFERENCE_REL = "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference"
RECORD_ROOT = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_pair_token_node_return_100k_v2"
CANDIDATE_ROOT = RECORD_ROOT / "pcqm_k1_node_adaptive_pairtoken/neural_atom_k1_pair_token_node_return"
ACCEPTANCE_PATH = RECORD_ROOT / "acceptance_v2.json"
TRAJECTORY_ID = "TC-k1-pair-token-node-return-100k-s42"
RUN_ID = "nvoid912/molgap-k1-node-adaptive-pairtoken-s42:v2"
CONTRACT_RUN_ID = "pcqm-k1-variants-100k-s42-v1-neural_atom_k1_pair_token_node_return"
DATASET_ID = "pcqm4mv2-ogb-fixed-100k-v1"
MANIFEST_SHA = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rel(path: str) -> str:
    return f"{ROOT_REL}/{path}"


def binding(path: str) -> dict[str, str]:
    absolute = REPO_ROOT / path
    return {"ref": path, "sha256": sha256_file(absolute)}


def verify_completion() -> dict[str, Any]:
    completion = load_json(CANDIDATE_ROOT / "completion_manifest.json")
    if completion.get("complete") is not True:
        raise RuntimeError("Candidate completion manifest is not terminal")
    for name, expected in completion["artifact_sha256"].items():
        actual = sha256_file(CANDIDATE_ROOT / name)
        if actual != expected:
            raise RuntimeError(f"Terminal artifact hash mismatch: {name}")
    return completion


def write_reference_binding_manifests() -> None:
    acceptance = load_json(REFERENCE_ROOT / "reference_acceptance.json")
    bundle = load_json(REFERENCE_ROOT / "reference_bundle.json")
    atomic_json(
        REFERENCE_ROOT / "checkpoint_manifest.json",
        {
            "format": "molgap-checkpoint-manifest-v1",
            "reference_id": bundle["reference_id"],
            "checkpoint_identity": bundle["checkpoint_identity"],
            "verified_external_artifacts": acceptance["verified_external_artifacts"],
        },
    )
    atomic_json(
        REFERENCE_ROOT / "source_config_binding.json",
        {
            "format": "molgap-reference-source-binding-v1",
            "reference_id": bundle["reference_id"],
            "source_commit_or_archive": bundle["source_commit_or_archive"],
            "architecture_config_identity": bundle["architecture_config_identity"],
            "contract_ref": bundle["contract_ref"],
        },
    )
    reference_event_specs = (
        ("prediction-input", "prediction_input"),
        ("labels-read", "labels_read"),
        ("metric-computed", "metric_computed"),
    )
    for suffix, access_kind in reference_event_specs:
        atomic_json(
            REFERENCE_ROOT / f"roles/{suffix}.json",
            {
                "schema": "molgap-role-event-v1",
                "role_event_id": f"role-TC-k1-v4-100k-reference-s42-{suffix}",
                "trajectory_id": "TC-k1-v4-100k-reference-s42",
                "action_id": "A001",
                "run_id": "kaseichou/molgap-pcqm-k1-v4-reference-s42:v2",
                "dataset_identity": DATASET_ID,
                "row_manifest_hash": MANIFEST_SHA,
                "role_name": "internal_development_100000_150000",
                "access_kind": access_kind,
                "selection_used": False,
                "evidence_ref": f"{REFERENCE_REL}/v5_evidence.json",
            },
        )
    reference_role_refs = [
        f"{REFERENCE_REL}/roles/train.json",
        f"{REFERENCE_REL}/roles/prediction-input.json",
        f"{REFERENCE_REL}/roles/labels-read.json",
        f"{REFERENCE_REL}/roles/metric-computed.json",
        f"{REFERENCE_REL}/roles/development.json",
    ]
    atomic_json(
        REFERENCE_ROOT / "role_history.json",
        {
            "format": "molgap-role-history-index-v1",
            "trajectory_id": "TC-k1-v4-100k-reference-s42",
            "protected_roles_read": False,
            "events": [
                {"ref": ref, "sha256": sha256_file(REPO_ROOT / ref)}
                for ref in reference_role_refs
            ],
        },
    )
    reference_acceptance = load_json(REFERENCE_ROOT / "reference_acceptance.json")
    reference_acceptance["compact_manifest_sha256"]["role_history.json"] = sha256_file(
        REFERENCE_ROOT / "role_history.json"
    )
    atomic_json(REFERENCE_ROOT / "reference_acceptance.json", reference_acceptance)


def write_role_evidence() -> list[str]:
    role_plan = load_json(ROOT / "role_plan.json")
    event_specs = (
        ("training-membership", "official_train_prefix_0_100000", "training_membership", False),
        ("prediction-input", "internal_development_100000_150000", "prediction_input", False),
        ("labels-read", "internal_development_100000_150000", "labels_read", False),
        ("metric-computed", "internal_development_100000_150000", "metric_computed", False),
        ("selection-used", "internal_development_100000_150000", "selection_used", True),
    )
    refs = []
    for suffix, role_name, access_kind, selection_used in event_specs:
        path = ROOT / f"roles/{suffix}.json"
        atomic_json(
            path,
            {
                "schema": "molgap-role-event-v1",
                "role_event_id": f"role-{TRAJECTORY_ID}-{suffix}",
                "trajectory_id": TRAJECTORY_ID,
                "action_id": "A002",
                "run_id": RUN_ID,
                "dataset_identity": DATASET_ID,
                "row_manifest_hash": MANIFEST_SHA,
                "role_name": role_name,
                "access_kind": access_kind,
                "selection_used": selection_used,
                "evidence_ref": rel("v5_evidence.json"),
            },
        )
        refs.append(rel(f"roles/{suffix}.json"))
    atomic_json(
        ROOT / "role_history.json",
        {
            "format": "molgap-role-history-index-v1",
            "trajectory_id": TRAJECTORY_ID,
            "protected_roles_read": False,
            "role_applicability_plan": {
                key: role_plan[key]
                for key in (
                    "training_membership",
                    "prediction_input",
                    "labels_read",
                    "metric_computed",
                    "selection_used",
                    "external_submission",
                )
            },
            "events": [
                {"ref": ref, "sha256": sha256_file(REPO_ROOT / ref)} for ref in refs
            ],
        },
    )
    return [spec[2] for spec in event_specs]


def write_compact_terminal_artifacts(
    acceptance: dict[str, Any], completion: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    candidate = acceptance["candidates"]["neural_atom_k1_pair_token_node_return"]
    record = candidate["record"]
    payload_path = CANDIDATE_ROOT / "best_development_payload.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    prediction_sha = tensor_sha256(payload["prediction_eV"])
    target_sha = tensor_sha256(payload["target_eV"])
    source_idx_sha = tensor_sha256(payload["source_idx"])
    reference_prediction = load_json(REFERENCE_ROOT / "prediction_manifest.json")
    if target_sha != reference_prediction["target_sha256"]:
        raise RuntimeError("Candidate target bytes differ from reusable reference")
    if source_idx_sha != reference_prediction["source_idx_sha256"]:
        raise RuntimeError("Candidate row order differs from reusable reference")

    atomic_json(
        ROOT / "prediction_manifest.json",
        {
            "format": "molgap-aligned-prediction-manifest-v1",
            "artifact_locator": (
                "external://local-platform-record/kaggle/"
                "pcqm_k1_pair_token_node_return_100k_v2/"
                "pcqm_k1_node_adaptive_pairtoken/neural_atom_k1_pair_token_node_return/"
                "best_development_payload.pt"
            ),
            "artifact_sha256": sha256_file(payload_path),
            "development_gap_mae_eV": record["training"]["development_gap_mae_eV"],
            "evaluation_role_identity": (
                "pcqm4mv2-ogb-fixed-100k-v1:internal-development-100000-150000"
            ),
            "ordering_semantics": "source_idx ascending",
            "prediction_sha256": prediction_sha,
            "target_sha256": target_sha,
            "source_idx_sha256": source_idx_sha,
            "row_count": int(payload["source_idx"].numel()),
            "unique_source_idx": int(torch.unique(payload["source_idx"]).numel()),
            "tensor_hash_semantics": "sha256-contiguous-c-order-bytes",
        },
    )
    atomic_json(ROOT / "row_manifest.json", load_json(REFERENCE_ROOT / "row_manifest.json"))
    atomic_json(ROOT / "target_manifest.json", load_json(REFERENCE_ROOT / "target_manifest.json"))
    atomic_json(ROOT / "runtime_certificate.json", load_json(CANDIDATE_ROOT / "runtime_certificate.json"))
    atomic_json(
        ROOT / "checkpoint_manifest.json",
        {
            "format": "molgap-checkpoint-manifest-v1",
            "run_id": RUN_ID,
            "best_epoch": record["training"]["best_epoch"],
            "best_model_sha256": record["training"]["best_model_sha256"],
            "last_checkpoint_sha256": record["training"]["checkpoint_sha256"],
            "recovery_artifacts": {
                name: digest
                for name, digest in completion["artifact_sha256"].items()
                if name.startswith("recovery_epoch_")
            },
            "external_root": (
                "external://local-platform-record/kaggle/"
                "pcqm_k1_pair_token_node_return_100k_v2/"
                "pcqm_k1_node_adaptive_pairtoken/neural_atom_k1_pair_token_node_return"
            ),
        },
    )
    paired = {
        "format": "molgap-paired-analysis-v1",
        "candidate_id": "neural_atom_k1_pair_token_node_return",
        "reference_id": "pcqm-k1-v4-100k-reference-s42",
        "rows": 50000,
        "candidate_development_gap_mae_eV": record["training"]["development_gap_mae_eV"],
        "reference_development_gap_mae_eV": acceptance["reference_development_gap_mae_eV"],
        "gain_eV": candidate["gate"]["gain_eV"],
        "paired_error_delta_bootstrap_95_eV": candidate["paired_error_delta_bootstrap_95_eV"],
        "required_gain_eV": candidate["gate"]["required_gain_eV"],
        "row_bootstrap_direction_favorable": candidate["paired_error_delta_bootstrap_95_eV"][1] < 0,
        "material_gate_passed": candidate["gate"]["passed"],
        "training_stochasticity_accounted": candidate["gate"]["training_stochasticity_accounted"],
        "row_bootstrap_is_sufficient_alone": candidate["gate"]["row_bootstrap_is_sufficient_alone"],
    }
    atomic_json(ROOT / "paired_analysis.json", paired)
    actual_seconds = sum(float(item["seconds"]) for item in load_json(CANDIDATE_ROOT / "trace.json")["epochs"])
    atomic_json(
        ROOT / "costs/expected_training.json",
        {
            "schema": "molgap-cost-event-v1",
            "cost_event_id": "cost-TC-k1-pair-token-node-return-100k-s42-training",
            "trajectory_id": TRAJECTORY_ID,
            "action_id": "A002",
            "run_id": RUN_ID,
            "attempt_id": "candidate-v2",
            "category": "training",
            "platform": "kaggle3",
            "hardware": "Tesla_T4_16GB",
            "measurement": {
                "device_hours": {"value": actual_seconds / 3600.0, "status": "measured"},
                "cpu_hours": {"value": None, "status": "measurement_missing"},
                "wall_hours": {"value": None, "status": "measurement_missing"},
                "queue_hours": {"value": None, "status": "measurement_missing"},
            },
            "evidence_ref": rel("v5_evidence.json"),
        },
    )
    atomic_json(
        ROOT / "acceptance_summary_v2.json",
        {
            "format": "molgap-v5-terminal-acceptance-summary-v1",
            "accepted": acceptance["accepted"],
            "model_inference_executed": acceptance["model_inference_executed"],
            "source_commit": acceptance["source_commit"],
            "source_archive_sha256": acceptance["source_archive_sha256"],
            "fixed_manifest_sha256": record["fixed_manifest_sha256"],
            "candidate": {
                "parameter_count": record["training"]["parameter_count"],
                "best_epoch": record["training"]["best_epoch"],
                "development_gap_mae_eV": record["training"]["development_gap_mae_eV"],
                "optimizer_steps": record["training"]["optimizer_steps"],
                "sample_presentations": record["training"]["sample_presentations"],
                "mean_graphs_per_second": record["training"]["mean_graphs_per_second"],
                "mean_epoch_seconds": record["training"]["mean_epoch_seconds"],
                "peak_allocated_mib": record["training"]["peak_allocated_mib"],
                "peak_reserved_mib": record["training"]["peak_reserved_mib"],
            },
            "paired_analysis_ref": rel("paired_analysis.json"),
            "raw_acceptance_locator": (
                "external://local-platform-record/kaggle/"
                "pcqm_k1_pair_token_node_return_100k_v2/acceptance_v2.json"
            ),
            "raw_acceptance_sha256": sha256_file(ACCEPTANCE_PATH),
            "protected_roles_read": False,
            "artifact_sha256": completion["artifact_sha256"],
        },
    )
    trace_fields = {
        "optimizer_step": True,
        "sample_presentations": True,
        "epoch_or_pass": True,
        "learning_rate": True,
        "live_train_metric": True,
        "live_dev_metric": True,
        "ema_dev_metric": False,
        "checkpoint_identity": True,
    }
    comparison_identity = dict(load_json(ROOT / "comparison_readiness_prelaunch.json")["matched_fields"])
    comparison_identity["architecture_config_identity"] = record["contract"]["architecture_fingerprint"]
    atomic_json(
        ROOT / "trace_manifest.json",
        {
            "schema": "molgap-trace-manifest-v1",
            "trajectory_id": TRAJECTORY_ID,
            "run_id": CONTRACT_RUN_ID,
            "contract_ref": rel("training_contract.json"),
            "model_identity": record["contract"]["architecture_fingerprint"],
            "reference_id": "pcqm-k1-v4-100k-reference-s42",
            "comparison_role": "candidate",
            "x_axis": "optimizer_steps",
            "presentation_semantics_ref": rel("training_contract.json"),
            "weight_semantics": "raw_model_no_ema",
            "metric_semantics": "internal_development_direct_gap_mae_eV",
            "evaluation_role_identity": comparison_identity["evaluation_role_identity"],
            "selection_semantics": "best_internal_development_raw_model_over_40_epochs",
            "trace_artifact_ref": (
                "external://local-platform-record/kaggle/"
                "pcqm_k1_pair_token_node_return_100k_v2/"
                "pcqm_k1_node_adaptive_pairtoken/neural_atom_k1_pair_token_node_return/trace.json"
            ),
            "terminal_evidence_ref": rel("v5_evidence.json"),
            "comparability_identity": {
                "scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-40epochs",
                "dataset_identity": comparison_identity["dataset_identity"],
                "row_split_identity": "train-0-100000-development-100000-150000@e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34",
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
            "exposure": {
                "optimizer_steps": record["training"]["optimizer_steps"],
                "sample_presentations": record["training"]["sample_presentations"],
            },
            "checkpoint_identity": record["training"]["checkpoint_sha256"],
            "trace_fields": trace_fields,
            "backtest_eligibility": {"eligible": True, "exclusion_reasons": []},
        },
    )
    return comparison_identity, list(trace_fields)


def make_bindings() -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_paths = {
        "checkpoint": rel("checkpoint_manifest.json"),
        "runtime_certificate": rel("runtime_certificate.json"),
        "prediction_manifest": rel("prediction_manifest.json"),
        "row_manifest": rel("row_manifest.json"),
        "target_manifest": rel("target_manifest.json"),
        "trace_manifest": rel("trace_manifest.json"),
        "role_history": rel("role_history.json"),
        "target_transform_asset": f"{REFERENCE_REL}/target_transform.json",
        "cost_records": rel("costs/expected_training.json"),
        "acceptance": rel("acceptance_summary_v2.json"),
        "decision": rel("decision.md"),
        "source_config": rel("source_config.json"),
        "paired_analysis": rel("paired_analysis.json"),
    }
    reference_bundle = load_json(REFERENCE_ROOT / "reference_bundle.json")
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
        raise RuntimeError("Candidate binding coverage drift")
    if set(reference_paths) != set(REQUIRED_OBSERVED_BINDINGS):
        raise RuntimeError("Reference binding coverage drift")
    return (
        {name: binding(path) for name, path in candidate_paths.items()},
        {name: binding(path) for name, path in reference_paths.items()},
    )


def main() -> None:
    completion = verify_completion()
    acceptance = load_json(ACCEPTANCE_PATH)
    if acceptance.get("accepted") is not True or acceptance.get("model_inference_executed") is not False:
        raise RuntimeError("No-inference mechanical acceptance is required")
    write_reference_binding_manifests()
    observed_roles = write_role_evidence()
    candidate_identity, trace_fields = write_compact_terminal_artifacts(acceptance, completion)
    candidate_bindings, reference_bindings = make_bindings()
    reference_bundle = load_json(REFERENCE_ROOT / "reference_bundle.json")
    role_plan = {
        key: load_json(ROOT / "role_plan.json")[key]
        for key in (
            "training_membership",
            "prediction_input",
            "labels_read",
            "metric_computed",
            "selection_used",
            "external_submission",
        )
    }
    reference_trace = {
        "optimizer_step": True,
        "sample_presentations": True,
        "epoch_or_pass": True,
        "learning_rate": True,
        "live_train_metric": True,
        "live_dev_metric": True,
        "ema_dev_metric": False,
        "checkpoint_identity": True,
    }
    complete_reference_artifacts = {
        name: "complete" for name in REQUIRED_OBSERVED_BINDINGS if name != "source_config"
    }
    complete_candidate_artifacts = dict(complete_reference_artifacts)
    complete_candidate_artifacts["paired_analysis"] = "complete"
    candidate_side = {
        "comparison_identity": candidate_identity,
        "artifacts": complete_candidate_artifacts,
        "artifact_bindings": candidate_bindings,
        "prediction_status": "complete",
        "row_alignment_status": "aligned",
        "runtime_certificate_status": "accepted",
        "role_status": "complete",
        "trace_status": "complete",
        "stochasticity_status": "row_bootstrap_measured_training_floor_policy",
        "terminal_complete": True,
        "role_applicability_plan": role_plan,
        "observed_role_event_kinds": observed_roles,
        "trace_field_availability": {name: name in trace_fields and name != "ema_dev_metric" for name in trace_fields},
    }
    reference_side = {
        "comparison_identity": reference_bundle["comparison_identity"],
        "artifacts": complete_reference_artifacts,
        "artifact_bindings": reference_bindings,
        "prediction_status": "complete",
        "row_alignment_status": "aligned",
        "runtime_certificate_status": "accepted",
        "role_status": "complete",
        "trace_status": "complete",
        "stochasticity_status": "training_stochasticity_unavailable",
        "terminal_complete": True,
        "role_applicability_plan": role_plan,
        "observed_role_event_kinds": observed_roles,
        "trace_field_availability": reference_trace,
        "reference_bundle_id": reference_bundle["reference_bundle_id"],
        "reference_bundle_sha256": load_json(ROOT / "comparison_readiness_prelaunch.json")["reference_bundle_sha256"],
    }
    readiness = assess_comparison_readiness(
        candidate_id="neural_atom_k1_pair_token_node_return",
        candidate=candidate_side,
        reference_id=reference_bundle["reference_id"],
        reference=reference_side,
        declared_intervention_fields=["architecture_config_identity"],
        experiment_purpose="architecture_comparison",
        intervention_group_id="architecture",
    )
    validate_comparison_readiness(
        readiness,
        evidence_verifier=lambda pointer, digest: verify_bound_artifact(
            REPO_ROOT, pointer, digest
        ),
    )
    if readiness["comparison_class"] != "STRICT_CAUSAL" or not readiness["strict_ready"]:
        raise RuntimeError(f"Terminal comparison is not strict-ready: {readiness}")
    atomic_json(ROOT / "comparison_readiness.json", readiness)


if __name__ == "__main__":
    main()
