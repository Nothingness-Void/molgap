"""Retain terminal V5/RML evidence without upgrading a confounded comparison."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import canonicalize_trace
from molgap.training_reproducibility import atomic_json, sha256_file


ROOT_REL = "experiments/pcqm_k1_pair_token_mose_100k"
ROOT = REPO_ROOT / ROOT_REL
RECORD = REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_pair_token_mose_s42_v1/pcqm_k1_pair_token_mose/neural_atom_k1_pair_token_mose"
TRAJECTORY_ID = "TC-k1-pair-token-mose-100k-s42"
RUN_ID = "kaseichou/molgap-k1-pairtoken-mose-s42:v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(name: str) -> str:
    return f"{ROOT_REL}/{name}"


def main() -> None:
    endpoint = load(ROOT / "contextual_endpoint.json")
    if (
        endpoint["mechanical_artifacts_accepted"] is not True
        or endpoint["comparison_class"] != "PAIRED_ENDPOINT"
        or endpoint["strict_causal_claim_allowed"] is not False
        or endpoint["model_inference_executed"] is not False
    ):
        raise RuntimeError("Contextual-only endpoint evidence is required")
    record = load(RECORD / "arm_record.json")
    raw = load(RECORD / "trace.json")
    if record["training"]["optimizer_steps"] != 31240 or len(raw["epochs"]) != 40:
        raise RuntimeError("Expected terminal training trace is absent")
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
    elapsed = 0.0
    observations = []
    for index, epoch in enumerate(raw["epochs"]):
        seconds = float(epoch["seconds"])
        elapsed += seconds
        observations.append(
            {
                "sequence": index,
                "event": "terminal" if index == 39 else "observation",
                "optimizer_step": int(epoch["optimizer_steps"]),
                "sample_presentations": int(epoch["sample_presentations"]),
                "epoch_or_pass": float(epoch["epoch"]),
                "learning_rate": float(epoch["learning_rate"]),
                "live_train_metric": float(epoch["train_normalized_mae"]),
                "live_dev_metric": float(epoch["development_gap_mae_eV"]),
                "ema_dev_metric": None,
                "wall_time_seconds": seconds,
                "cumulative_wall_time_seconds": elapsed,
                "device_time_seconds": seconds,
                "cumulative_device_time_seconds": elapsed,
                "checkpoint_identity": endpoint["checkpoint_sha256"] if index == 39 else None,
            }
        )
    semantics = {
        "live_train_metric": {
            "metric": "mean_absolute_error", "unit": "normalized_target_units",
            "target": "gap", "role_identity": "official_train_prefix_0_100000",
            "weights": "live", "direction": "minimize",
        },
        "live_dev_metric": {
            "metric": "mean_absolute_error", "unit": "eV", "target": "gap",
            "role_identity": "internal_development_100000_150000",
            "weights": "live", "direction": "minimize",
        },
        "ema_dev_metric": None,
    }
    trace = canonicalize_trace(
        {
            "trajectory_id": TRAJECTORY_ID,
            "run_id": RUN_ID,
            "device_time_semantics": "sum_over_devices",
            "metric_semantics": semantics,
            "observations": observations,
            "provenance": {
                "source_refs": [
                    "external://local-platform-record/kaggle/training/"
                    "pcqm_k1_pair_token_mose_s42_v1/pcqm_k1_pair_token_mose/"
                    "neural_atom_k1_pair_token_mose/trace.json"
                ],
                "recovery": "deterministic-field-map-from-retained-remote-trace",
            },
        }
    )
    atomic_json(ROOT / "canonical_trace.json", trace)

    prelaunch = load(ROOT / "comparison_readiness_prelaunch.json")
    fields = prelaunch["matched_fields"]
    identity = {
        "scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-rwse16-mose31",
        "dataset_identity": fields["dataset_identity"],
        "row_split_identity": "train-0-100000-development-100000-150000@"
        "e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34",
        "architecture_identity": record["contract"]["architecture_fingerprint"],
        "optimizer_identity": fields["optimizer_identity"],
        "lr_schedule_identity": fields["schedule_identity"],
        "target_transform_identity": fields["target_transform_identity"],
        "precision_identity": "fp32-no-tf32-deterministic-bs128-drop-last",
        "ema_semantics": "none",
        "evaluation_role_identity": fields["evaluation_role_identity"],
        "selection_role_identity": fields["selection_role_identity"],
        "x_axis_semantics": "optimizer_steps",
        "terminal_endpoint_identity": "31240-optimizer-steps",
        "matched_architecture_required": False,
    }
    manifest = {
        "schema": "molgap-trace-manifest-v1",
        "trajectory_id": TRAJECTORY_ID,
        "run_id": RUN_ID,
        "contract_ref": rel("training_contract.json"),
        "model_identity": record["contract"]["architecture_fingerprint"],
        "reference_id": "pcqm-k1-v4-100k-reference-s42",
        "comparison_role": "candidate",
        "x_axis": "optimizer_steps",
        "presentation_semantics_ref": rel("training_contract.json"),
        "weight_semantics": "raw_model_no_ema",
        "metric_semantics": "internal_development_direct_gap_mae_eV",
        "evaluation_role_identity": fields["evaluation_role_identity"],
        "selection_semantics": "best_internal_development_raw_model_over_40_epochs",
        "trace_artifact_ref": rel("canonical_trace.json"),
        "trace_artifact_sha256": sha256_file(ROOT / "canonical_trace.json"),
        "terminal_evidence_ref": rel("contextual_endpoint.json"),
        "comparability_identity": identity,
        "exposure": {"optimizer_steps": 31240, "sample_presentations": 3998720},
        "checkpoint_identity": endpoint["checkpoint_sha256"],
        "trace_fields": trace_fields,
        "backtest_eligibility": {
            "eligible": False,
            "exclusion_reasons": ["undeclared_feature_identity_mismatch"],
        },
    }
    atomic_json(ROOT / "trace_manifest.json", manifest)

    role_specs = (
        ("training-membership", "official_train_prefix_0_100000", "training_membership", False),
        ("prediction-input", "internal_development_100000_150000", "prediction_input", False),
        ("labels-read", "internal_development_100000_150000", "labels_read", False),
        ("metric-computed", "internal_development_100000_150000", "metric_computed", False),
        ("selection-used", "internal_development_100000_150000", "selection_used", True),
    )
    for suffix, role, access, selected in role_specs:
        event = {
            "schema": "molgap-role-event-v1",
            "role_event_id": f"role-{TRAJECTORY_ID}-{suffix}",
            "trajectory_id": TRAJECTORY_ID,
            "action_id": "A001",
            "run_id": RUN_ID,
            "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1",
            "row_manifest_hash": record["fixed_manifest_sha256"],
            "role_name": role,
            "access_kind": access,
            "selection_used": selected,
            "evidence_ref": rel("contextual_endpoint.json"),
        }
        atomic_json(ROOT / "roles" / f"{event['role_event_id']}.json", event)
    cost_path = ROOT / "costs/expected_training.json"
    cost = load(cost_path)
    cost["run_id"] = RUN_ID
    cost["attempt_id"] = "candidate-v1"
    cost["measurement"]["device_hours"] = {"value": elapsed / 3600.0, "status": "measured"}
    cost["measurement"]["wall_hours"] = {"value": None, "status": "measurement_missing"}
    cost["evidence_ref"] = rel("contextual_endpoint.json")
    atomic_json(cost_path, cost)

    trajectory_path = ROOT / "trajectory.json"
    trajectory = load(trajectory_path)
    action = trajectory["actions"][0]
    action["run_ids"] = [RUN_ID]
    action["attempt_ids"] = ["candidate-v1"]
    action["evidence_refs"] = list(dict.fromkeys(action["evidence_refs"] + [rel("contextual_endpoint.json"), rel("trace_manifest.json")]))
    trajectory["result"]["evidence_refs"] = list(dict.fromkeys(trajectory["result"]["evidence_refs"] + [rel("contextual_endpoint.json")]))
    trajectory["decision"] = {
        "decision_ref": rel("decision.md"),
        "outcome": "CLOSED",
        "next_allowed_actions": [],
        "reopen_conditions": ["new user authority and prospectively valid feature-intervention contract"],
        "final": True,
    }
    trajectory["comparison_class"] = "PAIRED_ENDPOINT"
    trajectory["comparison_blockers"] = ["SCIENTIFIC_CONTRACT_MISMATCH"]
    # Retain the frozen prelaunch claim as historical evidence; the terminal
    # endpoint and decision explicitly invalidate its strict comparison plan.
    trajectory["comparison_readiness_ref"] = rel("comparison_readiness_prelaunch.json")
    atomic_json(trajectory_path, trajectory)

    evidence_path = ROOT / "v5_evidence.json"
    evidence = load(evidence_path)
    evidence["scope"] = "terminal_fixed_100k_contextual_endpoint"
    evidence["outcome"] = {
        "execution_status": "complete",
        "artifact_status": "accepted",
        "comparison_status": "paired_endpoint_not_strict",
        "scientific_status": "not_promoted_contract_mismatch",
        "transfer_status": "not_ready",
        "budget_decision": "stop_under_contract",
        "full_handoff_status": "not_authorized",
    }
    evidence["role_use"] = {
        "official_train_prefix_0_100000": "used",
        "internal_development_100000_150000": "used",
        "official_validation": "untouched",
        "test_dev": "untouched",
        "test_challenge": "untouched",
    }
    evidence["authority"]["pointers"] = list(dict.fromkeys(
        evidence["authority"]["pointers"] +
        [rel("decision.md"), rel("contextual_endpoint.json"), rel("trace_manifest.json")]
    ))
    for name in ("decision.md", "contextual_endpoint.json", "canonical_trace.json", "trace_manifest.json"):
        artifact = {
                "name": name.removesuffix(".json").removesuffix(".md"),
                "locator": f"repo://{rel(name)}",
                "sha256": sha256_file(ROOT / name),
                "availability": "repository_retained",
            }
        evidence["artifacts"] = [
            existing for existing in evidence["artifacts"]
            if existing["name"] != artifact["name"]
        ] + [artifact]
    evidence["migration"]["verification_scope"] = (
        "terminal saved-artifact hashes, aligned development predictions, "
        "observed trace/cost/roles; no local model inference"
    )
    # These flags describe the local migration operation, not the remote run.
    evidence["migration"]["training_executed"] = False
    atomic_json(evidence_path, evidence)


if __name__ == "__main__":
    main()
