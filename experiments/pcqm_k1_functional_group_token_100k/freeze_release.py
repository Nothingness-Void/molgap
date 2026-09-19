"""Freeze prospective RML and the fail-closed GPU release gate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.constants import REPO_ROOT
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS
from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import write_server_comparison_prelaunch


ROOT = REPO_ROOT / "experiments/pcqm_k1_functional_group_token_100k"
ROOT_REL = "experiments/pcqm_k1_functional_group_token_100k"
REFERENCE_REL = (
    "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/"
    "reference_bundle.json"
)
MODE = "neural_atom_k1_functional_group_token"
TRAJECTORY_ID = "TC-k1-functional-group-token-100k-s42"
EVIDENCE_ID = "pcqm-k1-functional-group-token-100k-s42"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(name: str) -> str:
    return f"{ROOT_REL}/{name}"


def main() -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", ROOT_REL],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Source/config must be committed before release freezing")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    reference_path = REPO_ROOT / REFERENCE_REL
    reference = load_json(reference_path)
    sidecar = load_json(ROOT / "sidecar_acceptance_summary.json")
    contract = load_json(ROOT / "training_contract.json")
    if not sidecar.get("accepted") or contract.get(
        "functional_group_sidecar_aggregate_sha256"
    ) != sidecar.get("aggregate_sha256"):
        raise RuntimeError("Accepted sidecar is not bound into the training contract")

    architecture_identity = canonical_fingerprint(ARCHITECTURE_CONFIGS[MODE])
    source_config = {
        "format": "molgap-frozen-source-config-v1",
        "candidate_id": MODE,
        "source_commit": source_commit,
        "architecture_config_identity": architecture_identity,
        "training_contract_ref": rel("training_contract.json"),
        "training_contract_sha256": sha256_file(ROOT / "training_contract.json"),
        "implementation_refs": [
            "src/molgap/k1_functional_group_token.py",
            "src/molgap/pcqm_functional_group_sidecar.py",
            "src/molgap/pcqm_k1_variants.py",
            "src/molgap/pcqm_k1_variants_runner.py",
        ],
        "sidecar_acceptance_ref": rel("sidecar_acceptance_summary.json"),
        "sidecar_aggregate_sha256": sidecar["aggregate_sha256"],
        "single_changed_mechanism": "layer6-sparse-atom-functional-group-token-exchange",
    }
    atomic_json(ROOT / "source_config.json", source_config)

    candidate_identity = dict(reference["comparison_identity"])
    candidate_identity["architecture_config_identity"] = architecture_identity
    role_file = load_json(ROOT / "role_plan.json")
    role_plan = {
        key: role_file[key]
        for key in (
            "training_membership",
            "prediction_input",
            "labels_read",
            "metric_computed",
            "selection_used",
            "external_submission",
        )
    }
    trace_file = load_json(ROOT / "trace_plan.json")
    trace_plan = {
        key: trace_file[key]
        for key in (
            "optimizer_step",
            "sample_presentations",
            "epoch_or_pass",
            "learning_rate",
            "live_train_metric",
            "live_dev_metric",
            "ema_dev_metric",
            "checkpoint_identity",
        )
    }
    prelaunch = assess_comparison_prelaunch(
        candidate_id=MODE,
        candidate_plan={
            "comparison_identity": candidate_identity,
            "source_config_status": "frozen",
            "source_commit_or_archive": source_commit,
        },
        reference_id=reference["reference_id"],
        reference_bundle=reference,
        experiment_purpose="architecture_comparison",
        intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
        role_applicability_plan=role_plan,
        trace_plan=trace_plan,
        runtime_qualification_plan={
            "status": "declared",
            "runtime_certificate_required": True,
            "qualification_scope": "single-device-fp32-no-tf32-bs128-optimizer-inclusive",
        },
    )
    write_server_comparison_prelaunch(
        ROOT / "comparison_readiness_prelaunch.json",
        comparison_prelaunch=prelaunch,
        experiment_purpose="architecture_comparison",
        reference_bundle=reference,
        repo_root=REPO_ROOT,
        reference_bundle_path=reference_path,
    )

    budget = {
        "format": "molgap-budget-snapshot-v1",
        "trajectory_id": TRAJECTORY_ID,
        "available_pool": "Kaggle2-and-Kaggle3-shared-weekly-budget",
        "available_device_hours_user_reported": 60.0,
        "reserved_for_this_action_device_hours": 3.0,
        "kaggle1_reserved_for_desktop": True,
        "successor_compute_pre_authorized": False,
    }
    atomic_json(ROOT / "budget_snapshot.json", budget)
    cost = {
        "schema": "molgap-cost-event-v1",
        "cost_event_id": "cost-TC-k1-functional-group-token-100k-s42-training",
        "trajectory_id": TRAJECTORY_ID,
        "action_id": "A002",
        "run_id": "kaseichou/molgap-k1-functional-group-token-s42:v1-planned",
        "attempt_id": "candidate-v1-planned",
        "category": "training",
        "platform": "kaggle2",
        "hardware": "Tesla_T4_16GB",
        "measurement": {
            "device_hours": {"value": 3.0, "status": "estimated"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "wall_hours": {"value": 3.0, "status": "estimated"},
            "queue_hours": {"value": None, "status": "measurement_missing"},
        },
        "evidence_ref": rel("v5_evidence.json"),
    }
    (ROOT / "costs").mkdir(exist_ok=True)
    atomic_json(ROOT / "costs/expected_training.json", cost)
    trajectory = {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": TRAJECTORY_ID,
        "record_mode": "prospective",
        "track": "C",
        "owner": "server",
        "family_id": "k1-functional-group-token",
        "question": "Does sparse chemistry-role token exchange add useful relation content to K1?",
        "hypothesis": {
            "hypothesis_id": f"H-{TRAJECTORY_ID}",
            "observed_deficiency": "Return-allocation changes were favorable but sub-material; K1 lacks an explicit intermediate chemical hierarchy.",
            "supporting_evidence_ids": [
                "pcqm-k1-v4-100k-reference-s42",
                "pcqm-k1-pair-token-100k-s42",
                "pcqm-k1-pair-token-node-return-100k-s42",
            ],
            "alternative_explanations": [
                "existing OGB atom/bond states already encode the same roles",
                "the 12 role classes may merge chemically distinct instances",
            ],
            "changed_mechanism": "one sparse layer6 atom-to-functional-group-token-to-member-atom exchange",
            "cheapest_falsifier": "one seed42 fixed PCQM100K candidate-only screen against the reusable K1 reference",
            "related_closed_family_ids": [
                "k1-pair-token-node-return",
                "k1-global-motif-counts",
                "k1-ring-and-path-caches",
                "k1-geometry-branches",
            ],
            "expected_native_cost_ref": cost["cost_event_id"],
            "decision_changed_if_positive": "retain one seed42 shortlist; scale requires separate authority",
            "decision_changed_if_negative": "close chemistry-role token hierarchy before another route",
            "historical_unknowns": [
                "training stochasticity is unavailable and is not inferred from account identity"
            ],
        },
        "state_at_start": {
            "source_commit": source_commit,
            "source_config_identity": architecture_identity,
            "contract_refs": [rel("training_contract.json")],
            "reference_ids": [reference["reference_id"]],
            "parent_trajectory_ids": ["TC-k1-pair-token-node-return-100k-s42"],
            "prior_trajectory_ids": [
                "TC-k1-pair-token-100k-s42",
                "TC-k1-pair-token-node-return-100k-s42",
            ],
            "prior_evidence_ids": [
                "pcqm-k1-v4-100k-reference-s42",
                "pcqm-k1-pair-token-100k-s42",
                "pcqm-k1-pair-token-node-return-100k-s42",
            ],
            "role_snapshot_refs": [rel("role_plan.json")],
            "budget_snapshot_ref": rel("budget_snapshot.json"),
        },
        "actions": [
            {
                "action_id": "A001",
                "type": "cpu_only_sidecar_build_and_no_model_acceptance",
                "run_ids": ["kaseichou/molgap-k1-functional-group-sidecar-v1:v1"],
                "attempt_ids": ["sidecar-v1"],
                "source_commit": sidecar["source_commit"],
                "evidence_refs": [rel("sidecar_acceptance_summary.json")],
                "cost_event_ids": [],
            },
            {
                "action_id": "A002",
                "type": "planned_candidate_only_seed42_100k_screen",
                "run_ids": ["kaseichou/molgap-k1-functional-group-token-s42:v1-planned"],
                "attempt_ids": ["candidate-v1-planned"],
                "source_commit": source_commit,
                "evidence_refs": [
                    rel("comparison_readiness_prelaunch.json"),
                    rel("source_config.json"),
                ],
                "cost_event_ids": [cost["cost_event_id"]],
            },
        ],
        "result": {
            "evidence_ids": [EVIDENCE_ID],
            "evidence_refs": [rel("v5_evidence.json")],
        },
        "decision": {
            "decision_ref": rel("STATUS.md"),
            "outcome": "ACTIVE",
            "next_allowed_actions": ["submit exactly one frozen candidate job after packaging checks"],
            "reopen_conditions": ["terminal attribution must precede any successor"],
        },
        "comparison_class": "NO_COMPARISON",
        "comparison_readiness_ref": rel("comparison_readiness_prelaunch.json"),
        "comparison_blockers": [],
        "reference_bundle_id": reference["reference_bundle_id"],
    }
    atomic_json(ROOT / "trajectory.json", trajectory)
    artifacts = []
    for name in (
        "source_config.json",
        "training_contract.json",
        "comparison_readiness_prelaunch.json",
        "role_plan.json",
        "trace_plan.json",
        "budget_snapshot.json",
        "sidecar_acceptance_summary.json",
    ):
        artifacts.append(
            {
                "name": name.removesuffix(".json"),
                "locator": f"repo://{rel(name)}",
                "sha256": sha256_file(ROOT / name),
                "availability": "committed_prelaunch_metadata",
            }
        )
    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": EVIDENCE_ID,
        "track": "C",
        "scope": "prospective_fixed_100k_architecture_screen",
        "legacy_contract": "none-prospective-v5",
        "outcome": {
            "execution_status": "planned",
            "artifact_status": "prelaunch_complete_terminal_pending",
            "comparison_status": "prelaunch_strict_planned",
            "scientific_status": "pending",
            "transfer_status": "not_evaluated",
            "budget_decision": "one_candidate_authorized",
            "full_handoff_status": "not_applicable",
        },
        "authority": {
            "pointers": [
                rel("protocol.md"),
                rel("training_contract.json"),
                rel("source_config.json"),
                rel("comparison_readiness_prelaunch.json"),
                rel("trajectory.json"),
            ]
        },
        "artifacts": artifacts,
        "role_use": {
            "internal_development": "untouched",
            "official_validation": "untouched",
            "test_dev": "untouched",
            "test_challenge": "untouched",
        },
        "migration": {
            "migrated_at": "2026-09-20",
            "verification_scope": "prospective prelaunch metadata only; terminal evidence not yet claimed",
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
        },
    }
    atomic_json(ROOT / "v5_evidence.json", evidence)


if __name__ == "__main__":
    main()
