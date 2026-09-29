"""New prospective attempt after a version-1 pre-model device-gate failure."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import subprocess

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.constants import REPO_ROOT
from molgap.k1_motif_hierarchy import MODE
from molgap.k1_motif_study_runtime import RUN_ID, TRAJECTORY_ID
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS
from molgap.research_memory.plan import plan
from molgap.research_memory.trace import file_digest
from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import write_server_comparison_prelaunch

from .freeze_gpu_release import POLICY, REFERENCE, REL, ROOT, load, save


ATTEMPT = ROOT / "attempt_v2"


def main():
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", REL], cwd=REPO_ROOT, text=True
    ).strip():
        raise RuntimeError("Commit source and retry protocol before version-2 release")
    if (ATTEMPT / "source_config.json").exists():
        raise FileExistsError("Version-2 release already frozen")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    bundle = load(REFERENCE)
    config = canonical_fingerprint(ARCHITECTURE_CONFIGS[MODE])
    role, trace = load(ROOT / "role_plan.json"), load(ROOT / "trace_plan.json")
    save(ATTEMPT / "source_config.json", {
        "format": "molgap-frozen-source-config-v1", "source_commit": commit,
        "candidate_id": MODE, "architecture_config_identity": config,
        "training_contract_ref": f"{REL}/gpu_training_contract.json",
        "training_contract_sha256": file_digest(ROOT / "gpu_training_contract.json"),
        "motif_manifest_sha256": "77f1bff1fdd32c94bc5d39b53d28f79de51c35e804a2b3555f1f1ae8c7a3234b",
        "motif_aggregate_sha256": "5466ccd1f498619b045eb73d82f958949c1474ff0d303b99d6fd226737a0b8ae",
        "implementation_refs": [
            "src/molgap/k1_motif_hierarchy.py",
            "src/molgap/pcqm_motif_sidecar.py",
            "src/molgap/pcqm_k1_variants_runner.py",
            "src/molgap/k1_motif_study_runtime.py",
        ],
        "infrastructure_change_from_v1": "accept_one_actual_qualified_P100_or_T4",
    })
    prelaunch = assess_comparison_prelaunch(
        candidate_id=MODE, reference_id=bundle["reference_id"], reference_bundle=bundle,
        candidate_plan={
            "comparison_identity": dict(bundle["comparison_identity"],
                                        architecture_config_identity=config),
            "source_config_status": "frozen", "source_commit_or_archive": commit,
        },
        experiment_purpose="architecture_comparison", intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
        role_applicability_plan={key: role[key] for key in (
            "training_membership", "prediction_input", "labels_read", "metric_computed",
            "selection_used", "external_submission")},
        trace_plan={key: trace[key] for key in (
            "optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate",
            "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity")},
        runtime_qualification_plan={
            "status": "declared", "runtime_certificate_required": True,
            "qualification_scope": "single-device-fp32-no-tf32-bs128-optimizer-inclusive",
        },
    )
    write_server_comparison_prelaunch(
        ATTEMPT / "comparison_readiness_prelaunch.json",
        comparison_prelaunch=prelaunch, experiment_purpose="architecture_comparison",
        reference_bundle=bundle, repo_root=REPO_ROOT, reference_bundle_path=REFERENCE,
    )
    save(ATTEMPT / "budget_snapshot.json", {
        "format": "molgap-budget-snapshot-v1", "trajectory_id": TRAJECTORY_ID,
        "pool": "Kaggle2-user-authorized-infrastructure-repair",
        "current_account_quota_hours": None, "reserved_device_hours_ceiling": 10,
        "version_1_pre_model_seconds_observed": 174.56,
        "automatic_successor_authorized": False,
    })
    cost_id = f"cost-{TRAJECTORY_ID}"
    cost = copy.deepcopy(load(ROOT / "gpu_costs/expected_training.json"))
    cost.update(cost_event_id=cost_id, trajectory_id=TRAJECTORY_ID,
                run_id=RUN_ID, attempt_id="v2",
                hardware="actual_P100_or_T4_pending",
                evidence_ref=f"{REL}/attempt_v2/budget_snapshot.json")
    # The estimated cost is an input snapshot, not a discovered observed RML
    # event. Only plan() publishes the canonical cost under gpu_rml_plan/costs.
    save(ATTEMPT / "gpu_costs/expected_training.json", cost)
    trajectory = copy.deepcopy(load(ROOT / "gpu_rml_plan/trajectory.json"))
    trajectory.update(trajectory_id=TRAJECTORY_ID,
                      question="Can typed motif communication improve K1 after a clean runtime qualification?")
    trajectory["hypothesis"].update(
        hypothesis_id=f"H-{TRAJECTORY_ID}", expected_native_cost_ref=cost_id,
        historical_unknowns=["v1 actual accelerator was not reported; v1 did no model work"],
    )
    trajectory["state_at_start"] = {
        "source_commit": commit, "source_config_identity": config,
        "contract_refs": [f"{REL}/gpu_training_contract.json"],
        "reference_ids": [bundle["reference_id"]],
        "parent_trajectory_ids": ["TC-k1-v4-100k-reference-s42"],
        "prior_trajectory_ids": ["TC-k1-motif-hierarchy-100k-s42"],
        "prior_evidence_ids": [bundle["reference_id"]],
        "role_snapshot_refs": [f"{REL}/role_plan.json"],
        "budget_snapshot_ref": f"{REL}/attempt_v2/budget_snapshot.json",
    }
    trajectory["actions"] = [{
        "action_id": "A001", "type": "bounded_100k_training",
        "run_ids": [RUN_ID], "attempt_ids": ["v2"], "source_commit": commit,
        "evidence_refs": [f"{REL}/attempt_v2/source_config.json"],
        "cost_event_ids": [cost_id],
    }]
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
    trajectory["decision"] = {
        "decision_ref": f"{REL}/attempt_v2/protocol.md", "outcome": "ACTIVE",
        "next_allowed_actions": ["one repaired infrastructure submission"],
        "reopen_conditions": ["accepted terminal attribution and explicit compute decision"],
    }
    trajectory["comparison_readiness_ref"] = f"{REL}/attempt_v2/comparison_readiness_prelaunch.json"
    trajectory["comparison_class"] = "NO_COMPARISON"
    trajectory["comparison_blockers"] = []
    trajectory["reference_bundle_id"] = bundle["reference_bundle_id"]
    trajectory.pop("decision_state", None)
    planned = plan(REPO_ROOT, {"trajectory": trajectory, "costs": [cost],
        "decision_state": {
            "available_actions": ["RUN_BOUNDED_MOTIF_SCREEN", "DEFER"],
            "chosen_action": "RUN_BOUNDED_MOTIF_SCREEN", "policy_id": POLICY,
            "policy_version": "v1", "state_timestamp": datetime.now(timezone.utc).isoformat(),
        }}, f"{REL}/attempt_v2/gpu_rml_plan")
    print(json.dumps({"source_commit": commit, "prelaunch_ready": prelaunch["prelaunch_ready"],
                      "trajectory_id": TRAJECTORY_ID, "planned": bool(planned)}, indent=2))


if __name__ == "__main__":
    main()
