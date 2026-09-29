"""Bind one committed motif source to real K1 evidence and prospective RML."""
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
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import write_server_comparison_prelaunch


REL = "experiments/pcqm_motif_hierarchy_100k"
ROOT = REPO_ROOT / REL
TEMPLATE = REPO_ROOT / "experiments/pcqm_k1_sparse_triplet_100k"
REFERENCE = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
POLICY = "k1-motif-hierarchy-screen"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json_bytes(value))


def main():
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", REL], cwd=REPO_ROOT, text=True
    ).strip():
        raise RuntimeError("Commit source, runner and protocol before release")
    if (ROOT / "source_config.json").exists():
        raise FileExistsError("Frozen source identity already exists")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    bundle = load(REFERENCE)
    config = canonical_fingerprint(ARCHITECTURE_CONFIGS[MODE])
    role, trace = load(TEMPLATE / "role_plan.json"), load(TEMPLATE / "trace_plan.json")
    save(ROOT / "role_plan.json", role)
    save(ROOT / "trace_plan.json", trace)
    save(ROOT / "source_config.json", {
        "format": "molgap-frozen-source-config-v1",
        "source_commit": commit,
        "candidate_id": MODE,
        "architecture_config_identity": config,
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
    })
    candidate_identity = dict(bundle["comparison_identity"], architecture_config_identity=config)
    prelaunch = assess_comparison_prelaunch(
        candidate_id=MODE, reference_id=bundle["reference_id"], reference_bundle=bundle,
        candidate_plan={"comparison_identity": candidate_identity,
                        "source_config_status": "frozen", "source_commit_or_archive": commit},
        experiment_purpose="architecture_comparison", intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
        role_applicability_plan={key: role[key] for key in (
            "training_membership", "prediction_input", "labels_read",
            "metric_computed", "selection_used", "external_submission")},
        trace_plan={key: trace[key] for key in (
            "optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate",
            "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity")},
        runtime_qualification_plan={
            "status": "declared", "runtime_certificate_required": True,
            "qualification_scope": "single-device-fp32-no-tf32-bs128-optimizer-inclusive"},
    )
    write_server_comparison_prelaunch(
        ROOT / "comparison_readiness_prelaunch.json",
        comparison_prelaunch=prelaunch,
        experiment_purpose="architecture_comparison", reference_bundle=bundle,
        repo_root=REPO_ROOT, reference_bundle_path=REFERENCE,
    )
    save(ROOT / "gpu_budget_snapshot.json", {
        "format": "molgap-budget-snapshot-v1", "trajectory_id": TRAJECTORY_ID,
        "pool": "Kaggle2-user-authorized-bounded-screen", "current_account_quota_hours": None,
        "reserved_device_hours_ceiling": 10, "automatic_successor_authorized": False,
        "one_worker_reason": "only one independent approved candidate; no baseline retrain",
    })
    cost_id = f"cost-{TRAJECTORY_ID}"
    cost = copy.deepcopy(load(TEMPLATE / "costs/expected_training.json"))
    cost.update(cost_event_id=cost_id, trajectory_id=TRAJECTORY_ID, run_id=RUN_ID,
                attempt_id="v1", platform="kaggle2", hardware="Tesla_P100_16GB",
                evidence_ref=f"{REL}/gpu_budget_snapshot.json")
    for unit in ("device_hours", "wall_hours"):
        cost["measurement"][unit] = {"value": 10.0, "status": "estimated"}
    save(ROOT / "gpu_costs/expected_training.json", cost)
    trajectory = copy.deepcopy(load(TEMPLATE / "trajectory.json"))
    trajectory.update(trajectory_id=TRAJECTORY_ID, family_id="k1-motif-hierarchy",
                      question="Does a typed non-overlapping motif graph improve K1 Gap prediction?")
    trajectory["hypothesis"].update(
        hypothesis_id=f"H-{TRAJECTORY_ID}",
        observed_deficiency="K1 has real-bond EdgeState and a molecule slot, but no explicit communication between disjoint chemistry motifs.",
        supporting_evidence_ids=[bundle["reference_id"]],
        alternative_explanations=["K1 slot already encodes the same motif topology",
                                  "additional parameters overfit the reused development role"],
        changed_mechanism=ARCHITECTURE_CONFIGS[MODE]["change"],
        cheapest_falsifier="one frozen seed42 100K candidate versus the accepted K1 reference",
        related_closed_family_ids=["k1-functional-group-token", "k1-conjugated-hyperedge"],
        expected_native_cost_ref=cost_id,
        decision_changed_if_positive="consider one separate accepted-checkpoint NO_TRAIN 500K audit",
        decision_changed_if_negative="close this motif graph exchange without seed or scale successor",
    )
    trajectory["state_at_start"] = {
        "source_commit": commit, "source_config_identity": config,
        "contract_refs": [f"{REL}/gpu_training_contract.json"],
        "reference_ids": [bundle["reference_id"]],
        "parent_trajectory_ids": ["TC-k1-v4-100k-reference-s42"],
        "prior_trajectory_ids": [], "prior_evidence_ids": [bundle["reference_id"]],
        "role_snapshot_refs": [f"{REL}/role_plan.json"],
        "budget_snapshot_ref": f"{REL}/gpu_budget_snapshot.json",
    }
    trajectory["actions"] = [{
        "action_id": "A001", "type": "bounded_100k_training", "run_ids": [RUN_ID],
        "attempt_ids": ["v1"], "source_commit": commit,
        "evidence_refs": [f"{REL}/source_config.json"], "cost_event_ids": [cost_id],
    }]
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
    trajectory["decision"] = {
        "decision_ref": f"{REL}/gpu_protocol.md", "outcome": "ACTIVE",
        "next_allowed_actions": ["one frozen candidate submission"],
        "reopen_conditions": ["terminal attribution and explicit compute decision"],
    }
    trajectory["comparison_readiness_ref"] = f"{REL}/comparison_readiness_prelaunch.json"
    trajectory["comparison_class"] = "NO_COMPARISON"
    trajectory["comparison_blockers"] = []
    trajectory["reference_bundle_id"] = bundle["reference_bundle_id"]
    planned = plan(REPO_ROOT, {"trajectory": trajectory, "costs": [cost],
        "decision_state": {
            "available_actions": ["RUN_BOUNDED_MOTIF_SCREEN", "DEFER"],
            "chosen_action": "RUN_BOUNDED_MOTIF_SCREEN", "policy_id": POLICY,
            "policy_version": "v1", "state_timestamp": datetime.now(timezone.utc).isoformat(),
        }}, f"{REL}/gpu_rml_plan")
    print(json.dumps({"source_commit": commit, "prelaunch_ready": prelaunch["prelaunch_ready"],
                      "trajectory_id": TRAJECTORY_ID, "planned": bool(planned)}, indent=2))


if __name__ == "__main__":
    main()
