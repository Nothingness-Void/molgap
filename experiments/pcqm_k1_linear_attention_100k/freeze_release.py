"""Freeze this experiment using shared release validation and native RML plan."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import subprocess

from molgap.constants import REPO_ROOT
from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.server_acceptance import write_server_comparison_prelaunch
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS
from molgap.k1_linear_attention import MODE
from molgap.research_memory.plan import plan
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint

REL = "experiments/pcqm_k1_linear_attention_100k"
ROOT = REPO_ROOT / REL
TEMPLATE = REPO_ROOT / "experiments/pcqm_k1_sparse_triplet_100k"
REFERENCE = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
RUN = "nothingnessvoid/molgap-k1-linear-attention-s42:v1"
POLICY = "k1-linear-attention-bounded-research"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    atomic_write(path, json_bytes(value))


def main():
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", REL],
                               cwd=REPO_ROOT, text=True).strip():
        raise RuntimeError("Commit source and protocol before freezing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    now = datetime.now(timezone.utc).isoformat()
    bundle = load(REFERENCE)
    role, trace = load(TEMPLATE / "role_plan.json"), load(TEMPLATE / "trace_plan.json")
    save(ROOT / "role_plan.json", role)
    save(ROOT / "trace_plan.json", trace)
    config = canonical_fingerprint(ARCHITECTURE_CONFIGS[MODE])
    save(ROOT / "source_config.json", {
        "format": "molgap-frozen-source-config-v1", "source_commit": commit,
        "candidate_ids": [MODE], "architecture_config_identities": {MODE: config},
        "training_contract_ref": f"{REL}/training_contract.json",
        "training_contract_sha256": file_digest(ROOT / "training_contract.json"),
        "implementation_refs": ["src/molgap/k1_linear_attention.py", "src/molgap/k1_screen_trace.py",
                                "src/molgap/pcqm_k1_variants_runner.py", "src/molgap/k1_portability_audit.py"],
    })
    save(ROOT / "budget_snapshot.json", {
        "available_pool": "Kaggle1-user-authorized", "available_device_hours": None,
        "estimated_training_device_hours": 3.0, "estimated_audit_device_hours": 1.0,
        "max_training_wall_hours": 6.0, "max_audit_wall_hours": 1.5,
        "desktop_jobs_must_not_be_interrupted": True, "successor_authorized": False,
    })
    identity = dict(bundle["comparison_identity"], architecture_config_identity=config)
    prelaunch = assess_comparison_prelaunch(
        candidate_id=MODE, reference_id=bundle["reference_id"], reference_bundle=bundle,
        candidate_plan={"comparison_identity": identity, "source_config_status": "frozen",
                        "source_commit_or_archive": commit},
        experiment_purpose="architecture_comparison", intervention_group_id="architecture",
        declared_intervention_fields=["architecture_config_identity"],
        role_applicability_plan={k: role[k] for k in ("training_membership", "prediction_input",
            "labels_read", "metric_computed", "selection_used", "external_submission")},
        trace_plan={k: trace[k] for k in ("optimizer_step", "sample_presentations", "epoch_or_pass",
            "learning_rate", "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity")},
        runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
                                    "qualification_scope": "fp32-no-tf32-bs128-optimizer-inclusive"},
    )
    write_server_comparison_prelaunch(ROOT / "comparison_readiness_prelaunch.json",
        comparison_prelaunch=prelaunch, experiment_purpose="architecture_comparison",
        reference_bundle=bundle, repo_root=REPO_ROOT, reference_bundle_path=REFERENCE)
    policy = {
        "schema": "molgap-policy-v1", "policy_id": POLICY, "version": "v1",
        "policy_type": "research_action", "status": "candidate",
        "comparability_selector": {"scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-40epochs"},
        "required_observable_fields": ["distinct_pure2d_question_frozen"],
        "created_from_source_digest": file_digest(ROOT / "protocol.md"),
        "approval": {"authority_ref": f"{REL}/protocol.md", "activation": "controller-only-bounded-user-authorization"},
        "cost_model": {"kind": "measured_only", "assumptions": ["no cross-hardware scalar conversion"]},
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "action_rule": {"field": "distinct_pure2d_question_frozen", "operator": "eq", "threshold": True,
                        "action": "RUN_BOUNDED_2D_SCREEN"}, "borderline_action": "DEFER",
    }
    save(REPO_ROOT / f"research_memory/policies/{POLICY}.v1.json", policy)
    for audit in (False, True):
        suffix = "audit_plan" if audit else "rml_plan"
        tid = "TC-k1-linear-attention-post100k-audit-s42" if audit else "TC-k1-linear-attention-100k-s42"
        cost_id = f"cost-{tid}"
        trajectory = copy.deepcopy(load(TEMPLATE / "trajectory.json"))
        trajectory.update(trajectory_id=tid, family_id="node-query-linear-global-exchange",
                          question="Does node-specific linear exchange improve portable 2D regression?")
        trajectory["hypothesis"].update(
            hypothesis_id=f"H-{tid}", expected_native_cost_ref=cost_id,
            observed_deficiency="K1 shared-slot variants lost late-horizon or frozen-role gains; query-specific retrieval is untested.",
            supporting_evidence_ids=[bundle["reference_id"]],
            alternative_explanations=["smooth kernel underfits", "node query overfits", "reused-role selection optimism"],
            changed_mechanism="frozen-checkpoint NO_TRAIN portability audit" if audit else ARCHITECTURE_CONFIGS[MODE]["change"],
            cheapest_falsifier="one candidate seed42 plus separately recorded frozen-500K inference",
            decision_changed_if_positive="propose an explicitly authorized late-horizon matched 500K bridge",
            decision_changed_if_negative="close exact mechanism without successor",
        )
        role_ref = f"{REL}/role_plan.json"
        if audit:
            audit_role = dict(role, training_membership="not_applicable", selection_used="not_applicable")
            save(ROOT / "audit_role_plan.json", audit_role)
            role_ref = f"{REL}/audit_role_plan.json"
        trajectory["state_at_start"] = {
            "source_commit": commit, "source_config_identity": config,
            "contract_refs": [f"{REL}/training_contract.json"], "reference_ids": [bundle["reference_id"]],
            "parent_trajectory_ids": ["TC-k1-linear-attention-100k-s42"] if audit else ["TC-k1-v4-100k-reference-s42"],
            "prior_trajectory_ids": [], "prior_evidence_ids": [bundle["reference_id"]],
            "role_snapshot_refs": [role_ref], "budget_snapshot_ref": f"{REL}/budget_snapshot.json",
        }
        trajectory["actions"] = [{"action_id": "A001", "type": "conditional_NO_TRAIN_audit" if audit else "bounded_100k_training",
            "run_ids": [RUN], "attempt_ids": ["v1"], "source_commit": commit,
            "evidence_refs": [f"{REL}/source_config.json"], "cost_event_ids": [cost_id]}]
        trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
        trajectory["decision"] = {"decision_ref": f"{REL}/protocol.md", "outcome": "ACTIVE",
            "next_allowed_actions": ["mechanically accepted training then frozen audit" if audit else "one frozen single-GPU submission"],
            "reopen_conditions": ["terminal attribution and explicit new compute authority"]}
        for key in ("comparison_readiness_ref", "comparison_class", "comparison_blockers", "reference_bundle_id"):
            trajectory.pop(key, None)
        if not audit:
            trajectory["comparison_readiness_ref"] = f"{REL}/comparison_readiness_prelaunch.json"
            trajectory["reference_bundle_id"] = bundle["reference_bundle_id"]
            trajectory["comparison_class"] = "NO_COMPARISON"
            trajectory["comparison_blockers"] = []
        cost = copy.deepcopy(load(TEMPLATE / "costs/expected_training.json"))
        cost.update(cost_event_id=cost_id, trajectory_id=tid, run_id=RUN, attempt_id="v1",
                    platform="kaggle1", hardware="requested_P100_actual_pending",
                    category="inference" if audit else "training", evidence_ref=f"{REL}/budget_snapshot.json")
        for unit in ("device_hours", "wall_hours"):
            cost["measurement"][unit] = {"value": 1.0 if audit else 3.0, "status": "estimated"}
        print(plan(REPO_ROOT, {"trajectory": trajectory, "costs": [cost], "decision_state": {
            "available_actions": ["RUN_BOUNDED_2D_SCREEN", "DEFER"], "chosen_action": "RUN_BOUNDED_2D_SCREEN",
            "policy_id": POLICY, "policy_version": "v1", "state_timestamp": now,
        }}, f"{REL}/{suffix}"))


if __name__ == "__main__":
    main()
