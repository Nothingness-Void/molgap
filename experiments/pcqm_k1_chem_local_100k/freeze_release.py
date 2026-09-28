"""Freeze two training trajectories and a conditional NO_TRAIN audit plan."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.constants import REPO_ROOT
from molgap.k1_chem_local import MODES
from molgap.k1_chem_local_study_runtime import RUN_ID, TRAJECTORIES
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS
from molgap.research_memory.plan import plan
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint
from molgap.server_acceptance import write_server_comparison_prelaunch


REL = "experiments/pcqm_k1_chem_local_100k"
ROOT = REPO_ROOT / REL
TEMPLATE = REPO_ROOT / "experiments/pcqm_k1_sparse_triplet_100k"
REFERENCE = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
POLICY = "k1-chem-local-bounded-screen"
AUDIT_ID = "TC-k1-chem-local-post100k-audit-s42"
AUDIT_RUN = "kaseichou/molgap-k1-chem-local-audit-s42:v1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    atomic_write(path, json_bytes(value))


def main():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", REL], cwd=REPO_ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Commit source, runner, protocol and freeze adapter before release")
    if (ROOT / "source_config.json").exists():
        raise FileExistsError("Do not overwrite frozen source identity")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    bundle = load(REFERENCE)
    role, trace = load(TEMPLATE / "role_plan.json"), load(TEMPLATE / "trace_plan.json")
    now = datetime.now(timezone.utc).isoformat()
    save(ROOT / "role_plan.json", role)
    save(ROOT / "trace_plan.json", trace)
    save(ROOT / "audit_role_plan.json", dict(role, training_membership="not_applicable", selection_used="not_applicable"))
    configs = {mode: canonical_fingerprint(ARCHITECTURE_CONFIGS[mode]) for mode in MODES}
    save(ROOT / "source_config.json", {
        "format": "molgap-frozen-source-config-v1", "source_commit": commit,
        "candidate_ids": list(MODES), "architecture_config_identities": configs,
        "training_contract_ref": f"{REL}/training_contract.json",
        "training_contract_sha256": file_digest(ROOT / "training_contract.json"),
        "implementation_refs": ["src/molgap/k1_chem_local.py",
            "src/molgap/k1_chem_local_study_runtime.py",
            "src/molgap/pcqm_k1_variants_runner.py"],
    })
    save(ROOT / "budget_snapshot.json", {
        "available_pool": "user-authorized-Kaggle2-bounded-research",
        "available_device_hours": None,
        "estimated_training_device_hours_per_arm": 3,
        "max_training_wall_hours_per_worker": 6,
        "max_audit_device_hours": 1.5,
        "maximum_training_active_device_hours": 12,
        "bootstrap_overhead_and_idle_allocations_recorded_separately": True,
        "automatic_successor_authorized": False,
    })
    save(REPO_ROOT / f"research_memory/policies/{POLICY}.v1.json", {
        "schema": "molgap-policy-v1", "policy_id": POLICY, "version": "v1",
        "policy_type": "research_action", "status": "candidate",
        "comparability_selector": {"scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-40epochs"},
        "required_observable_fields": ["local_color_hypothesis_frozen"],
        "created_from_source_digest": file_digest(ROOT / "protocol.md"),
        "approval": {"authority_ref": f"{REL}/protocol.md", "activation": "controller-only-bounded-user-authorization"},
        "cost_model": {"kind": "measured_only", "assumptions": ["no cross-hardware scalar conversion"]},
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "action_rule": {"field": "local_color_hypothesis_frozen", "operator": "eq", "threshold": True,
                        "action": "RUN_BOUNDED_LOCAL_COLOR_STUDY"},
        "borderline_action": "DEFER",
    })
    created = []
    for mode in (*MODES, "audit"):
        audit = mode == "audit"
        tid = AUDIT_ID if audit else TRAJECTORIES[mode]
        sub = "audit" if audit else "arms/" + mode.removeprefix("neural_atom_k1_")
        config = canonical_fingerprint(configs) if audit else configs[mode]
        run = AUDIT_RUN if audit else RUN_ID
        if not audit:
            identity = dict(bundle["comparison_identity"], architecture_config_identity=config)
            prelaunch = assess_comparison_prelaunch(
                candidate_id=mode, reference_id=bundle["reference_id"], reference_bundle=bundle,
                candidate_plan={"comparison_identity": identity,
                    "source_config_status": "frozen", "source_commit_or_archive": commit},
                experiment_purpose="architecture_comparison", intervention_group_id="architecture",
                declared_intervention_fields=["architecture_config_identity"],
                role_applicability_plan={key: role[key] for key in (
                    "training_membership", "prediction_input", "labels_read",
                    "metric_computed", "selection_used", "external_submission")},
                trace_plan={key: trace[key] for key in (
                    "optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate",
                    "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity")},
                runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
                    "qualification_scope": "fp32-no-tf32-bs128-optimizer-inclusive"})
            write_server_comparison_prelaunch(ROOT / sub / "comparison_readiness_prelaunch.json",
                comparison_prelaunch=prelaunch, experiment_purpose="architecture_comparison",
                reference_bundle=bundle, repo_root=REPO_ROOT, reference_bundle_path=REFERENCE)
        cost_id = f"cost-{tid}"
        trajectory = copy.deepcopy(load(TEMPLATE / "trajectory.json"))
        trajectory.update(trajectory_id=tid, family_id="k1-chemistry-separated-local",
            question="Does delayed mixing of typed real-bond messages improve portable K1 Gap prediction?")
        trajectory["hypothesis"].update(
            hypothesis_id=f"H-{tid}", expected_native_cost_ref=cost_id,
            observed_deficiency="Prior K1 local capacity overfit; global relation modifications lost 100K gains on frozen500K molecules.",
            supporting_evidence_ids=[bundle["reference_id"]],
            alternative_explanations=["new capacity rather than chemistry", "sparse color buckets", "reused-role selection optimism"],
            changed_mechanism=("conditional accepted-checkpoint NO_TRAIN portability audit" if audit
                               else ARCHITECTURE_CONFIGS[mode]["change"] + ":" + ARCHITECTURE_CONFIGS[mode]["color_source"]),
            cheapest_falsifier="two equal-capacity seed42 100K arms, then only eligible accepted-checkpoint NO_TRAIN audit",
            decision_changed_if_positive="retain one shortlist candidate pending explicit 500K compute decision",
            decision_changed_if_negative="close this color separation without a seed or scale successor")
        role_ref = f"{REL}/{'audit_role_plan' if audit else 'role_plan'}.json"
        trajectory["state_at_start"] = {
            "source_commit": commit, "source_config_identity": config,
            "contract_refs": [f"{REL}/training_contract.json"],
            "reference_ids": [bundle["reference_id"]],
            "parent_trajectory_ids": list(TRAJECTORIES.values()) if audit else ["TC-k1-v4-100k-reference-s42"],
            "prior_trajectory_ids": [], "prior_evidence_ids": [bundle["reference_id"]],
            "role_snapshot_refs": [role_ref], "budget_snapshot_ref": f"{REL}/budget_snapshot.json"}
        trajectory["actions"] = [{"action_id": "A001", "type": "conditional_NO_TRAIN_audit" if audit else "bounded_100k_training",
            "run_ids": [run], "attempt_ids": ["v1"], "source_commit": commit,
            "evidence_refs": [f"{REL}/source_config.json"], "cost_event_ids": [cost_id]}]
        trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
        trajectory["decision"] = {"decision_ref": f"{REL}/protocol.md", "outcome": "ACTIVE",
            "next_allowed_actions": ["accepted training then conditional NO_TRAIN audit" if audit else "one frozen candidate submission"],
            "reopen_conditions": ["terminal attribution and explicit new compute decision"]}
        for key in ("comparison_readiness_ref", "comparison_class", "comparison_blockers", "reference_bundle_id"):
            trajectory.pop(key, None)
        if not audit:
            trajectory.update(comparison_readiness_ref=f"{REL}/{sub}/comparison_readiness_prelaunch.json",
                comparison_class="NO_COMPARISON", comparison_blockers=[],
                reference_bundle_id=bundle["reference_bundle_id"])
        cost = copy.deepcopy(load(TEMPLATE / "costs/expected_training.json"))
        cost.update(cost_event_id=cost_id, trajectory_id=tid, run_id=run, attempt_id="v1",
            platform="kaggle2", hardware="actual_GPU_pending", category="inference" if audit else "training",
            evidence_ref=f"{REL}/budget_snapshot.json")
        for unit in ("device_hours", "wall_hours"):
            cost["measurement"][unit] = {"value": 1.0 if audit else 3.0, "status": "estimated"}
        created.append(plan(REPO_ROOT, {"trajectory": trajectory, "costs": [cost], "decision_state": {
            "available_actions": ["RUN_BOUNDED_LOCAL_COLOR_STUDY", "DEFER"],
            "chosen_action": "RUN_BOUNDED_LOCAL_COLOR_STUDY", "policy_id": POLICY,
            "policy_version": "v1", "state_timestamp": now}}, f"{REL}/{sub}/rml_plan"))
    print(json.dumps({"source_commit": commit, "plan_count": len(created)}, indent=2))


if __name__ == "__main__":
    main()
