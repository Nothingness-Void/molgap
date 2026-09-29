"""Freeze this diagnostic using existing V5 gate and RML planning APIs."""
import copy
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from molgap.comparison_readiness import assess_comparison_prelaunch, ROLE_EVENT_KINDS, TRACE_FIELD_DECLARATIONS
from molgap.server_acceptance import write_server_comparison_prelaunch
from molgap.research_memory.plan import plan
from molgap.research_memory.trace import atomic_write, json_bytes, file_digest
from molgap.k1_representation_audit import panel_ids, MODEL500, PAYLOAD500, TRANSFORM500


def main():
    root = Path.cwd()
    base = Path("experiments/pcqm_k1_explainability_audit")
    dest = base / "representation"
    if (dest / "rml_plan").exists():
        raise ValueError("Do not overwrite a frozen diagnostic")
    dest.mkdir(exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    timestamp = datetime.now(timezone.utc).isoformat()
    refpath = Path("experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json")
    ref = json.loads(refpath.read_text())
    roles = {k: "applicable" if k in {"prediction_input", "labels_read", "metric_computed"} else "not_applicable" for k in ROLE_EVENT_KINDS}
    write_server_comparison_prelaunch(dest / "comparison_readiness_prelaunch.json",
        comparison_prelaunch=assess_comparison_prelaunch(
            candidate_id="k1-representation-100k-500k", candidate_plan={
                "source_config_status": "frozen", "source_commit_or_archive": commit},
            reference_id=ref["reference_id"], reference_bundle=ref, experiment_purpose="NO_TRAIN",
            intervention_group_id="representation-observation-only", declared_intervention_fields=[],
            role_applicability_plan=roles, trace_plan={k: k in {"checkpoint_identity", "epoch_or_pass", "live_dev_metric"} for k in TRACE_FIELD_DECLARATIONS},
            runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
                "qualification_scope": "frozen payload reproduction then hook invariance and no cross-graph gradient coupling"}),
        experiment_purpose="NO_TRAIN", reference_bundle=ref, repo_root=root, reference_bundle_path=refpath)
    def write(name, value):
        atomic_write(dest / name, json_bytes(value))
    write("row_manifest.json", {"source_idx": panel_ids(), "selection": "sha256-label-independent-v1"})
    write("role_plan.json", {"role_applicability": roles, "evaluation_role": "internal-development-only",
        "panel_source_range": [500000,550000], "original100_reproduction_only": [100000,150000],
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False})
    write("budget_snapshot.json", {"platform": "scnet-kunshan", "max_device_hours": .5,
        "max_jobs": 1, "training_authorized": False, "auto_retry_authorized": False})
    write("input_binding.json", {"model500_sha256": MODEL500, "payload500_sha256": PAYLOAD500,
        "transform500_sha256": TRANSFORM500, "authority500": "experiments/pcqm_k1_pair_token_scale_attribution/results/round1_terminal.json",
        "model100_reference": str(refpath).replace("\\", "/"), "row_manifest_sha256": file_digest(dest / "row_manifest.json")})
    policy = json.loads(Path("research_memory/policies/k1-relation-frozen-intervention.v1.json").read_text())
    policy.update(policy_id="k1-representation-observation", created_from_source_digest=file_digest(base / "representation_protocol.md"))
    policy["approval"] = {"activation": "one-user-authorized-NO_TRAIN-diagnostic", "authority_ref": (base / "representation_protocol.md").as_posix()}
    policy["comparability_selector"] = {"scientific_contract": "frozen-K1-100k-500k-context-only"}
    policy["action_rule"] = {"action": "RUN_REPRESENTATION_DIAGNOSTIC", "field": "accepted_checkpoints_available", "operator": "eq", "threshold": True}
    policy["required_observable_fields"] = ["accepted_checkpoints_available"]
    atomic_write(Path("research_memory/policies/k1-representation-observation.v1.json"), json_bytes(policy))
    prior = json.loads(Path("experiments/pcqm_k1_cross_scale_frozen/trajectory.json").read_text())
    trajectory = copy.deepcopy(prior)
    tid = "TC-k1-representation-100k-500k"
    trajectory.update(trajectory_id=tid, family_id="k1-representation-observation", question="How do frozen local/global representations differ on identical development molecules across training scales?", actions=[], result={"evidence_ids": [], "evidence_refs": []})
    h = trajectory["hypothesis"]
    h.update(hypothesis_id="H-"+tid, observed_deficiency="Aggregate dispersion does not identify task-relevant information loss.",
        supporting_evidence_ids=[ref["reference_id"]], changed_mechanism="none; observe frozen activations and gradients",
        cheapest_falsifier="same 1024 label-independent rows, two accepted checkpoints",
        expected_native_cost_ref=(dest/"budget_snapshot.json").as_posix(),
        decision_changed_if_positive="Nominate a diagnostic-backed mechanism only; no automatic training",
        decision_changed_if_negative="Stop the information-collapse explanation without changing models")
    trajectory["state_at_start"].update(source_commit=commit, source_config_identity="k1-representation-observation-v1",
        contract_refs=[(base/"representation_protocol.md").as_posix(),(dest/"input_binding.json").as_posix()],
        reference_ids=[ref["reference_id"]], parent_trajectory_ids=[], prior_trajectory_ids=[prior["trajectory_id"]],
        prior_evidence_ids=[ref["reference_id"]], role_snapshot_refs=[(dest/"role_plan.json").as_posix()],
        budget_snapshot_ref=(dest/"budget_snapshot.json").as_posix())
    trajectory["decision"] = {"decision_ref":(base/"representation_protocol.md").as_posix(), "outcome":"ACTIVE",
        "next_allowed_actions":["one bounded frozen diagnostic and terminal acceptance"], "reopen_conditions":["new explicit authority"]}
    trajectory["actions"] = [{"action_id":"A001", "type":"planned_NO_TRAIN_representation_diagnostic",
        "run_ids":[], "attempt_ids":[], "source_commit":commit,
        "evidence_refs":[(dest/"input_binding.json").as_posix()], "cost_event_ids":[]}]
    plan(root, {"trajectory": trajectory, "decision_state": {"available_actions":["RUN_REPRESENTATION_DIAGNOSTIC","DEFER"],
        "chosen_action":"RUN_REPRESENTATION_DIAGNOSTIC", "policy_id":policy["policy_id"],"policy_version":"v1", "state_timestamp":timestamp}}, dest/"rml_plan")
    print("V5 reference-evidence gate and prospective RML plan PASS")


if __name__ == "__main__":
    main()
