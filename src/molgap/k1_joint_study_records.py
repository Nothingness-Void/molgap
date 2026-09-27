"""Narrow prospective/publication adapter using existing V5 and RML gates."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .constants import REPO_ROOT
from .k1_joint_study_runtime import ATTEMPT, AUDIT_TRAJECTORY, RECIPES, RUN_ID, TRAJECTORIES
from .research_memory.trace import atomic_write, file_digest, json_bytes

REL = "experiments/pcqm_k1_joint_atom_reconstruction_100k"
RELEASE_REL = f"{REL}/attempts/v{ATTEMPT}"
REFERENCE = "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
TRANSFORM = "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/target_transform.json"
POLICY = "k1-joint-atom-bounded-research"
DATASET = "kaseichou/molgap-k1-joint-atom-source"
TRANSFORM_FILE_SHA = "20e6730d57080b0bec9901a26a931034aad162848e0075940fd1e1273bf084b3"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    atomic_write(Path(path), json_bytes(value))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def freeze():
    from .comparison_readiness import assess_comparison_prelaunch
    from .server_acceptance import write_server_comparison_prelaunch
    from .k1_joint_objective import objective_config, objective_fingerprint
    from .research_memory.plan import plan, plan_many

    root = REPO_ROOT / RELEASE_REL
    authority_root = REPO_ROOT / REL
    if git("status", "--porcelain", "--", "src", REL):
        raise RuntimeError("Commit executable source and protocol before freezing")
    if (root / "source_config.json").exists():
        raise FileExistsError("Do not replace an existing prospective release")
    commit = git("rev-parse", "HEAD")
    bundle = load(REPO_ROOT / REFERENCE)
    role = {"format": "molgap-v5-role-applicability-plan-v1",
        **{k: "applicable" for k in ("training_membership", "prediction_input", "labels_read", "metric_computed", "selection_used")},
        "external_submission": "not_applicable",
        "protected_roles": {k: "forbidden" for k in ("official_validation", "test_dev", "test_challenge")}}
    trace = {"format": "molgap-v5-trace-plan-v1",
        **{k: True for k in ("optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate", "live_train_metric", "live_dev_metric", "checkpoint_identity")},
        "ema_dev_metric": False, "atomic_checkpoint_every_epochs": 1,
        "retrievable_recovery_chunk_every_epochs": 10}
    audit_role = dict(role, training_membership="not_applicable", selection_used="not_applicable")
    for name, value in (("role_plan", role), ("trace_plan", trace), ("audit_role_plan", audit_role)):
        save(root / (name + ".json"), value)
    configs = {r: objective_config(r) for r in RECIPES}
    fingerprints = {r: objective_fingerprint(r) for r in RECIPES}
    save(root / "source_config.json", {
        "format": "molgap-frozen-source-config-v1", "source_commit": commit,
        "candidate_ids": list(RECIPES), "run_id": RUN_ID,
        "objective_identities": fingerprints,
        "training_contract_ref": f"{REL}/training_contract.json",
        "training_contract_sha256": file_digest(authority_root / "training_contract.json"),
        "attempt_id": f"v{ATTEMPT}",
        "prior_failure_ref": f"{REL}/cancellation_v2/decision.md",
        "implementation_refs": ["src/molgap/k1_joint_objective.py", "src/molgap/pcqm_k1_variants_runner.py", "src/molgap/k1_joint_study_runtime.py"],
        "registry_capability_gap": "existing ExperimentSpec K1 mode binds a full-run recipe, not this 100K objective intervention; use direct shared RML and source-bundle APIs"})
    save(root / "budget_snapshot.json", {
        "authorization": "one Kaggle2 T4x2 dual-arm seed42 study plus bounded terminal inference",
        "available_device_hours": None, "quota_api_verified": False,
        "maximum_worker_wall_hours_including_audit": 6,
        "maximum_allocated_T4_hours_excluding_bootstrap": 12,
        "estimated_training_device_hours_per_arm": 3,
        "estimated_audit_device_hours_total": 1,
        "record_bootstrap_and_idle_device_allocation": True,
        "automatic_training_successor_authorized": False})
    save(REPO_ROOT / f"research_memory/policies/{POLICY}.v1.json", {
        "schema": "molgap-policy-v1", "policy_id": POLICY, "version": "v1",
        "policy_type": "research_action", "status": "candidate",
        "comparability_selector": {"scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-40epochs"},
        "required_observable_fields": ["joint_gap_objective_question_frozen"],
        "created_from_source_digest": file_digest(authority_root / "protocol.md"),
        "approval": {"authority_ref": f"{REL}/protocol.md", "activation": "controller-only-bounded-user-authorization"},
        "cost_model": {"kind": "measured_only", "assumptions": ["no cross-hardware scalar conversion"]},
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "action_rule": {"field": "joint_gap_objective_question_frozen", "operator": "eq", "threshold": True,
                        "action": "RUN_BOUNDED_JOINT_STUDY"}, "borderline_action": "DEFER"})
    for recipe in RECIPES:
        arm = root / "arms" / recipe
        save(arm / "objective_config.json", configs[recipe])
        identity = dict(bundle["comparison_identity"], loss_identity=fingerprints[recipe])
        prelaunch = assess_comparison_prelaunch(candidate_id=recipe,
            reference_id=bundle["reference_id"], reference_bundle=bundle,
            candidate_plan={"comparison_identity": identity, "source_config_status": "frozen", "source_commit_or_archive": commit},
            experiment_purpose="training_objective_comparison", intervention_group_id="joint-training-objective",
            declared_intervention_fields=["loss_identity"],
            role_applicability_plan={k: role[k] for k in ("training_membership", "prediction_input", "labels_read", "metric_computed", "selection_used", "external_submission")},
            trace_plan={k: trace[k] for k in ("optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate", "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity")},
            runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
                "qualification_scope": "fp32-no-tf32-bs128-optimizer-inclusive"})
        write_server_comparison_prelaunch(arm / "comparison_readiness_prelaunch.json",
            comparison_prelaunch=prelaunch, experiment_purpose="training_objective_comparison",
            reference_bundle=bundle, repo_root=REPO_ROOT, reference_bundle_path=REPO_ROOT / REFERENCE)
    now = datetime.now(timezone.utc).isoformat()

    def item(recipe):
        audit = recipe == "audit"
        tid = AUDIT_TRAJECTORY if audit else TRAJECTORIES[recipe]
        sub = "audit" if audit else f"arms/{recipe}"
        cost_id = f"cost-{tid}"
        trajectory = {
            "schema": "molgap-trajectory-v1", "record_mode": "prospective", "owner": "server", "track": "C",
            "trajectory_id": tid, "family_id": "k1-joint-objective",
            "question": "Does all-step Gap supervision with local categorical reconstruction produce a portable gain?",
            "hypothesis": {"hypothesis_id": f"H-{tid}",
                "observed_deficiency": "Relation branches co-adapt; prior sequential reconstruction did not retain Gap supervision every step.",
                "changed_mechanism": "frozen checkpoint NO_TRAIN transfer audit" if audit else recipe,
                "cheapest_falsifier": "one dual-arm seed42 screen, then fixed500K development inference without fitting",
                "expected_native_cost_ref": cost_id, "supporting_evidence_ids": [bundle["reference_id"]],
                "alternative_explanations": ["benefit comes only from corruption", "auxiliary gradients interfere", "reused-role selection optimism"],
                "related_closed_family_ids": ["k1-receiver-relation-resolution", "sequential-global-histogram-reconstruction"],
                "historical_unknowns": ["training stochasticity not measured"],
                "decision_changed_if_positive": "retain a provisional shortlist; scale-up requires new authorization",
                "decision_changed_if_negative": "close the exact objective recipe; no automatic successor"},
            "state_at_start": {"source_commit": commit,
                "source_config_identity": file_digest(root / "source_config.json") if audit else fingerprints[recipe],
                "contract_refs": [f"{REL}/training_contract.json"], "reference_ids": [bundle["reference_id"]],
                "parent_trajectory_ids": list(TRAJECTORIES.values()) if audit else ["TC-k1-v4-100k-reference-s42"],
                "prior_trajectory_ids": [], "prior_evidence_ids": [bundle["reference_id"]],
                "role_snapshot_refs": [f"{RELEASE_REL}/{'audit_role_plan' if audit else 'role_plan'}.json"],
                "budget_snapshot_ref": f"{RELEASE_REL}/budget_snapshot.json"},
            "actions": [{"action_id": "A001", "type": "conditional_NO_TRAIN_audit" if audit else "bounded_100k_objective_training",
                "run_ids": [RUN_ID], "attempt_ids": [f"v{ATTEMPT}"], "source_commit": commit,
                "evidence_refs": [f"{RELEASE_REL}/source_config.json"], "cost_event_ids": [cost_id]}],
            "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {"decision_ref": f"{REL}/protocol.md", "outcome": "ACTIVE",
                "next_allowed_actions": ["frozen checkpoint audit after training completion gate" if audit else "one frozen candidate submission"],
                "reopen_conditions": ["terminal attribution plus explicit new compute decision"]}}
        if not audit:
            trajectory.update(comparison_class="NO_COMPARISON", comparison_blockers=[],
                comparison_readiness_ref=f"{RELEASE_REL}/{sub}/comparison_readiness_prelaunch.json",
                reference_bundle_id=bundle["reference_bundle_id"])
        cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": tid,
            "action_id": "A001", "run_id": RUN_ID, "attempt_id": f"v{ATTEMPT}", "platform": "kaggle2",
            "hardware": "requested_Tesla_T4_actual_pending", "category": "inference" if audit else "training",
            "evidence_ref": f"{RELEASE_REL}/budget_snapshot.json", "measurement": {
                "device_hours": {"value": 1.0 if audit else 3.0, "status": "estimated"},
                "wall_hours": {"value": 0.5 if audit else 3.0, "status": "estimated"},
                "cpu_hours": {"value": None, "status": "measurement_missing"},
                "queue_hours": {"value": None, "status": "measurement_missing"}}}
        return {"output": f"{RELEASE_REL}/{sub}/rml_plan", "spec": {"trajectory": trajectory, "costs": [cost],
            "decision_state": {"available_actions": ["RUN_BOUNDED_JOINT_STUDY", "DEFER"], "chosen_action": "RUN_BOUNDED_JOINT_STUDY",
                "policy_id": POLICY, "policy_version": "v1", "state_timestamp": now}}}

    arms = plan_many(REPO_ROOT, [item(r) for r in RECIPES])
    audit_plan = item("audit")
    audit = plan(REPO_ROOT, audit_plan["spec"], audit_plan["output"])
    return {"source_commit": commit, "arms": arms, "audit": audit}


def package(output):
    from .k1_joint_objective import objective_fingerprint
    from .server_acceptance import validate_server_scientific_prelaunch
    from .v4_bundle import build_v4_source_bundle

    root, output = REPO_ROOT / RELEASE_REL, Path(output).resolve()
    if output.exists():
        raise FileExistsError("Do not overwrite an immutable source package")
    commit = load(root / "source_config.json")["source_commit"]
    if commit != git("rev-parse", "HEAD"):
        raise RuntimeError("Package at the frozen code HEAD, before committing generated plan records")
    bundle = load(REPO_ROOT / REFERENCE)
    release = {"source_commit": commit, "run_id": RUN_ID, "arms": {},
        "audit_trajectory_id": AUDIT_TRAJECTORY, "audit_authority_ref": f"{REL}/protocol.md",
        "target_transform_file_sha256": TRANSFORM_FILE_SHA,
        "reference_bundle_sha256": file_digest(REPO_ROOT / REFERENCE)}
    for recipe in RECIPES:
        path = root / "arms" / recipe / "comparison_readiness_prelaunch.json"
        prelaunch = load(path)
        validate_server_scientific_prelaunch(comparison_prelaunch=prelaunch,
            experiment_purpose="training_objective_comparison", reference_bundle=bundle,
            repo_root=REPO_ROOT, reference_bundle_path=REPO_ROOT / REFERENCE)
        release["arms"][recipe] = {"prelaunch_ready": prelaunch["prelaunch_ready"],
            "prelaunch_sha256": file_digest(path), "trajectory_id": TRAJECTORIES[recipe],
            "loss_identity": objective_fingerprint(recipe),
            "objective_file_sha256": file_digest(path.parent / "objective_config.json")}
    if file_digest(REPO_ROOT / TRANSFORM) != TRANSFORM_FILE_SHA:
        raise RuntimeError("Frozen target transform bytes changed")
    names = [name for name in git("ls-files", "--", "src").splitlines() if name.endswith(".py")]
    if not names or any(not n.startswith("src/molgap/") or not n.endswith(".py") for n in names):
        raise RuntimeError("Source allowlist must contain only tracked molgap Python files")
    result = build_v4_source_bundle(repo_root=REPO_ROOT, relative_paths=names,
        output_dir=output, source_commit=commit, archive_name="source_payload.bin")
    atomic_write(output / "target_transform.json", (REPO_ROOT / TRANSFORM).read_bytes())
    save(output / "JOINT_RELEASE.json", release)
    save(output / "dataset-metadata.json", {"id": DATASET, "title": "MolGap K1 Joint Atom Source",
        "licenses": [{"name": "other"}], "isPrivate": True})
    return result
