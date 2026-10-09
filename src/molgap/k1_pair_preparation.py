"""Fixed-data K1 pair draft preparation; no publication, release validation or execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from molgap.evidence_pointers import resolve_repo_pointer
from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_screen_training import validate_recipe, validate_screen_recipe
from molgap.research_memory.policy import validate_policy
from molgap.research_memory.schemas import validate_cost_event, validate_trajectory
from molgap.research_memory.trace import file_digest
from molgap.screen_policy import canonical_fingerprint
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256

REL = "experiments/pcqm_k1_t4_cost_quality"
RUN = "molgap-k1-t4-cost-quality-100k-s42-v1"
POLICY = "pcqm-k1-t4-cost-quality-100k-preparation"
STATE_SHA = "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"
PRIOR = "pcqm-k1-consistency-fusion-transfer-20261004"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_inputs(root: Path, *, source_commit: str, initial_state: Path,
                 state_timestamp: str = "2026-10-09", approved_release: dict | None = None,
                 relative_dir: str = REL, logical_run_id: str = RUN,
                 initialization_seed: int = 42, initialization_sha256: str = STATE_SHA,
                 candidate_addon: str | None = None, allocation_wall_limit_seconds: int = 14400,
                 candidate_arm_id: str = "single", family_id: str = "k1-t4-cost-quality",
                 policy_id: str = POLICY, question: str | None = None,
                 changed_mechanism: str | None = None, supporting_evidence_ids: list[str] | None = None,
                 parent_trajectory_ids: list[str] | None = None,
                 hypothesis_overrides: dict | None = None, experiment_id: str | None = None,
                 source_dataset: str | None = None,
                 candidate_addon_source: str = "src/molgap/k1_screen_training.py",
                 candidate_objective: dict | None = None, _state_inspector=None) -> dict:
    """Build unpublished drafts; parent binds fresh protocol/release before publication.

    Candidate addons are validated by the family registry. Their default objective
    describes a dropout-bearing first view and clean second view, both with BN
    training enabled. Other mechanisms must supply candidate_objective explicitly.
    """
    root = Path(root)
    initial_state = Path(initial_state)
    rel, run = relative_dir, logical_run_id
    fresh = rel != REL or run != RUN
    if (not isinstance(rel, str) or "\\" in rel or ":" in rel
            or not rel.startswith("experiments/") or any(p in ("", ".", "..") for p in rel.split("/"))):
        raise ValueError("relative_dir must be a confined experiment-relative POSIX path")
    if (not isinstance(run, str) or not run or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in run)):
        raise ValueError("logical_run_id must be a fresh lowercase RUN slug")
    if fresh and (rel == REL or run == RUN):
        raise ValueError("Fresh records require both a fresh relative_dir and logical_run_id")
    if type(allocation_wall_limit_seconds) is not int or allocation_wall_limit_seconds <= 60:
        raise ValueError("Allocation wall limit must leave the 60s cleanup reserve")
    if (not isinstance(candidate_arm_id, str) or not candidate_arm_id or candidate_arm_id == "mean2"
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in candidate_arm_id)):
        raise ValueError("candidate_arm_id must be distinct from mean2 and path-safe")
    arm_ids = ("mean2", candidate_arm_id)
    customized = (fresh or initialization_seed != 42 or initialization_sha256 != STATE_SHA
                  or candidate_addon is not None or allocation_wall_limit_seconds != 14400
                  or candidate_arm_id != "single" or family_id != "k1-t4-cost-quality"
                  or policy_id != POLICY or any(value is not None for value in (
                      question, changed_mechanism, supporting_evidence_ids, parent_trajectory_ids,
                      hypothesis_overrides, experiment_id, source_dataset, candidate_objective))
                  or candidate_addon_source != "src/molgap/k1_screen_training.py")
    if customized and not fresh:
        raise ValueError("Changed records require fresh identities; frozen seed42 records cannot be reused")
    if fresh:
        for arm_id in arm_ids:
            if ((root / rel / f"training_plan_{arm_id}.json").exists()
                    or (root / rel / "rml" / arm_id).exists()):
                raise ValueError("Fresh records cannot replace existing plans or RML outputs")
    if approved_release is not None and customized:
        raise ValueError("Fresh records remain drafts; parent must bind the new explicit release after building")
    prior_ids = [PRIOR] if supporting_evidence_ids is None else list(supporting_evidence_ids)
    parents = [] if parent_trajectory_ids is None else list(parent_trajectory_ids)
    release = approved_release
    inspector = inspect_frozen_state_artifact if _state_inspector is None else _state_inspector
    state_report = inspector(initial_state, expected_state_sha256=initialization_sha256)
    template = read(root / "experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json")
    old_recipe = read(root / "experiments/pcqm_k1_fusion_distillation/training_recipe_distill_weak.json")
    expected = old_recipe["acceptance_requirements"]
    protocol = rel + "/protocol.md"
    records, arms, bindings = {}, [], []
    policy = {"schema": "molgap-policy-v1", "policy_id": policy_id, "version": "1",
        "policy_type": "research_action", "status": "draft",
        "comparability_selector": {"scientific_contract": run if fresh else "k1-t4-cost-quality-100k-v1"},
        "required_observable_fields": ["parent_release_validated"],
        "action_rule": {"field": "parent_release_validated", "operator": "eq", "threshold": 1,
                        "action": "REVIEW_TRAIN_PAIR_100K"}, "borderline_action": "NO_TRAIN",
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "cost_model": {"kind": "measured_only", "assumptions": [
            "Count both allocated T4 devices including idle, setup, qualification, evaluation and cleanup.",
            (f"{2 * allocation_wall_limit_seconds / 3600:g} allocated T4 device-hours and "
             f"{allocation_wall_limit_seconds / 3600:g} wall-hours are ceilings, not measured costs.") ]},
        "approval": {"approved_by": None, "approved_at": None, "authority_ref": None},
        "created_from_source_digest": file_digest(root / protocol)}
    validate_policy(policy)
    records["policy.json"] = policy
    records["role_plan.json"] = {"train": [0, 100000], "development": [100000, 150000],
        "dataset": "nvoid912/pcqm4mv2-ogb-fixed-100k-v1", "development_usage": "per-epoch selection and frozen paired endpoint",
        "official_validation": False, "test_dev": False, "test_challenge": False,
        "common_ood_p8_hard": False, "actual_consumption": "pending"}
    for arm_id, role, addon in (("mean2", "reference", "k1_two_pass_mean"), (candidate_arm_id, "candidate", candidate_addon)):
        recipe = build_family_recipe(("neural_atom_k1", "2"), addon=addon,
            source_idx_sha256=expected["source_idx_sha256"], target_sha256=expected["target_sha256"],
            seed=initialization_seed, initialization_sha256=initialization_sha256)
        # Allocation lifetime is a launcher bound, not an optimizer/recipe override.
        recipe["allocation_wall_limit_seconds"] = allocation_wall_limit_seconds
        validate_recipe(recipe, mode=recipe["mode"])
        if recipe["acceptance_requirements"] != expected:
            raise ValueError("Authoritative fixed100K exposure mismatch")
        arm = copy.deepcopy(template)
        arm.update(arm_id=arm_id, scientific_role=role, addon_semantics="ordered" if addon else "baseline")
        arm["initialization"] = {"kind": "frozen_state", "seed": initialization_seed, "state_sha256": initialization_sha256}
        arm["training"]["sampler"]["sha256"] = recipe["row_order_fingerprint"]
        if initialization_seed != 42:
            arm["training"]["sampler"]["name"] = f"seed{initialization_seed}-python-epoch-shuffle-v4"
        addon_source = candidate_addon_source if role == "candidate" else "src/molgap/k1_screen_training.py"
        arm["addons"] = [] if addon is None else [{"name": addon, "version": "1", "config": {},
            "source_sha256": normalized_source_sha256(resolve_repo_pointer(root, addon_source))}]
        arm["training"]["recipe"]["sha256"] = canonical_fingerprint(recipe)
        if addon is not None:
            arm["training"]["objective"]["sha256"] = canonical_fingerprint({
                "base_loss": "normalized-gap-l1", "forward_passes": 2,
                "reduction": "arithmetic-mean-of-two-dropout-bearing-L1-losses",
                "optimizer_updates_per_batch": 1, "consistency_penalty": False,
                "batchnorm_updates_per_batch": 2})
        if role == "candidate" and addon is not None:
            arm["training"]["objective"]["sha256"] = canonical_fingerprint(
                candidate_objective if candidate_objective is not None else {
                    "base_loss": "normalized-gap-l1", "forward_passes": 2,
                    "reduction": "arithmetic-mean-of-two-normalized-L1-losses",
                    "optimizer_updates_per_batch": 1, "consistency_penalty": False,
                    "dropout_enabled_per_forward": [True, False],
                    "batchnorm_training_per_forward": [True, True],
                    "batchnorm_updates_per_batch": 2})
        elif role == "candidate" and candidate_objective is not None:
            arm["training"]["objective"]["sha256"] = canonical_fingerprint(candidate_objective)
        if training_adapter(arm).mode(arm) != recipe["mode"]:
            raise ValueError("Trainer mode/recipe mismatch")
        arms.append(arm)
        records[f"training_recipe_{arm_id}.json"] = recipe
        tid = "TB-" + run + "-" + arm_id if fresh else "TB-k1-t4-cost-quality-" + arm_id + "-100k-s42-v1"
        cost_id = "cost-" + tid + "-planned-allocation"
        cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id,
            "trajectory_id": tid, "action_id": "A001", "run_id": run + ":" + arm_id + (":downstream" if fresh else ""),
            "attempt_id": run + "-v1" if fresh else "kaggle3-cost-quality-001", "platform": "kaggle",
            "hardware": "Tesla T4; one of two allocated devices including idle time",
            "category": "training", "evidence_ref": protocol, "measurement": {
                "device_hours": {"status": "estimated", "value": allocation_wall_limit_seconds / 3600},
                "wall_hours": {"status": "estimated", "value": allocation_wall_limit_seconds / 3600},
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "queue_hours": {"status": "measurement_missing", "value": None}}}
        validate_cost_event(cost)
        trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": tid,
            "record_mode": "prospective", "track": "B", "owner": "desktop",
            "family_id": family_id, "question": question if question is not None else "Can one forward replace mean2 within frozen cost-quality tolerance?",
            "hypothesis": {"hypothesis_id": "H-" + tid,
                "observed_deficiency": "Native T4 single versus mean2 cost and quality remain unqualified.",
                "supporting_evidence_ids": list(prior_ids),
                "alternative_explanations": ["Dropout and BN update count affect quality.", "One-seed optimization noise affects endpoints.", "Non-step overhead limits allocation savings."],
                "changed_mechanism": changed_mechanism if changed_mechanism is not None else (
                    "Equal normalized L1 losses from dropout-bearing first and Dropout-off clean second views; BN training stays enabled with two updates; no consistency penalty."
                    if candidate_addon is not None else
                    "One normalized Gap L1 forward versus arithmetic mean of two dropout-bearing L1 forwards; no consistency penalty."),
                "cheapest_falsifier": "Parent-validated native T4 profile and completed local diagnostic first; no hidden urgent fitting failure; then bounded paired100K only if separately released.",
                "related_closed_family_ids": ["k1-dropout-consistency", "k1-fusion-distillation"],
                "expected_native_cost_ref": cost_id,
                "decision_changed_if_positive": "Cost-quality screen nomination only; no500K/full/promotion.",
                "decision_changed_if_negative": "Close with attribution; no posthoc tuning or successor.",
                "historical_unknowns": ["Training stochasticity", "Historical mean2 complete native cost", "Actual paired runtime/artifacts"]},
            "state_at_start": {"source_commit": source_commit, "source_config_identity": canonical_fingerprint(arm),
                "contract_refs": [protocol, rel + "/evidence_review.md", rel + f"/training_recipe_{arm_id}.json"],
                "reference_ids": [], "prior_evidence_ids": list(prior_ids), "parent_trajectory_ids": list(parents),
                "role_snapshot_refs": [rel + "/role_plan.json"], "budget_snapshot_ref": protocol},
            "actions": [{"action_id": "A001", "type": "conditional_pair_100k_after_parent_release",
                "source_commit": source_commit, "run_ids": [cost["run_id"]], "attempt_ids": [cost["attempt_id"]],
                "evidence_refs": [protocol], "cost_event_ids": [cost_id]}],
            "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {"outcome": "ACTIVE", "decision_ref": protocol,
                         "next_allowed_actions": [], "reopen_conditions": []}}
        if hypothesis_overrides is not None:
            if set(hypothesis_overrides) - set(trajectory["hypothesis"]):
                raise ValueError("Unknown hypothesis override fields")
            trajectory["hypothesis"].update(copy.deepcopy(hypothesis_overrides))
        validate_trajectory(trajectory)
        plan = {"trajectory": trajectory, "costs": [cost], "decision_state": {
            "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
            "available_actions": ["PREPARE_ONLY", "NO_TRAIN", "REVIEW_TRAIN_PAIR_100K"], "chosen_action": "PREPARE_ONLY",
            "policy_id": policy_id, "policy_version": "1", "role_snapshot_refs": [rel + "/role_plan.json"],
            "budget_snapshot_ref": protocol, "state_timestamp": state_timestamp, "source_commit": source_commit}}
        records[f"training_plan_{arm_id}.json"] = plan
        bindings.append({"arm_id": arm_id, "trajectory_id": tid,
            "plan_spec_ref": rel + f"/training_plan_{arm_id}.json", "plan_spec_sha256": canonical_fingerprint(plan),
            "output": rel + ("/rml/" if fresh else "/kaggle3_v1/") + arm_id})
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2",
        "experiment_id": experiment_id if experiment_id is not None else (run if fresh else "pcqm-k1-t4-cost-quality-100k"), "logical_run_id": run, "arms": arms,
        "platform": {"name": "kaggle", "accelerator": "Tesla T4", "device_count": 2,
            "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings, "same_run_replay": {"reference_arm_id": "mean2", "candidate_arm_ids": [candidate_arm_id]}},
        "evidence": {"policy": {"name": "molgap-v5", "version": "1",
            "sha256": normalized_source_sha256(root / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")},
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": "molgap-experiment-terminal-descriptor-v1"})
    records["experiment_spec.json"] = spec.to_dict()
    for arm in arms:
        validate_screen_recipe(spec, arm["arm_id"], records[f"training_recipe_{arm['arm_id']}.json"])
    # Legacy target bytes supply encoding context, never accepted reference authority.
    target_ref = "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json"
    target_manifest = read(root / target_ref)
    if target_manifest["development_target_sha256"] != expected["target_sha256"]:
        raise ValueError("Authoritative target manifest/recipe mismatch")
    target_pointer = {"path": target_ref, "sha256": file_digest(root / target_ref)}
    acceptance = []
    for arm in arms:
        acceptance.append({"arm_id": arm["arm_id"], "adapter": "k1-screen-v1",
            "expected": copy.deepcopy(expected),
            "contract": {"path": rel + f"/training_recipe_{arm['arm_id']}.json",
                         "sha256": arm["training"]["recipe"]["sha256"]},
            "target_manifest": copy.deepcopy(target_pointer)})
    records["family_acceptance_plan.json"] = {"format": "molgap-family-same-run-acceptance-plan-v1",
        "spec_identity": spec.identity, "arms": acceptance}
    source = source_dataset if source_dataset is not None else (
        "nvoid912/" + run + "-source" if fresh else "nvoid912/molgap-k1-t4-cost-quality-source-s42-v1")
    records["workflow_plan.json"] = {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": [], "arms": [{"arm_id": a["arm_id"], "device": i,
            "recipe": rel + f"/training_recipe_{a['arm_id']}.json", "initial_state": initial_state.as_posix()}
            for i, a in enumerate(arms)], "acceptance_plan": rel + "/family_acceptance_plan.json",
        "kaggle": {"account": "nvoid912", "kernel": "nvoid912/" + run, "title": run,
            "datasets": [source, "nvoid912/pcqm4mv2-ogb-fixed-100k-v1"], "source_dataset": source, "accelerator": "NvidiaTeslaT4"}}
    records["preparation_report.json"] = {"status": "LOCAL_DRAFT_ONLY", "initial_state": state_report,
        "source_inventory_owner": "molgap.experiment_source_inventory.SHARED_SOURCE_FILES",
        "shared_source_count": len(SHARED_SOURCE_FILES), "source_commit": source_commit,
        "training_authorized": False, "prospective_published": False, "strict_ready": False,
        "blockers": ["Parent must commit executable source and rebind plans before prepare-workflow.",
            "Parent must validate passed native T4 profile and completed local diagnostic/no urgent fitting failure.",
            "Release API pending: raw profile completion is not acceptance; profile-acceptance and clean-fit decision schemas/validator are required for a separate release-bound prospective plan.",
            f"Parent must verify recipe-bound {allocation_wall_limit_seconds}s allocation timeout and 60s cleanup reserve, durable complete-epoch recovery/output custody.",
            "Register explicit local policy through RML owner before prospective publication.",
            "Parent must review mean2 objective/addon functional binding in committed executable source before prepare-workflow.",
            "Strict runtime, paired artifacts, observed allocation costs and same-run accepted reference remain pending."]}
    if fresh:
        records["preparation_report.json"]["blockers"] = [
            "Parent must commit executable source and rebind plans before prepare-workflow.",
            "Parent must review the fresh protocol, evidence review and hypothesis before publication.",
            "Parent must bind the new explicit user release after building; legacy parent release is not applicable.",
            f"Parent must verify recipe-bound {allocation_wall_limit_seconds}s allocation timeout and 60s cleanup reserve, durable complete-epoch recovery/output custody.",
            "Register explicit local policy through RML owner before prospective publication.",
            "Strict runtime, paired artifacts, observed allocation costs and fresh same-run accepted reference remain pending."]
    if release is not None:
        binding = release["binding"]
        policy["action_rule"]["action"] = "TRAIN_PAIR_100K"
        policy["status"] = "approved"
        policy["approval"] = {"approved_by": release["record"]["approved_by"],
            "approved_at": release["record"]["approved_at"], "authority_ref": binding["path"]}
        validate_policy(policy)
        for arm_id in arm_ids:
            plan = records[f"training_plan_{arm_id}.json"]
            trajectory = plan["trajectory"]
            trajectory["state_at_start"]["contract_refs"].append(binding["path"])
            trajectory["state_at_start"]["budget_snapshot_ref"] = binding["path"]
            trajectory["actions"][0]["type"] = "paired100k_after_parentrelease"
            trajectory["actions"][0]["evidence_refs"].append(binding["path"])
            trajectory["decision"].update(decision_ref=binding["path"], next_allowed_actions=["A001"])
            plan["decision_state"].update(available_actions=["NO_TRAIN", "TRAIN_PAIR_100K"],
                chosen_action="TRAIN_PAIR_100K", budget_snapshot_ref=binding["path"])
            plan["parent_release"] = copy.deepcopy(binding)
            validate_trajectory(trajectory)
        payload = records["experiment_spec.json"]
        for arm in payload["prospective"]["arms"]:
            arm["plan_spec_sha256"] = canonical_fingerprint(records[f"training_plan_{arm['arm_id']}.json"])
        spec = ExperimentSpec(payload)
        records["experiment_spec.json"] = spec.to_dict()
        for name in ("family_acceptance_plan.json", "workflow_plan.json"):
            records[name]["spec_identity"] = spec.identity
        report = records["preparation_report.json"]
        report.update(status="LOCAL_RELEASE_BOUND_UNPUBLISHED", parent_release=binding,
            parent_release_validated=True, chosen_action="TRAIN_PAIR_100K")
        report["blockers"] = [b for b in report["blockers"]
            if not b.startswith(("Parent must validate passed", "Release API pending:"))]
    return records
