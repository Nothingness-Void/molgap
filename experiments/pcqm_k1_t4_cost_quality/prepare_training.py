"""Validate or stage local prospective inputs. Never package, publish, or train."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT as ROOT
from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.evidence_pointers import load_json_object, resolve_repo_pointer, verify_bound_artifact
from molgap.experiment_spec import ExperimentSpec, _canonical
from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
from molgap.k1_screen_training import validate_recipe, validate_screen_recipe
from molgap.research_memory.schemas import validate_cost_event, validate_trajectory
from molgap.research_memory.policy import validate_policy
from molgap.research_memory.trace import atomic_write, file_digest
from molgap.screen_policy import canonical_fingerprint
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256

REL = "experiments/pcqm_k1_t4_cost_quality"
RUN = "molgap-k1-t4-cost-quality-100k-s42-v1"
POLICY = "pcqm-k1-t4-cost-quality-100k-preparation"
STATE = Path("D:/w/k1-dropout-consistency/experiments/pcqm_k1_dropout_consistency/initial_state.pt")
STATE_SHA = "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"
PRIOR = "pcqm-k1-consistency-fusion-transfer-20261004"
RELEASE_FORMAT = "molgap-k1-t4-parent-release-v1"
CLEAN = "experiments/pcqm_k1_clean_fit_generalization_500k"


def _bound(root: Path, binding: dict) -> Path:
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or any(not isinstance(binding[key], str) or not binding[key] for key in ("path", "sha256"))):
        raise ValueError("Evidence requires a repository-local path and SHA pin")
    verify_bound_artifact(root, binding["path"], binding["sha256"])
    return resolve_repo_pointer(root, binding["path"])


def _validate_profile_release(root: Path, acceptance_path: Path) -> dict:
    """Read-only acceptance owner, never profiler completion or a copied validator."""
    path = root / REL / "profile/close.py"
    if not path.is_file():
        raise ValueError("Parent-coordinated profile.close acceptance callable is unavailable")
    module_spec = importlib.util.spec_from_file_location("k1_native_profile_close", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    validator = getattr(module, "accept", None)
    if not callable(validator):
        raise ValueError("profile.close.accept is unavailable")
    if acceptance_path != (root / REL / "profile/acceptance.json").resolve():
        raise ValueError("Profile pin must name the owning accepted record, not raw completion")
    report = validator(root)
    receipt = load_json_object(acceptance_path)
    if (not isinstance(report, dict) or not isinstance(report.get("analysis"), dict)
            or not isinstance(receipt.get("analysis"), dict)
            or not isinstance(receipt.get("outcome"), dict)
            or "single_step_saving_fraction" not in report["analysis"]):
        raise ValueError("Profile acceptance requires structured owner analysis/outcome, not booleans")
    if (receipt.get("analysis") != report["analysis"]
            or receipt.get("outcome", {}).get("artifact_status") != "local_hash_verified"
            or receipt.get("outcome", {}).get("scientific_status") != "NO_TRAIN"):
        raise ValueError("Pinned profile acceptance differs from actual validated artifacts")
    return {"status": "ACCEPTED",
            "native_step_reduction": report["analysis"]["single_step_saving_fraction"]}


def validate_parent_release(root: Path, release_file: Path, *, source_commit: str) -> dict:
    """Validate parent judgement and custody, not a new clean-fit scientific gate."""
    root = Path(root).resolve()
    path = Path(release_file)
    path = path if path.is_absolute() else root / path
    try:
        ref = path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Parent release must be repository-local") from exc
    if not ref.startswith(REL + "/") or "/profile/" in ref:
        raise ValueError("Parent release must belong to the owning question outside profile")
    path = resolve_repo_pointer(root, ref)
    release = load_json_object(path)
    if release.get("format") != RELEASE_FORMAT:
        raise ValueError("Expected explicit parent release record, not raw completion")
    if (release.get("controller") != "human-controller"
            or not isinstance(release.get("approved_by"), str) or not release["approved_by"].strip()
            or not isinstance(release.get("approved_at"), str) or not release["approved_at"].strip()):
        raise ValueError("Explicit human-controller approval identity/date required")
    if (release.get("action") != "TRAIN_PAIR_100K"
            or release.get("allowed_actions") != ["TRAIN_PAIR_100K"]):
        raise ValueError("Parent release permits TRAIN_PAIR_100K only")
    if release.get("source_commit") != source_commit:
        raise ValueError("Parent release source commit mismatch")
    budget = release.get("budget")
    if (not isinstance(budget, dict) or set(budget) != {"allocated_t4_device_hours", "wall_seconds"}
            or type(budget.get("allocated_t4_device_hours")) not in (int, float)
            or type(budget.get("wall_seconds")) not in (int, float)
            or budget != {"allocated_t4_device_hours": 8, "wall_seconds": 14400}):
        raise ValueError("Parent release requires exactly 8 allocated T4 device-hours/14400 wall seconds")
    profile_path = _bound(root, release.get("profile_acceptance"))
    clean = release.get("clean_fit")
    if not isinstance(clean, dict):
        raise ValueError("Missing clean-fit pins and manual assessment")
    paths = {key: _bound(root, clean.get(key)) for key in ("terminal", "acceptance", "decision", "finalization")}
    for key, name in (("terminal", "rml/rml_finalized/terminal_input.json"), ("acceptance", "acceptance.json"),
                      ("decision", "terminal_decision.md"), ("finalization", "closure_receipt.json")):
        if paths[key] != (root / CLEAN / name).resolve():
            raise ValueError("Clean-fit pins must name the integrated canonical records")
    terminal = load_json_object(paths["terminal"])
    receipt = load_json_object(paths["finalization"])
    acceptance = load_json_object(paths["acceptance"])
    if (not isinstance(terminal.get("decision"), dict)
            or not isinstance(terminal.get("artifact_hashes"), dict)
            or not isinstance(receipt.get("input_artifact_hashes"), dict)
            or not isinstance(receipt.get("published_hashes"), dict)
            or not isinstance(acceptance.get("outcome"), dict)):
        raise ValueError("Clean-fit requires structured finalized records, not booleans")
    if (receipt.get("format") != "molgap-rml-finalization-v1"
            or not receipt.get("finalization_id") or not receipt.get("finalized_at")
            or receipt.get("outcome") != "NO_TRAIN"
            or terminal.get("format") != "molgap-rml-terminal-package-v1"
            or receipt.get("trajectory_id") != terminal.get("trajectory_id")
            or terminal.get("decision", {}).get("outcome") != "NO_TRAIN"
            or terminal.get("decision", {}).get("decision_ref") != clean["decision"]["path"]
            or terminal.get("acceptance_ref") != clean["acceptance"]["path"]
            or acceptance.get("format") != "molgap-local-frozen-diagnostic-acceptance-v1"
            or acceptance.get("outcome", {}).get("execution_status") != "complete_no_training"
            or acceptance.get("outcome", {}).get("artifact_status") != "local_hash_verified"
            or acceptance.get("outcome", {}).get("scientific_status") != "NO_TRAIN"):
        raise ValueError("Clean-fit requires an actual finalized NO_TRAIN accepted receipt")
    for key in ("acceptance", "decision"):
        pin = clean[key]
        if (receipt.get("input_artifact_hashes", {}).get(pin["path"]) != pin["sha256"]
                or terminal.get("artifact_hashes", {}).get(pin["path"]) != pin["sha256"]):
            raise ValueError("Clean-fit finalization/terminal does not pin accepted decision")
    if receipt.get("published_hashes", {}).get("terminal_input.json") != clean["terminal"]["sha256"]:
        raise ValueError("Clean-fit finalization does not bind terminal input")
    if (clean.get("assessment") != "NO_URGENT_FITTING_FAILURE"
            or not isinstance(clean.get("rationale"), str) or not clean["rationale"].strip()):
        raise ValueError("Manual assessment explaining no urgent fitting failure required")
    report = _validate_profile_release(root, profile_path)
    if not isinstance(report, dict) or report.get("status") != "ACCEPTED":
        raise ValueError("Native profile acceptance owner did not validate artifacts")
    reduction = report.get("native_step_reduction")
    if type(reduction) not in (int, float) or not math.isfinite(reduction) or not .25 <= reduction <= 1:
        raise ValueError("Accepted native profile reduction must be >=0.25")
    binding = {"path": ref, "sha256": file_digest(path)}
    verify_bound_artifact(root, ref, binding["sha256"])
    return {"binding": binding, "record": release, "profile_report": report}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_inputs(root: Path, *, source_commit: str, initial_state: Path,
                 state_timestamp: str = "2026-10-09", parent_release_file: Path | None = None) -> dict:
    """Reuse authoritative identities; generated records remain unpublished drafts."""
    root = Path(root)
    release = None
    if parent_release_file is not None:
        # Never promote existing canonical or staged plans in place.
        for arm_id in ("mean2", "single"):
            if ((root / REL / f"training_plan_{arm_id}.json").exists()
                    or (root / REL / "kaggle3_v1" / arm_id).exists()):
                raise ValueError("Release plans require an unpublished fresh build")
        release = validate_parent_release(root, parent_release_file, source_commit=source_commit)
    state_report = inspect_frozen_state_artifact(initial_state, expected_state_sha256=STATE_SHA)
    template = read(root / "experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json")
    old_recipe = read(root / "experiments/pcqm_k1_fusion_distillation/training_recipe_distill_weak.json")
    expected = old_recipe["acceptance_requirements"]
    protocol = REL + "/protocol.md"
    records, arms, bindings = {}, [], []
    policy = {"schema": "molgap-policy-v1", "policy_id": POLICY, "version": "1",
        "policy_type": "research_action", "status": "draft",
        "comparability_selector": {"scientific_contract": "k1-t4-cost-quality-100k-v1"},
        "required_observable_fields": ["parent_release_validated"],
        "action_rule": {"field": "parent_release_validated", "operator": "eq", "threshold": 1,
                        "action": "REVIEW_TRAIN_PAIR_100K"}, "borderline_action": "NO_TRAIN",
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "cost_model": {"kind": "measured_only", "assumptions": [
            "Count both allocated T4 devices including idle, setup, qualification, evaluation and cleanup.",
            "8 allocated T4 device-hours and 4 wall-hours are ceilings, not measured costs."]},
        "approval": {"approved_by": None, "approved_at": None, "authority_ref": None},
        "created_from_source_digest": file_digest(root / protocol)}
    validate_policy(policy)
    records["policy.json"] = policy
    records["role_plan.json"] = {"train": [0, 100000], "development": [100000, 150000],
        "dataset": "nvoid912/pcqm4mv2-ogb-fixed-100k-v1", "development_usage": "per-epoch selection and frozen paired endpoint",
        "official_validation": False, "test_dev": False, "test_challenge": False,
        "common_ood_p8_hard": False, "actual_consumption": "pending"}
    for arm_id, role, addon in (("mean2", "reference", "k1_two_pass_mean"), ("single", "candidate", None)):
        recipe = build_family_recipe(("neural_atom_k1", "2"), addon=addon,
            source_idx_sha256=expected["source_idx_sha256"], target_sha256=expected["target_sha256"])
        # Allocation lifetime is a launcher bound, not an optimizer/recipe override.
        recipe["allocation_wall_limit_seconds"] = 14400
        validate_recipe(recipe, mode=recipe["mode"])
        if recipe["acceptance_requirements"] != expected:
            raise ValueError("Authoritative fixed100K exposure mismatch")
        arm = copy.deepcopy(template)
        arm.update(arm_id=arm_id, scientific_role=role, addon_semantics="ordered" if addon else "baseline")
        arm["initialization"] = {"kind": "frozen_state", "seed": 42, "state_sha256": STATE_SHA}
        arm["addons"] = [] if addon is None else [{"name": addon, "version": "1", "config": {},
            "source_sha256": normalized_source_sha256(root / "src/molgap/k1_screen_training.py")}]
        arm["training"]["recipe"]["sha256"] = canonical_fingerprint(recipe)
        if addon is not None:
            arm["training"]["objective"]["sha256"] = canonical_fingerprint({
                "base_loss": "normalized-gap-l1", "forward_passes": 2,
                "reduction": "arithmetic-mean-of-two-dropout-bearing-L1-losses",
                "optimizer_updates_per_batch": 1, "consistency_penalty": False,
                "batchnorm_updates_per_batch": 2})
        if training_adapter(arm).mode(arm) != recipe["mode"]:
            raise ValueError("Trainer mode/recipe mismatch")
        arms.append(arm)
        records[f"training_recipe_{arm_id}.json"] = recipe
        tid = "TB-k1-t4-cost-quality-" + arm_id + "-100k-s42-v1"
        cost_id = "cost-" + tid + "-planned-allocation"
        cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id,
            "trajectory_id": tid, "action_id": "A001", "run_id": RUN + ":" + arm_id,
            "attempt_id": "kaggle3-cost-quality-001", "platform": "kaggle",
            "hardware": "Tesla T4; one of two allocated devices including idle time",
            "category": "training", "evidence_ref": protocol, "measurement": {
                "device_hours": {"status": "estimated", "value": 4.0},
                "wall_hours": {"status": "estimated", "value": 4.0},
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "queue_hours": {"status": "measurement_missing", "value": None}}}
        validate_cost_event(cost)
        trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": tid,
            "record_mode": "prospective", "track": "B", "owner": "desktop",
            "family_id": "k1-t4-cost-quality", "question": "Can one forward replace mean2 within frozen cost-quality tolerance?",
            "hypothesis": {"hypothesis_id": "H-" + tid,
                "observed_deficiency": "Native T4 single versus mean2 cost and quality remain unqualified.",
                "supporting_evidence_ids": [PRIOR],
                "alternative_explanations": ["Dropout and BN update count affect quality.", "One-seed optimization noise affects endpoints.", "Non-step overhead limits allocation savings."],
                "changed_mechanism": "One normalized Gap L1 forward versus arithmetic mean of two dropout-bearing L1 forwards; no consistency penalty.",
                "cheapest_falsifier": "Parent-validated native T4 profile and completed local diagnostic first; no hidden urgent fitting failure; then bounded paired100K only if separately released.",
                "related_closed_family_ids": ["k1-dropout-consistency", "k1-fusion-distillation"],
                "expected_native_cost_ref": cost_id,
                "decision_changed_if_positive": "Cost-quality screen nomination only; no500K/full/promotion.",
                "decision_changed_if_negative": "Close with attribution; no posthoc tuning or successor.",
                "historical_unknowns": ["Training stochasticity", "Historical mean2 complete native cost", "Actual paired runtime/artifacts"]},
            "state_at_start": {"source_commit": source_commit, "source_config_identity": canonical_fingerprint(arm),
                "contract_refs": [protocol, REL + "/evidence_review.md", REL + f"/training_recipe_{arm_id}.json"],
                "reference_ids": [], "prior_evidence_ids": [PRIOR], "parent_trajectory_ids": [],
                "role_snapshot_refs": [REL + "/role_plan.json"], "budget_snapshot_ref": protocol},
            "actions": [{"action_id": "A001", "type": "conditional_pair_100k_after_parent_release",
                "source_commit": source_commit, "run_ids": [cost["run_id"]], "attempt_ids": [cost["attempt_id"]],
                "evidence_refs": [protocol], "cost_event_ids": [cost_id]}],
            "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {"outcome": "ACTIVE", "decision_ref": protocol,
                         "next_allowed_actions": [], "reopen_conditions": []}}
        validate_trajectory(trajectory)
        plan = {"trajectory": trajectory, "costs": [cost], "decision_state": {
            "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
            "available_actions": ["PREPARE_ONLY", "NO_TRAIN", "REVIEW_TRAIN_PAIR_100K"], "chosen_action": "PREPARE_ONLY",
            "policy_id": POLICY, "policy_version": "1", "role_snapshot_refs": [REL + "/role_plan.json"],
            "budget_snapshot_ref": protocol, "state_timestamp": state_timestamp, "source_commit": source_commit}}
        records[f"training_plan_{arm_id}.json"] = plan
        bindings.append({"arm_id": arm_id, "trajectory_id": tid,
            "plan_spec_ref": REL + f"/training_plan_{arm_id}.json", "plan_spec_sha256": canonical_fingerprint(plan),
            "output": REL + "/kaggle3_v1/" + arm_id})
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2",
        "experiment_id": "pcqm-k1-t4-cost-quality-100k", "logical_run_id": RUN, "arms": arms,
        "platform": {"name": "kaggle", "accelerator": "Tesla T4", "device_count": 2,
            "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings, "same_run_replay": {"reference_arm_id": "mean2", "candidate_arm_ids": ["single"]}},
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
            "contract": {"path": REL + f"/training_recipe_{arm['arm_id']}.json",
                         "sha256": arm["training"]["recipe"]["sha256"]},
            "target_manifest": copy.deepcopy(target_pointer)})
    records["family_acceptance_plan.json"] = {"format": "molgap-family-same-run-acceptance-plan-v1",
        "spec_identity": spec.identity, "arms": acceptance}
    source = "nvoid912/molgap-k1-t4-cost-quality-source-s42-v1"
    records["workflow_plan.json"] = {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": [], "arms": [{"arm_id": a["arm_id"], "device": i,
            "recipe": REL + f"/training_recipe_{a['arm_id']}.json", "initial_state": initial_state.as_posix()}
            for i, a in enumerate(arms)], "acceptance_plan": REL + "/family_acceptance_plan.json",
        "kaggle": {"account": "nvoid912", "kernel": "nvoid912/" + RUN, "title": RUN,
            "datasets": [source, "nvoid912/pcqm4mv2-ogb-fixed-100k-v1"], "source_dataset": source, "accelerator": "NvidiaTeslaT4"}}
    records["preparation_report.json"] = {"status": "LOCAL_DRAFT_ONLY", "initial_state": state_report,
        "source_inventory_owner": "molgap.experiment_source_inventory.SHARED_SOURCE_FILES",
        "shared_source_count": len(SHARED_SOURCE_FILES), "source_commit": source_commit,
        "training_authorized": False, "prospective_published": False, "strict_ready": False,
        "blockers": ["Parent must commit executable source and rebind plans before prepare-workflow.",
            "Parent must validate passed native T4 profile and completed local diagnostic/no urgent fitting failure.",
            "Release API pending: raw profile completion is not acceptance; profile-acceptance and clean-fit decision schemas/validator are required for a separate release-bound prospective plan.",
            "Parent must verify recipe-bound 14400s allocation timeout and 60s cleanup reserve, durable complete-epoch recovery/output custody.",
            "Register explicit local policy through RML owner before prospective publication.",
            "Parent must review mean2 objective/addon functional binding in committed executable source before prepare-workflow.",
            "Strict runtime, paired artifacts, observed allocation costs and same-run accepted reference remain pending."]}
    if release is not None:
        binding = release["binding"]
        policy["action_rule"]["action"] = "TRAIN_PAIR_100K"
        policy["status"] = "approved"
        policy["approval"] = {"approved_by": release["record"]["approved_by"],
            "approved_at": release["record"]["approved_at"], "authority_ref": binding["path"]}
        validate_policy(policy)
        for arm_id in ("mean2", "single"):
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


def stage_inputs(records: dict, output: Path) -> None:
    """Fresh draft directory only; no canonical trajectory publication or RML writes."""
    if output.exists():
        raise FileExistsError("Reconcile existing drafts; output must be fresh")
    output.mkdir(parents=True)
    for name, record in records.items():
        atomic_write(output / name, _canonical(record).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-local", type=Path, help="Fresh draft output; absent means validation only")
    parser.add_argument("--initial-state", type=Path, default=STATE)
    parser.add_argument("--parent-release-file", type=Path, help="Repository-local explicit human-controller release")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    records = build_inputs(ROOT, source_commit=commit, initial_state=args.initial_state,
                           parent_release_file=args.parent_release_file)
    if args.stage_local:
        output = args.stage_local.resolve()
        if not output.is_relative_to((ROOT / REL).resolve()) or "profile" in output.relative_to(ROOT / REL).parts:
            raise ValueError("Draft output must remain in the owning question, outside profile")
        stage_inputs(records, output)
    print(json.dumps(records["preparation_report.json"], sort_keys=True))


if __name__ == "__main__":
    main()
