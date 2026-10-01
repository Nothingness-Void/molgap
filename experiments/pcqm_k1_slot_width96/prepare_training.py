"""Prepare one registered candidate's local inputs; publish via experiment_cli."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

from molgap.constants import EXPERIMENTS_DIR, REPO_ROOT
from molgap.comparison_readiness import assess_comparison_prelaunch, validate_comparison_prelaunch
from molgap.experiment_execution import build_family_recipe
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_slot_width96 import CONFIG, EXPECTED_PARAMETER_COUNT, REFERENCE_PARAMETER_COUNT
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256
from molgap.v5_common import validate_v5_evidence_envelope

HERE = EXPERIMENTS_DIR / "pcqm_k1_slot_width96"
REL = HERE.relative_to(REPO_ROOT).as_posix()
TID = "TB-k1-slot-width96-kaggle3-100k-s42-v1"
RUN = "molgap-k1-slot96-100k-s42-v1"
POLICY = "pcqm-k1-slot-width96-kaggle3-100k-launch"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def pointer(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(REPO_ROOT.resolve()).as_posix(), "sha256": sha256_file(path)}


def qualification():
    """Inspect retained qualification outputs and state; never construct a model."""
    report = read(HERE / "qualification_result.json")
    ablation = read(HERE / "ablation_result.json")
    frozen = read(HERE / "cpu_frozen.json")
    if ((HERE / "cpu_failure.json").exists() or
        report.get("status") != "CPU_INITIALIZATION_QUALIFIED" or
        ablation.get("status") != "CPU_ABLATION_COMPLETE" or
        report.get("candidate_parameters") != EXPECTED_PARAMETER_COUNT or
        report.get("reference_parameters") != REFERENCE_PARAMETER_COUNT or
        any(report.get(key) is not True for key in
            ("finite", "saved_state_roundtrip", "real_pure2d_forward_finite",
             "reference_default_matches_explicit_latent64")) or
        any(ablation.get(key) is not True for key in
            ("finite", "target_source_exact", "all_weights_preserved", "layer3_layer6_preserved",
             "proceed_to_slot_capacity_screen"))):
        raise ValueError("CPU qualification/utility evidence is incomplete")
    utility = ablation["zero9_minus_original_mae_eV"]
    reconstruction = ablation["original_replicate_max_delta_eV"]
    if (not math.isfinite(utility) or utility < 0.001 or not math.isfinite(reconstruction)
        or not 0 <= reconstruction <= 1e-4 or ablation["gate_eV"] != 0.001):
        raise ValueError("Frozen utility or reconstruction gate failed")
    if ablation["rows_sha256"] != sha256_file(HERE / "ablation_rows.npz"):
        raise ValueError("CPU paired-row artifact changed")
    for relative, digest in frozen["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != digest:
            raise ValueError(f"CPU frozen source changed: {relative}")
    if frozen["trajectory_sha256"] != sha256_file(HERE / "qualification/trajectory.json"):
        raise ValueError("CPU prospective trajectory changed")
    state = HERE / "qualification_initial_state.pt"
    if sha256_file(state) != report["initial_state_file_sha256"]:
        raise ValueError("Qualified initial-state file changed")
    inspect_frozen_state_artifact(state, expected_state_sha256=report["initial_state_sha256"])
    return report, ablation, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-evidence", type=Path, required=True,
                        help="Parent accepted CPU finalized v5_evidence.json within this checkout")
    parser.add_argument("--source-commit", required=True,
                        help="Reviewed immutable source commit used by the eventual package")
    parser.add_argument("--state-timestamp", default="2026-10-02")
    parser.add_argument("--refresh-source", action="store_true",
                        help="Rebind unpublished sidecars after source/recipe commit; preserve recipe bytes")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if args.source_commit != commit:
        raise ValueError("Source commit must equal this owner checkout HEAD")
    report, ablation, state = qualification()
    cpu_evidence = read(args.cpu_evidence)
    cpu_ref = pointer(args.cpu_evidence)["path"]
    validate_v5_evidence_envelope(cpu_evidence, repo_root=REPO_ROOT)
    cpu_trajectory = read(args.cpu_evidence.parent / "trajectory.json")
    if (cpu_trajectory["trajectory_id"] != report["trajectory_id"] or
        cpu_evidence["evidence_id"] not in cpu_trajectory["result"]["evidence_ids"]):
        raise ValueError("Accepted CPU evidence belongs to another trajectory")
    cpu_eid = cpu_evidence["evidence_id"]
    custody = read(HERE / "reference_binding/custody_manifest.json")
    bundle = read(HERE / "reference_binding/reference_bundle.json")
    expected = read(HERE / "reference_binding/expected.json")
    reference_bindings = read(HERE / "reference_binding/reference_artifact_bindings.json")
    recipe = build_family_recipe(("neural_atom_k1", "2"), addon="k1_slot_width96",
        source_idx_sha256=expected["source_idx_sha256"], target_sha256=expected["target_sha256"],
        initialization_sha256=report["initial_state_sha256"])
    if recipe["acceptance_requirements"] != expected:
        raise ValueError("Candidate exposure differs from retained reference")
    arm = copy.deepcopy(read(HERE / "reference_binding/original_reference_arm.json"))
    arm.update(arm_id="slot96", scientific_role="candidate", addon_semantics="ordered")
    arm["initialization"] = {"kind": "frozen_state", "seed": 42,
                             "state_sha256": report["initial_state_sha256"]}
    arm["addons"] = [{"name": "k1_slot_width96", "version": "1", "config": CONFIG,
        "source_sha256": normalized_source_sha256(REPO_ROOT / "src/molgap/k1_slot_width96.py")}]
    arm["training"]["recipe"]["sha256"] = hashlib.sha256(canonical(recipe)).hexdigest()
    identity = dict(bundle["comparison_identity"])
    identity["architecture_config_identity"] = canonical_fingerprint({"base": arm["base"], "addons": arm["addons"]})
    roles = {key: "applicable" for key in
             ("training_membership", "prediction_input", "labels_read", "metric_computed", "selection_used")}
    roles["external_submission"] = "not_applicable"
    trace = {key: True for key in ("optimizer_step", "sample_presentations", "epoch_or_pass",
                                  "learning_rate", "live_train_metric", "live_dev_metric", "checkpoint_identity")}
    trace["ema_dev_metric"] = False
    comparison = assess_comparison_prelaunch(candidate_id=RUN, candidate_plan={
        "comparison_identity": identity, "source_config_status": "frozen", "source_commit_or_archive": commit},
        reference_id=bundle["reference_id"], reference_bundle=bundle,
        experiment_purpose="architecture_comparison", intervention_group_id="k1-slot-width",
        declared_intervention_fields=["architecture_config_identity"], role_applicability_plan=roles,
        trace_plan=trace, runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
            "qualification_scope": "slot96 actual assigned T4 tuple; finite gradients, repeatability and resume"})
    validate_comparison_prelaunch(comparison)
    if not comparison["prelaunch_ready"]:
        raise ValueError(f"Retained comparison plan blocked: {comparison['blocker_codes']}")
    refs = [cpu_eid, custody["reference_evidence_id"], "pcqm-k1-slot-readout-diagnostic-20261002"]
    protocol = f"{REL}/protocol.md"
    cost_id = f"cost-{TID}-expected-training"
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": cost_id,
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN + ":slot96:downstream",
        "attempt_id": "kaggle3-slot96-001", "platform": "kaggle", "hardware": "Tesla T4 x2 allocation; candidate assigned GPU0",
        "category": "training", "evidence_ref": protocol, "measurement": {
            "wall_hours": {"value": 4.0, "status": "estimated"},
            "device_hours": {"value": 8.0, "status": "estimated"},
            "cpu_hours": {"value": None, "status": "measurement_missing"},
            "queue_hours": {"value": None, "status": "measurement_missing"}}}
    policy = {"schema": "molgap-policy-v1", "policy_id": POLICY, "version": "1",
        "policy_type": "research_action", "status": "candidate",
        "comparability_selector": {"scientific_contract": "k1-slot-width96-100k-v1"},
        "required_observable_fields": ["cpu_utility_gate_passed"],
        "action_rule": {"field": "cpu_utility_gate_passed", "operator": "eq", "threshold": 1,
                        "action": "TRAIN_SINGLE_100K"}, "borderline_action": "NO_TRAIN",
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "cost_model": {"kind": "measured_only", "assumptions": []},
        "approval": {"approved_by": None, "approved_at": None, "authority_ref": None},
        "created_from_source_digest": sha256_file(HERE / "protocol.md")}
    action_inputs = {"trajectory_id": TID, "state_timestamp": args.state_timestamp,
                     "evidence_ids": refs, "cpu_utility_gate_passed": 1}
    trajectory = {"schema": "molgap-trajectory-v1", "trajectory_id": TID, "record_mode": "prospective",
        "track": "B", "owner": "desktop", "family_id": "k1-slot-width96",
        "question": "Does isolated K1 latent64-to96 capacity improve matched fixed100K Gap prediction?",
        "hypothesis": {"hypothesis_id": "H-" + TID,
            "observed_deficiency": "Node widening regressed; last-slot return is substantial and retained zero-slot utility clears its gate, but isolated slot capacity remains unmeasured.",
            "supporting_evidence_ids": refs,
            "alternative_explanations": ["Slot64 already suffices.", "Extra capacity can worsen generalization.", "Single-seed optimization and runtime differences can explain endpoints."],
            "changed_mechanism": "Only latent slot64 becomes96; atom192 edge64 layers9 active-slot1 mean pooling RWSE16 are fixed.",
            "cheapest_falsifier": "Accepted CPU utility/construction, required T4 qualification, then one fixed40epoch candidate; reuse reference without retraining.",
            "related_closed_family_ids": ["k1-slot-readout-diagnostic", "k1-node-width", "k1-full-convergence"],
            "expected_native_cost_ref": cost_id,
            "decision_changed_if_positive": "Review paired50K accuracy, native costs and strict qualification; no automatic scale-up or adoption.",
            "decision_changed_if_negative": "Retain terminal evidence and failure attribution; no successor.",
            "historical_unknowns": ["Training stochasticity is unavailable from a single seed."]},
        "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(arm),
            "contract_refs": [protocol, f"{REL}/training_recipe.json", f"{REL}/evidence_review.md", cpu_ref],
            "reference_ids": [bundle["reference_id"]], "prior_evidence_ids": refs,
            "parent_trajectory_ids": [report["trajectory_id"]],
            "role_snapshot_refs": [f"{REL}/role_plan.json"], "budget_snapshot_ref": protocol},
        "actions": [{"action_id": "A001", "type": "authorized_single_candidate_100k_training",
            "source_commit": commit, "run_ids": [cost["run_id"]], "attempt_ids": [cost["attempt_id"]],
            "evidence_refs": [protocol], "cost_event_ids": [cost_id]}],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {"outcome": "ACTIVE", "decision_ref": protocol,
                     "next_allowed_actions": ["A001"], "reopen_conditions": []}}
    training_plan = {"trajectory": trajectory, "costs": [cost],
        "action_inputs_ref": f"{REL}/training_action_inputs.json", "decision_state": {
            "known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [],
            "available_actions": ["TRAIN_SINGLE_100K", "NO_TRAIN"], "chosen_action": "TRAIN_SINGLE_100K",
            "policy_id": POLICY, "policy_version": "1", "role_snapshot_refs": [f"{REL}/role_plan.json"],
            "budget_snapshot_ref": protocol, "state_timestamp": args.state_timestamp, "source_commit": commit}}
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2",
        "experiment_id": "pcqm-k1-slot-width96-kaggle3-100k", "logical_run_id": RUN, "arms": [arm],
        "platform": {"name": "kaggle", "accelerator": "Tesla T4", "device_count": 2,
                     "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": [{"arm_id": "slot96", "trajectory_id": TID,
            "plan_spec_ref": f"{REL}/training_plan_kaggle3_v1.json",
            "plan_spec_sha256": hashlib.sha256(canonical(training_plan)).hexdigest(),
            "output": f"{REL}/kaggle3_v1/slot96"}]},
        "evidence": {"policy": {"name": "molgap-v5", "version": "1",
            "sha256": normalized_source_sha256(REPO_ROOT / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")},
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": "molgap-experiment-terminal-descriptor-v1"})
    generated_pointer = lambda name, record: {"path": f"{REL}/{name}",
        "sha256": hashlib.sha256(canonical(record)).hexdigest()}
    acceptance = {"format": "molgap-family-acceptance-plan-v1", "spec_identity": spec.identity,
        "arms": [{"arm_id": "slot96", "adapter": "k1-screen-v1", "expected": expected,
            "contract": generated_pointer("training_recipe.json", recipe),
            "comparison_prelaunch": generated_pointer("comparison_prelaunch.json", comparison),
            "reference_bundle": pointer(HERE / "reference_binding/reference_bundle.json"),
            "reference_artifacts": reference_bindings}]}
    source = "nvoid912/molgap-k1-slot96-source-s42-v1"
    workflow = {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": [], "arms": [{"arm_id": "slot96", "device": 0,
            "recipe": f"{REL}/training_recipe.json", "initial_state": str(state.resolve())}],
        "acceptance_plan": f"{REL}/family_acceptance_plan.json", "kaggle": {
            "account": "nvoid912", "kernel": "nvoid912/" + RUN, "title": RUN,
            "datasets": [source, "nvoid912/pcqm4mv2-ogb-fixed-100k-v1"],
            "source_dataset": source, "accelerator": "NvidiaTeslaT4"}}
    outputs = {HERE / "training_recipe.json": recipe, HERE / "candidate_arm.json": arm,
        HERE / "comparison_prelaunch.json": comparison, HERE / "training_action_inputs.json": action_inputs,
        HERE / "training_plan_kaggle3_v1.json": training_plan,
        HERE / "experiment_spec_kaggle3_v1.json": spec.to_dict(),
        HERE / "family_acceptance_plan.json": acceptance, HERE / "workflow_plan_kaggle3_v1.json": workflow,
        REPO_ROOT / "research_memory/policies" / (POLICY + ".1.json"): policy}
    if (HERE / "kaggle3_v1/slot96").exists():
        raise FileExistsError("Prospective already published; source refresh requires reconciliation")
    existing = [path.exists() for path in outputs]
    if any(existing):
        if not args.refresh_source or not all(existing):
            raise FileExistsError("Configuration outputs exist; use explicit refresh only for a complete unpublished set")
        for name in ("training_recipe.json", "candidate_arm.json", "training_action_inputs.json"):
            if (HERE / name).read_bytes() != canonical(outputs[HERE / name]):
                raise ValueError("Source refresh may not change frozen recipe, candidate or action inputs")
        policy_path = REPO_ROOT / "research_memory/policies" / (POLICY + ".1.json")
        if policy_path.read_bytes() != canonical(policy):
            raise ValueError("Source refresh may not change the policy")
    elif args.refresh_source:
        raise ValueError("Source refresh requires a complete existing unpublished configuration")
    for path, record in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(record))
    print(json.dumps({"status": "CONFIGS_PREPARED_UNPUBLISHED", "spec_identity": spec.identity,
        "source_commit": commit, "utility_delta_eV": ablation["zero9_minus_original_mae_eV"],
        "prospective_published": False, "submitted": False}))


if __name__ == "__main__":
    main()
