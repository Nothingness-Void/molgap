"""Prospective planning and saved-artifact acceptance for relation screens."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess

from .constants import REPO_ROOT
from .k1_relation_study_runtime import SLOTS, RUNS, TRAJECTORIES
from .research_memory.trace import atomic_write, json_bytes, file_digest, load_canonical_trace

REL = "experiments/pcqm_k1_relation_resolution_100k"
ROOT = REPO_ROOT / REL
TEMPLATE = REPO_ROOT / "experiments/pcqm_k1_sparse_triplet_100k"
REFERENCE = REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
POLICY = "k1-relation-resolution-bounded-research"
AUDIT_RUN = "kaseichou/molgap-k1-relation-audit-s42:v1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    atomic_write(path, json_bytes(value))


def freeze():
    from .comparison_readiness import assess_comparison_prelaunch
    from .server_acceptance import write_server_comparison_prelaunch
    from .pcqm_k1_variants import ARCHITECTURE_CONFIGS
    from .research_memory.plan import plan
    from .screen_policy import canonical_fingerprint
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", REL],
                               cwd=REPO_ROOT, text=True).strip():
        raise RuntimeError("Commit source and protocol before freezing")
    if (ROOT / "source_config.json").exists():
        raise FileExistsError("Do not overwrite a frozen study")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    now = datetime.now(timezone.utc).isoformat()
    bundle = load(REFERENCE)
    role, trace = load(TEMPLATE / "role_plan.json"), load(TEMPLATE / "trace_plan.json")
    save(ROOT / "role_plan.json", role)
    save(ROOT / "trace_plan.json", trace)
    audit_role = dict(role, training_membership="not_applicable", selection_used="not_applicable")
    save(ROOT / "audit_role_plan.json", audit_role)
    configs = {mode: canonical_fingerprint(ARCHITECTURE_CONFIGS[mode]) for mode in TRAJECTORIES}
    save(ROOT / "source_config.json", {
        "format": "molgap-frozen-source-config-v1", "source_commit": commit,
        "candidate_ids": list(TRAJECTORIES), "architecture_config_identities": configs,
        "training_contract_ref": f"{REL}/training_contract.json",
        "training_contract_sha256": file_digest(ROOT / "training_contract.json"),
        "implementation_refs": ["src/molgap/k1_relation_resolution.py",
            "src/molgap/k1_relation_study_runtime.py", "src/molgap/pcqm_k1_variants_runner.py"],
    })
    save(ROOT / "budget_snapshot.json", {
        "available_pool": "user-authorized-Kaggle2-two-notebook-slots",
        "available_device_hours": None, "estimated_training_device_hours_per_arm": 3,
        "max_training_wall_hours_per_worker": 6, "max_audit_device_hours": 1.5,
        "maximum_training_active_device_hours": 18,
        "bootstrap_overhead_and_idle_allocations_recorded_separately": True,
        "automatic_training_successor_authorized": False,
    })
    save(REPO_ROOT / f"research_memory/policies/{POLICY}.v1.json", {
        "schema": "molgap-policy-v1", "policy_id": POLICY, "version": "v1",
        "policy_type": "research_action", "status": "candidate",
        "comparability_selector": {"scientific_contract": "pcqm4mv2-ogb-fixed-100k-gap-v4-s42-fp32-bs128-40epochs"},
        "required_observable_fields": ["distinct_receiver_information_question_frozen"],
        "created_from_source_digest": file_digest(ROOT / "protocol.md"),
        "approval": {"authority_ref": f"{REL}/protocol.md", "activation": "controller-only-bounded-user-authorization"},
        "cost_model": {"kind": "measured_only", "assumptions": ["no cross-hardware scalar conversion"]},
        "observation_point": None, "promotion_rule": None, "early_stop_rule": None,
        "action_rule": {"field": "distinct_receiver_information_question_frozen", "operator": "eq", "threshold": True,
                        "action": "RUN_BOUNDED_RELATION_STUDY"}, "borderline_action": "DEFER",
    })
    plans = []
    for mode in (*TRAJECTORIES, "audit"):
        audit = mode == "audit"
        if audit:
            sub = "audit"
            tid = "TC-k1-relation-resolution-post100k-audit-s42"
            config, run = canonical_fingerprint(configs), AUDIT_RUN
        else:
            sub = "arms/" + mode.removeprefix("neural_atom_k1_")
            tid, config = TRAJECTORIES[mode], configs[mode]
            slot = next(slot for slot, modes in SLOTS.items() if mode in modes)
            run = RUNS[slot]
            identity = dict(bundle["comparison_identity"], architecture_config_identity=config)
            prelaunch = assess_comparison_prelaunch(
                candidate_id=mode, reference_id=bundle["reference_id"], reference_bundle=bundle,
                candidate_plan={"comparison_identity": identity, "source_config_status": "frozen", "source_commit_or_archive": commit},
                experiment_purpose="architecture_comparison", intervention_group_id="architecture",
                declared_intervention_fields=["architecture_config_identity"],
                role_applicability_plan={k: role[k] for k in ("training_membership", "prediction_input", "labels_read", "metric_computed", "selection_used", "external_submission")},
                trace_plan={k: trace[k] for k in ("optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate", "live_train_metric", "live_dev_metric", "ema_dev_metric", "checkpoint_identity")},
                runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
                    "qualification_scope": "fp32-no-tf32-bs128-optimizer-inclusive"})
            write_server_comparison_prelaunch(ROOT / sub / "comparison_readiness_prelaunch.json",
                comparison_prelaunch=prelaunch, experiment_purpose="architecture_comparison",
                reference_bundle=bundle, repo_root=REPO_ROOT, reference_bundle_path=REFERENCE)
        cost_id = f"cost-{tid}"
        trajectory = copy.deepcopy(load(TEMPLATE / "trajectory.json"))
        trajectory.update(trajectory_id=tid, family_id="k1-receiver-relation-resolution",
                          question="Does receiver-resolved relation information improve portable 2D regression?")
        trajectory["hypothesis"].update(
            hypothesis_id=f"H-{tid}", expected_native_cost_ref=cost_id,
            observed_deficiency="Global PairToken pooling discards receiver identity; previous gates, SPD and linear exchange did not retain late/portable gains.",
            supporting_evidence_ids=[bundle["reference_id"]],
            alternative_explanations=["receiver messages overfit", "pair-to-pair signal is redundant", "reused-role selection optimism"],
            changed_mechanism="accepted-checkpoint NO_TRAIN portability audit" if audit else ARCHITECTURE_CONFIGS[mode]["change"],
            cheapest_falsifier="one frozen seed42 candidate and separate accepted-checkpoint 500K internal-dev inference",
            decision_changed_if_positive="retain shortlist pending explicitly authorized scale study",
            decision_changed_if_negative="close exact mechanism without automatic successor")
        role_ref = f"{REL}/{'audit_role_plan' if audit else 'role_plan'}.json"
        trajectory["state_at_start"] = {
            "source_commit": commit, "source_config_identity": config,
            "contract_refs": [f"{REL}/training_contract.json"], "reference_ids": [bundle["reference_id"]],
            "parent_trajectory_ids": list(TRAJECTORIES.values()) if audit else ["TC-k1-v4-100k-reference-s42"],
            "prior_trajectory_ids": [], "prior_evidence_ids": [bundle["reference_id"]],
            "role_snapshot_refs": [role_ref], "budget_snapshot_ref": f"{REL}/budget_snapshot.json"}
        trajectory["actions"] = [{"action_id": "A001", "type": "conditional_NO_TRAIN_audit" if audit else "bounded_100k_training",
            "run_ids": [run], "attempt_ids": ["v1"], "source_commit": commit,
            "evidence_refs": [f"{REL}/source_config.json"], "cost_event_ids": [cost_id]}]
        trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
        trajectory["decision"] = {"decision_ref": f"{REL}/protocol.md", "outcome": "ACTIVE",
            "next_allowed_actions": ["accepted training then one separate NO_TRAIN audit" if audit else "one frozen candidate submission"],
            "reopen_conditions": ["terminal attribution and explicit new compute decision"]}
        for key in ("comparison_readiness_ref", "comparison_class", "comparison_blockers", "reference_bundle_id"):
            trajectory.pop(key, None)
        if not audit:
            trajectory.update(comparison_readiness_ref=f"{REL}/{sub}/comparison_readiness_prelaunch.json",
                comparison_class="NO_COMPARISON", comparison_blockers=[], reference_bundle_id=bundle["reference_bundle_id"])
        cost = copy.deepcopy(load(TEMPLATE / "costs/expected_training.json"))
        cost.update(cost_event_id=cost_id, trajectory_id=tid, run_id=run, attempt_id="v1", platform="kaggle2",
            hardware="actual_GPU_pending", category="inference" if audit else "training", evidence_ref=f"{REL}/budget_snapshot.json")
        for unit in ("device_hours", "wall_hours"):
            cost["measurement"][unit] = {"value": 1.0 if audit else 3.0, "status": "estimated"}
        plans.append(plan(REPO_ROOT, {"trajectory": trajectory, "costs": [cost], "decision_state": {
            "available_actions": ["RUN_BOUNDED_RELATION_STUDY", "DEFER"], "chosen_action": "RUN_BOUNDED_RELATION_STUDY",
            "policy_id": POLICY, "policy_version": "v1", "state_timestamp": now}}, f"{REL}/{sub}/rml_plan"))
    return {"source_commit": commit, "plans": plans}


def validate_mechanism_evidence(mode, checks):
    """Require the checks emitted by the frozen remote preflight, not aliases."""
    required = {"zero_return_projection", "zero_update", "valid_row_assignment_sum_one",
        "padding_assignment_zero", "finite", "different_receiver_features_within_graph",
        "graph0_perturbation_isolated", "receiver_specific_return_features",
        "permutation_equivariant_before_return", "resume_two_step_bitwise_equal"}
    if mode.endswith("rrwp_pair"):
        required.update({"rrwp_batched_graphs_match_independent_expected",
                         "rrwp_isolate_self_transition"})
    if mode.endswith("triplet_aggregate"):
        required.add("tgt_vector_loop_agreement")
    missing = sorted(key for key in required if checks.get(key) is not True)
    if missing:
        raise ValueError(f"Missing or failed remote mechanism checks: {missing}")
    if checks.get("largest_train_shape_probe", {}).get("graphs") != 128:
        raise ValueError("Missing full-batch worst-shape memory qualification")


def accept_training(reference_root, candidate_root, source_commit, archive_sha256, slot):
    """Verify retained tensors and telemetry; never instantiate an encoder."""
    from .k1_relation_resolution import PARAMETERS
    candidate_root, reference_root = Path(candidate_root), Path(reference_root)
    frozen = load(ROOT / "source_config.json")
    if frozen["source_commit"] != source_commit:
        raise ValueError("Supplied scientific source differs from prospective freeze")
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_relation_shared_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    result = shared.accept(reference_root, candidate_root, modes=SLOTS[slot],
                           expected_parameters=PARAMETERS, initialization_policy="nested-function")
    for mode in SLOTS[slot]:
        arm = candidate_root / mode
        record = result["candidates"][mode]["record"]
        canonical = load_canonical_trace(arm / "canonical_trace.json")
        raw = load(arm / "trace.json")["epochs"]
        roles, cost = load(arm / "observed_role_history.json"), load(arm / "native_cost.json")
        if (canonical["trajectory_id"] != TRAJECTORIES[mode] or canonical["run_id"] != RUNS[slot]
            or len(canonical["observations"]) != 40 or len(raw) != 40
            or canonical["observations"][-1]["event"] != "terminal"
            or canonical["observations"][-1]["optimizer_step"] != 31240
            or canonical["observations"][-1]["sample_presentations"] != 3998720
            or canonical["observations"][-1]["checkpoint_identity"] != file_digest(arm / "last_checkpoint.pt")):
            raise ValueError("Native canonical trace/run/checkpoint identity incomplete")
        for observed, native in zip(canonical["observations"], raw):
            if (observed["live_dev_metric"] != native["development_gap_mae_eV"]
                or observed["live_train_metric"] != native["train_normalized_mae"]
                or observed["optimizer_step"] != native["optimizer_steps"]
                or observed["sample_presentations"] != native["sample_presentations"]):
                raise ValueError("Canonical/native trace mismatch")
        for key in ("training_labels_read", "development_labels_read", "development_metric_computed", "development_selection_used"):
            if roles.get(key) is not True:
                raise ValueError("Missing observed role use")
        wall_seconds = cost.get("wall_seconds")
        if (roles.get("run_id") != RUNS[slot] or cost.get("run_id") != RUNS[slot]
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds) or wall_seconds <= 0
            or cost.get("training_completed") is not True
            or record["source_commit"] != source_commit
            or record["contract"]["source_archive_sha256"] != archive_sha256):
            raise ValueError("Native identity, source or cost incomplete")
        validate_mechanism_evidence(mode, record["preflight"]["mechanism_checks"])
        if any(roles.get(flag) is not False for flag in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
            raise ValueError("Protected role access cannot be accepted")
    result.update(source_commit=source_commit, source_archive_sha256=archive_sha256,
                  slot=slot, post100k_audit_accepted=False, full_training_authorized=False)
    return result
