"""Declare this question; shared family/workflow/RML owners do the execution."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

from molgap.comparison_readiness import assess_comparison_prelaunch
from molgap.experiment_execution import build_family_recipe
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_pretrained_combo import ADDON_MODES, objective_name, objective_identity
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
REL = EXP.relative_to(ROOT).as_posix()
OLD = ROOT / "experiments/pcqm_k1_dropout_consistency"
RUN = "molgap-k1-pretrain-consistency-pair-100k-s42-v1"
POLICY = "pcqm-k1-pretrained-consistency-teacher-kaggle1-100k-launch"
PRIOR = "pcqm-k1-consistency-fusion-transfer-20261004"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False), encoding="utf-8", newline="")


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    initialization = read(EXP / "initialization_provenance.json")
    template = read(OLD / "experiment_spec_kaggle3_v1.json")
    old_plan = read(OLD / "training_plan_dropout_mean2.json")
    old_accept = read(OLD / "family_acceptance_plan.json")["arms"][0]
    old_prelaunch = read(OLD / "comparison_prelaunch_dropout_mean2.json")
    bundle = read(ROOT / old_accept["reference_bundle"]["path"])
    # Retain exact accepted reference bytes for mechanical row/target binding.
    for pin in old_accept["reference_artifacts"].values():
        destination = ROOT / pin["path"]
        if not destination.is_file():
            source = Path("D:/文档/molgap") / pin["path"]
            if not source.is_file() or sha256_file(source) != pin["sha256"]:
                raise ValueError("Required accepted reference artifact unavailable: " + pin["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if sha256_file(destination) != pin["sha256"]:
            raise ValueError("Reference artifact bytes changed: " + pin["path"])
    policy = read(ROOT / "research_memory/policies/pcqm-k1-dropout-consistency-kaggle3-100k-launch.1.json")
    policy.update(policy_id=POLICY, created_from_source_digest=sha256_file(EXP / "protocol.md"),
                  comparability_selector={"scientific_contract": "k1-pretrained-consistency-teacher-100k-v1"})
    write(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    write(EXP / "role_plan.json", {
        "train": "source_idx[0,100000): supervision, dropout consistency, ArmB-only fixed teacher targets; qualification train-only",
        "development": "source_idx[100000,150000): clean live selection; already selection-used, not fresh",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
        "authority": "Desktop user authorized the proposed new combination pair and specified Kaggle1 on2026-10-05",
    })
    arms, bindings, acceptance, workflow_arms = [], [], [], []
    for device, (addon, mode) in enumerate(ADDON_MODES.items()):
        config = {key: initialization[key] for key in ("initialization_sha256", "pretrained_source_sha256", "head_reset_sha256")}
        config.update(consistency_weight=0.1, teacher_cache=None if device == 0 else {
            "weight": 1.0, "teacher_identity": initialization["teacher_identity"],
            "cache_manifest_sha256": initialization["teacher_cache_manifest_sha256"]})
        recipe = build_family_recipe(("neural_atom_k1", "2"), addon=addon, addon_config=config,
            source_idx_sha256=old_accept["expected"]["source_idx_sha256"], target_sha256=old_accept["expected"]["target_sha256"])
        recipe["runtime_platform_id"] = "kaggle1-t4x2"
        recipe_path = f"{REL}/training_recipe_{mode}.json"
        write(ROOT / recipe_path, recipe)
        arm = copy.deepcopy(template["arms"][0])
        arm.update(arm_id=mode, scientific_role="reference" if device == 0 else "candidate",
                   initialization={"kind": "frozen_state", "seed": 42, "state_sha256": config["initialization_sha256"]},
                   addons=[{"name": addon, "version": "1", "config": config,
                            "source_sha256": normalized_source_sha256(ROOT / "src/molgap/k1_pretrained_combo.py")}])
        arm["training"]["recipe"]["sha256"] = sha256_file(ROOT / recipe_path)
        arm["training"]["objective"] = {"name": objective_name(mode), "version": "1", "sha256": objective_identity(mode, config)}
        arms.append(arm)
        tid = f"TB-k1-{mode.replace('_','-')}-kaggle1-100k-s42-v1"
        run = RUN + ":" + mode + ":downstream"
        plan = copy.deepcopy(old_plan)
        action_path = f"{REL}/action_inputs_{mode}.json"
        write(ROOT / action_path, {"evidence_ids": [PRIOR], "evidence_review_complete": 1,
                                  "state_timestamp": "2026-10-05", "trajectory_id": tid})
        plan["action_inputs_ref"] = action_path
        plan["decision_state"].update(policy_id=POLICY, budget_snapshot_ref=f"{REL}/protocol.md",
            role_snapshot_refs=[f"{REL}/role_plan.json"], source_commit=commit, state_timestamp="2026-10-05")
        cost = plan["costs"][0]
        cost_id = f"cost-{tid}-expected-training"
        cost.update(trajectory_id=tid, cost_event_id=cost_id, attempt_id="kaggle1-pretrained-combo-001", run_id=run,
                    evidence_ref=f"{REL}/protocol.md", hardware=f"Tesla T4; assigned GPU{device} in2T4 pair")
        trajectory = plan["trajectory"]
        trajectory.update(trajectory_id=tid, family_id="k1-pretrained-consistency-teacher",
            question="Does a fixed mean-output teacher improve a pretrained consistency K1 beyond an identical new same-job control?" if device else
                     "What clean Gap endpoint does retained pretraining plus K1 consistency achieve under the fixed100K contract?")
        trajectory["actions"] = [{"action_id": "A001", "attempt_ids": ["kaggle1-pretrained-combo-001"],
            "cost_event_ids": [cost_id], "evidence_refs": [f"{REL}/protocol.md"], "run_ids": [run],
            "source_commit": commit, "type": "authorized_pretrained_consistency_teacher_pair_100k"}]
        trajectory["decision"].update(decision_ref=f"{REL}/protocol.md")
        trajectory["hypothesis"] = {
            "hypothesis_id": "H-" + tid,
            "observed_deficiency": "The retained ensemble improves clean Gap, but prior direct-output single-pass distillation fails compression; pretrained consistency plus a mean-output teacher is untested.",
            "changed_mechanism": "Identical retained10-pass backbone/reset-head initialization; two independent dropout label-L1 forwards with0.1 output consistency" + (" and1.0 fixed teacher MSE on the mean prediction." if device else "; no teacher term."),
            "alternative_explanations": ["Pretraining provides no transferable improvement.", "Teacher constrains already useful output variation or transfers teacher bias.", "Training-seed variation or selection reuse explains numerical changes."],
            "cheapest_falsifier": "One focused synthetic/interface batch and all-arm training-only T4 repeatability/resume qualification; then fixed100K40epoch endpoint.",
            "decision_changed_if_positive": "Review1meV positive-bound teacher increment, compression loss and strict independent RML qualification; no automatic scale-up/adoption.",
            "decision_changed_if_negative": "Close this declared combination without weight/seed/schedule sweep; preserve objective and exposure attribution.",
            "supporting_evidence_ids": [PRIOR], "related_closed_family_ids": ["k1-fusion-distillation", "k1-dropout-consistency", "k1-pretraining"],
            "historical_unknowns": ["Training stochasticity", "Strict historical comparison/pretraining native allocation", "Causal benefit of the shared pretraining initialization"],
            "expected_native_cost_ref": cost_id,
        }
        trajectory["state_at_start"].update(budget_snapshot_ref=f"{REL}/protocol.md",
            contract_refs=[f"{REL}/protocol.md", recipe_path, f"{REL}/evidence_review.md", f"{REL}/initialization_provenance.json"],
            parent_trajectory_ids=["TB-k1-consistency-fusion-transfer-20261004"], prior_evidence_ids=[PRIOR],
            reference_ids=["pcqm-k1-v4-192-kaggle3-reference-custody-20261002"],
            role_snapshot_refs=[f"{REL}/role_plan.json"], source_commit=commit, source_config_identity=canonical_fingerprint(arm))
        plan_path = f"{REL}/training_plan_{mode}.json"
        write(ROOT / plan_path, plan)
        bindings.append({"arm_id": mode, "trajectory_id": tid, "plan_spec_ref": plan_path,
                         "plan_spec_sha256": sha256_file(ROOT / plan_path), "output": f"{REL}/kaggle1_v1/{mode}"})
        identity = copy.deepcopy(bundle["comparison_identity"])
        identity["loss_identity"] = arm["training"]["objective"]["sha256"]
        prelaunch = assess_comparison_prelaunch(candidate_id=RUN + ":" + mode,
            candidate_plan={"comparison_identity": identity, "source_config_status": "frozen", "source_commit_or_archive": commit},
            reference_id=bundle["reference_id"], reference_bundle=bundle, experiment_purpose="training_objective_comparison",
            intervention_group_id="k1-pretrained-combo-objective", declared_intervention_fields=["loss_identity"],
            role_applicability_plan=old_prelaunch["role_applicability_plan"], trace_plan=old_prelaunch["trace_plan"],
            runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
                "qualification_scope": "Kaggle1 assigned T4 train-only BS128; same pretrained state; dropout signal; detached fixed train-only teacher; repeatability/resume/selected-state and <=3xclean-step budget. Historical scope label denotes the retained calibration contract, not current account/qualification. Initialization intervention and historical scientific comparison gaps are declared in protocol; primary comparison is new same-job B versus A."})
        prelaunch_path = f"{REL}/comparison_prelaunch_{mode}.json"
        write(ROOT / prelaunch_path, prelaunch)
        entry = copy.deepcopy(old_accept)
        entry.update(arm_id=mode, expected=recipe["acceptance_requirements"],
            comparison_prelaunch={"path": prelaunch_path, "sha256": sha256_file(ROOT / prelaunch_path)},
            contract={"path": recipe_path, "sha256": sha256_file(ROOT / recipe_path)})
        acceptance.append(entry)
        workflow_arms.append({"arm_id": mode, "device": device, "recipe": recipe_path,
                              "initial_state": initialization["prepared_state_path"]})
    template.update(arms=arms, experiment_id="pcqm-k1-pretrained-consistency-teacher-kaggle1-100k", logical_run_id=RUN,
        prospective={"arms": bindings, "same_run_replay": {"reference_arm_id": arms[0]["arm_id"], "candidate_arm_ids": [arms[1]["arm_id"]]}})
    spec = ExperimentSpec(template)
    write(EXP / "experiment_spec.json", spec.to_dict())
    write(EXP / "family_acceptance_plan.json", {"format": "molgap-family-acceptance-plan-v1", "spec_identity": spec.identity, "arms": acceptance})
    write(EXP / "workflow_plan.json", {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": [], "arms": workflow_arms, "acceptance_plan": f"{REL}/family_acceptance_plan.json",
        "kaggle": {"account": "nothingnessvoid", "kernel": "nothingnessvoid/" + RUN,
            "title": "MolGap K1 Pretrain Consistency Pair 100K S42 V1",
            "datasets": ["nothingnessvoid/molgap-k1-pretrain-consistency-source-s42-v1",
                         "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1", "nothingnessvoid/molgap-k1-fusion-teacher-train100k-v1"],
            "source_dataset": "nothingnessvoid/molgap-k1-pretrain-consistency-source-s42-v1", "accelerator": "NvidiaTeslaT4"}})
    print(json.dumps({"status": "declarations_prepared", "spec_identity": spec.identity, "arms": [a["arm_id"] for a in arms], "account": "nothingnessvoid"}))


if __name__ == "__main__":
    main()
