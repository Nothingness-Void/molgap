"""Freeze a small execution question through existing declaration owners."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from .common.declarations import ExecutionProfile, ExecutionQuestion, build_execution_proposal
from .execution_comparison import freeze_comparison_declarations
from .experiment_family_workflow import _json
from .family_training_runner import read_recipe, family_arm_bindings
from .experiment_execution import build_family_recipe
from .family_workflow_trial import PLAN_FORMAT, EXPERIMENT_ID, INTERFACES
from .research_memory.trace import atomic_write, json_bytes
from .screen_policy import canonical_fingerprint
from .training_reproducibility import sha256_file
from .v4_runtime import normalized_source_sha256

BASE = "experiments/family_workflow_trial_100k"
RUN = "family-workflow-trial-100k-s42-v1"
TEMPLATES = {
    "k1": "experiments/k1_infrastructure_t4_100k/execution_providers",
    "gptrans": "experiments/gptrans_infrastructure_t4_100k/followup/timing",
}


def freeze_trial(root, *, initial_states, gptrans_statistics):
    """No model construction, training, cache rebuilding or platform mutation."""
    root = Path(root).absolute()
    here = root / BASE
    if any((here / n).exists() for n in ("experiment_spec.json", "k1", "gptrans")):
        raise ValueError("Trial declarations already exist; preserve the frozen run")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    authority_ref, role_ref = BASE + "/authority.json", BASE + "/role_plan.json"
    authority = _json(root / authority_ref)
    timestamp = datetime.now(timezone.utc).isoformat()
    profile = ExecutionProfile(owner="server", track="C", platform="kaggle2",
        hardware="Tesla_T4x2_shared_allocation", cheapest_falsifier="one32-step train-only mature-family workflow trial",
        estimated_device_hours=1, estimated_wall_hours=.5)
    question = ExecutionQuestion(policy_id="family-workflow-trial", action="RUN_WORKFLOW_TRIAL",
        question="Can the new front door submit, train, retain and accept mature-family outputs with bounded storage?",
        deficiency="Local preparation alone did not establish remote training or end-to-end acceptance.",
        changed_mechanism="Exercise existing mature models and fixed loader through the new workflow front door.",
        supporting_evidence=("pcqm-k1-infrastructure-optimized-100k-s42-v2",
            "pcqm-gptrans-infrastructure-optimized-100k-s42-v3"),
        alternatives=("remote bootstrap delay", "loader overhead", "output retrieval mismatch"),
        closed_families=("k1-infrastructure-execution", "gptrans-infrastructure-execution"),
        positive_decision="Accept only the bounded workflow execution chain, not model quality.",
        negative_decision="Keep workflow unaccepted; retain exact failure and do not automatically resubmit.")
    arms, bindings, jobs, entries, records = [], [], [], [], []
    for device, (name, interface) in enumerate(INTERFACES.items()):
        template = TEMPLATES[name]
        original, recipe = read_recipe(root / template / "optimized/contract.json")
        recipe = replace(recipe, dataset=replace(recipe.dataset, development=False),
            development_rows=0, optimizer_steps=32, checkpoint_every=8,
            finite_check_every=1, selection="final-live" if name == "k1" else "final-ema")
        if name == "gptrans":
            recipe = replace(recipe, target_mean_eV=gptrans_statistics[0], target_std_eV=gptrans_statistics[1])
        old_spec = _json(root / template / "experiment_spec.json")
        arm = deepcopy(next(a for a in old_spec["arms"] if a["arm_id"] == "optimized"))
        arm.update(arm_id=name, family={"name": interface[0], "version": interface[1]}, scientific_role="reference")
        arm["data"]["roles"] = [r for r in arm["data"]["roles"] if r["role"] == "train"]
        role = arm["data"]["roles"][0]
        role["usage_sha256"] = canonical_fingerprint({"purpose": "WORKFLOW_EXECUTION_TRIAL",
            "membership": role["membership_sha256"], "arm": name, "steps": 32})
        record = build_family_recipe(interface, configuration=recipe, train_role_identity=role["usage_sha256"],
            logging={"mode": "batches", "interval": 1})
        contract_ref = BASE + "/" + name + "/contract.json"
        atomic_write(root / contract_ref, json_bytes(record))
        bound = family_arm_bindings(recipe)
        arm["base"] = bound["base"]
        arm["training"] = {k: bound[k] for k in ("sampler", "transform", "objective")}
        arm["training"].update(recipe={"name": "shared-family-training", "version": "1",
            "sha256": normalized_source_sha256(root / contract_ref)}, overrides={})
        tid = "TC-family-workflow-trial-" + name + "-100k-s42-v1"
        proposal = build_execution_proposal(question=question, arm=arm, trajectory_id=tid,
            source_commit=source, run=RUN, version="v1", contract_ref=contract_ref,
            contract_refs=[contract_ref, BASE + "/protocol.md", authority_ref], authority_ref=authority_ref,
            role_ref=role_ref, decision_ref=BASE + "/decision_prelaunch.md", timestamp=timestamp, profile=profile)
        plan_ref = BASE + "/" + name + "/plan_input.json"
        atomic_write(root / plan_ref, json_bytes(proposal))
        arms.append(arm)
        bindings.append({"arm_id": name, "trajectory_id": tid, "plan_spec_ref": plan_ref,
            "plan_spec_sha256": sha256_file(root / plan_ref), "output": BASE + "/" + name + "/rml"})
        initial = Path(initial_states[name]).absolute()
        if not initial.is_file():
            raise ValueError("Missing mature frozen initialization")
        jobs.append({"arm_id": name, "device": device, "recipe": contract_ref, "initial_state": str(initial)})
        entries.append({"arm_id": name, "adapter": "k1-session-v1" if name == "k1" else "gptrans-session-v1",
            "contract": {"path": contract_ref, "sha256": sha256_file(root / contract_ref)},
            "expected": record["acceptance_requirements"]})
        records.append(plan_ref)
    result = freeze_comparison_declarations(root, base=BASE, experiment_id=EXPERIMENT_ID,
        logical_run_id=RUN, arms=arms, bindings=bindings, jobs=jobs, entries=entries,
        plan_format=PLAN_FORMAT, authority_ref=authority_ref,
        source_extras=records + [authority_ref, role_ref, BASE + "/protocol.md", BASE + "/decision_prelaunch.md"],
        account="kaseichou", kernel="kaseichou/molgap-family-workflow-trial-100k-v1",
        title="MolGap Family Workflow Trial 100K V1",
        source_dataset="kaseichou/molgap-family-workflow-trial-source-v1",
        data_datasets=["kaseichou/pcqm4mv2-ogb-fixed-100k-v1"])
    atomic_write(here / "experiment.json", json_bytes({"format": "molgap-experiment-input-v1",
        "spec_ref": BASE + "/experiment_spec.json", "workflow_ref": BASE + "/workflow_plan.json",
        "machine_profile": "E:/MolGap/machine-profile.json", "authority_ref": authority_ref,
        "monitor_paths": [BASE + "/STATUS.md"], "attempt_id": "v1"}))
    return result
