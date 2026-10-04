"""Reuse RML planning and source staging for train-only execution qualification."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_package import _name
from molgap.experiment_staging import stage_release_inputs, UploadArtifact
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file
from molgap.v4_runtime import normalized_source_sha256
from molgap.pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256

BASE = "experiments/pcqm_gptrans_scale_qualification"


def freeze_profile():
    root = REPO_ROOT
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    contract = json.loads((root / BASE / "contract.json").read_text())
    initial = root / "platforms/_records/kaggle/training/gptrans_author_inputs_verification_recovery_v2/gptrans_author_inputs/degree_initial_state.pt"
    if sha256_file(initial) != contract["initial_file_sha256"]:
        raise ValueError("Accepted G1 initialization identity differs from the profile contract")
    prior = json.loads((root / "experiments/pcqm_gptrans_readout_100k/gpu/spec.json").read_text())
    arm = deepcopy(prior["arms"][0])
    arm.update(arm_id="scale_profile", scientific_role="ablation", family={"name": "gptrans_scale_profile", "version": "1"}, addons=[], addon_semantics="baseline")
    arm["data"]["dataset"]["sha256"] = FIXED_500K_MANIFEST_SHA256
    arm["data"]["dataset"]["name"] = "pcqm4mv2"
    arm["data"]["split"] = {"name": "fixed500k-train-only-profile", "version": "1", "sha256": FIXED_500K_MANIFEST_SHA256}
    arm["data"]["roles"] = [{"role": "train", "membership_sha256": canonical_fingerprint({"start": 0, "stop": 500000}),
        "row_order_sha256": canonical_fingerprint({"source_order": [0,500000]}), "usage_sha256": sha256_file(root / BASE / "contract.json")}]
    arm["training"]["recipe"] = {"name": "gptrans_scale_profile_v1", "version": "1", "sha256": normalized_source_sha256(root / BASE / "contract.json")}
    arm["training"]["sampler"] = {"name": "seed42-step-fixed500k-profiling-v1", "version": "1", "sha256": canonical_fingerprint({"seed": 42, "rows": 500000, "steps": 35, "physical_batch": 128})}
    tid = "TC-gptrans-g1-scale500k-qualification-s42"
    source = json.loads((root / "experiments/pcqm_gptrans_ema_portability/attempt_v4/rml_plan/rml_finalized/trajectory.json").read_text())
    snapshot = json.loads((root / "experiments/pcqm_gptrans_readout_100k/gpu/degree_node_mean_readout_ema999/plan_input.json").read_text())
    trajectory = snapshot["trajectory"]
    trajectory.update(trajectory_id=tid, family_id="gptrans-g1-scale-qualification", question="Can G1 and two EMA filters fit the equal-exposure500K runtime budget?")
    trajectory["hypothesis"].update(hypothesis_id="H-"+tid, observed_deficiency="Scale trainer/runtime eligibility is unqualified.",
        changed_mechanism="Execution-only disposable calibration, no trained candidate",
        cheapest_falsifier="37train-role-only disposable optimizer updates", expected_native_cost_ref="cost-"+tid,
        supporting_evidence_ids=source["result"]["evidence_ids"], alternative_explanations=["Loader and real validation overhead can invalidate an optimistic proxy"],
        decision_changed_if_positive="Plan scale adapter; no automatic long training release", decision_changed_if_negative="Do not release500K training", related_closed_family_ids=["gptrans-final-readout"], historical_unknowns=["True dev distribution/runtime and checkpoint overhead remain unmeasured"])
    trajectory["state_at_start"].update(source_commit=commit, source_config_identity=canonical_fingerprint(arm),
        contract_refs=[BASE+"/contract.json",BASE+"/protocol.md"], reference_ids=[], parent_trajectory_ids=[source["trajectory_id"]],
        prior_trajectory_ids=[source["trajectory_id"]], prior_evidence_ids=source["result"]["evidence_ids"],
        role_snapshot_refs=[BASE+"/contract.json"], budget_snapshot_ref=BASE+"/protocol.md")
    trajectory["actions"] = [{"action_id":"A001", "type":"NO_TRAIN_profiling", "source_commit":commit,
        "run_ids":["gptrans-g1-scale-qualification-s42"], "attempt_ids":["v1"], "evidence_refs":[BASE+"/contract.json"], "cost_event_ids":["cost-"+tid]}]
    trajectory["result"] = {"evidence_ids":[],"evidence_refs":[]}
    trajectory["decision"].update(decision_ref=BASE+"/protocol.md", outcome="ACTIVE", next_allowed_actions=["Bounded qualification then controller analysis"], reopen_conditions=["Completed output or actionable execution fault"])
    snapshot["decision_state"].update(active_reference_ids=[], available_actions=["PROFILE_G1_SCALE","DEFER"], chosen_action="PROFILE_G1_SCALE",
        policy_id="gptrans-g1-followup", policy_version="v1", budget_snapshot_ref=BASE+"/protocol.md", role_snapshot_refs=[BASE+"/contract.json"], source_commit=commit, state_timestamp=datetime.now(timezone.utc).isoformat())
    snapshot["costs"] = [{"schema":"molgap-cost-event-v1","cost_event_id":"cost-"+tid,"trajectory_id":tid,"action_id":"A001","run_id":"gptrans-g1-scale-qualification-s42","attempt_id":"v1","category":"preflight","platform":"kaggle2","hardware":"T4-allocation-one-visible-worker","evidence_ref":BASE+"/protocol.md","measurement":{"wall_hours":{"status":"estimated","value":.75},"device_hours":{"status":"estimated","value":1.5},"cpu_hours":{"status":"measurement_missing","value":None},"queue_hours":{"status":"measurement_missing","value":None}}}]
    atomic_json(root/BASE/"plan_input.json", snapshot)
    declaration = deepcopy(prior)
    declaration.update(experiment_id="gptrans-g1-scale-qualification", logical_run_id="gptrans-g1-scale-qualification-s42", arms=[arm],
        prospective={"arms":[{"arm_id":"scale_profile","trajectory_id":tid,"plan_spec_ref":BASE+"/plan_input.json","plan_spec_sha256":sha256_file(root/BASE/"plan_input.json"),"output":BASE+"/rml_plan"}]})
    spec = ExperimentSpec(declaration)
    spec.write(root/BASE/"spec.json")
    result = plan(root, snapshot, BASE+"/rml_plan")
    return spec, result


def prepare(output: Path):
    root = REPO_ROOT
    if (root / BASE / "rml_plan/trajectory.json").is_file():
        spec = ExperimentSpec.from_json((root / BASE / "spec.json").read_text())
        result = {"status": "FROZEN_PLAN_REUSED", "path": BASE + "/rml_plan", "compute_released": False}
    else:
        spec, result = freeze_profile()
    initial = root / "platforms/_records/kaggle/training/gptrans_author_inputs_verification_recovery_v2/gptrans_author_inputs/degree_initial_state.pt"
    sources = subprocess.check_output(["git","ls-files","src/molgap"],text=True).splitlines()
    allowed = []
    for p in sources:
        if p.endswith(".py") and "/archive/" not in p:
            try:
                allowed.append(_name(p))
            except ValueError:
                continue
    sources = allowed
    sources += [BASE+"/contract.json",BASE+"/run.py","platforms/kaggle/bootstrap_gptrans_profile.py",BASE+"/kernel-metadata.json"]
    staged = stage_release_inputs(spec, root, sources, output,
        artifacts={"degree_initial_state.pt":UploadArtifact.from_file(initial),
            "target_transform.json":UploadArtifact.from_file(root/"experiments/pcqm_gptrans_author_alignment/recovered_reference/target_transform.json")},
        recipe_files={"scale_profile":BASE+"/contract.json"},initial_states={"scale_profile":"degree_initial_state.pt"},
        required_modules=["molgap.gptrans_scale_profile","molgap.kaggle_python_environment"],
        entry_template=root/"platforms/kaggle/bootstrap_gptrans_profile.py",kernel_metadata=root/BASE/"kernel-metadata.json",
        dataset_metadata={"title":"MolGap GPTrans G1 Scale Profile Source","id":"kaseichou/molgap-gptrans-g1-scale-profile-source","licenses":[{"name":"other"}],"isPrivate":True})
    binding = root / BASE / "release_binding.json"
    if binding.exists():
        atomic_json(root / BASE / "preparation_failed_binding_v1.json", json.loads(binding.read_text()))
    atomic_json(binding, {"prospective":result,"staging":staged,"release_root":str(output)})
    return staged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().output),indent=2))
