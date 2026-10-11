"""Question declarations composed with the existing prospective/package workflow."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .comparison_readiness import ROLE_EVENT_KINDS, TRACE_FIELD_DECLARATIONS, assess_comparison_prelaunch
from .experiment_spec import ExperimentSpec
from .experiment_staging import UploadArtifact
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, sha256_file
from .v4_runtime import normalized_source_sha256

BASE = "experiments/pcqm_gptrans_execution40"
OLD = "experiments/pcqm_gptrans_capacity_relations_100k"
RUN = "gptrans-execution40-dual-s42"
ARMS = ("execution_original", "execution_optimized")


def freeze(root):
    root = Path(root).resolve()
    read = lambda p: json.loads((root / p).read_text())
    previous = ExperimentSpec.from_json((root / OLD / "gpu/spec.json").read_text()).to_dict()
    model = "degree_bond_local_ema999"
    prior_arm = next(a for a in previous["arms"] if a["arm_id"] == model)
    prior_config = read(OLD + "/gpu/screen_config.json")
    facts = prior_config["arms"][model]
    initial = root / "platforms/_records/kaggle/initializations" / facts["initial_file"]
    if sha256_file(initial) != facts["initial_file_sha256"]:
        raise ValueError("Retained shared initialization changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    policy = read("research_memory/policies/gptrans-g1-followup.v1.json")
    policy.update(policy_id="gptrans-execution40", comparability_selector={"scientific_contract":"gptrans-execution40-fp32-bs128-horizon60"},
        required_observable_fields=["user_authorized_matched_execution40"],
        created_from_source_digest=sha256_file(root / BASE / "protocol.md"),
        approval={"activation":"user-authorized execution assay; no automatic successor", "authority_ref":BASE+"/protocol.md"},
        action_rule={"field":"user_authorized_matched_execution40", "operator":"eq", "threshold":True, "action":"RUN_EXECUTION40"})
    atomic_json(root / "research_memory/policies/gptrans-execution40.v1.json", policy)
    role_ref, budget_ref = BASE + "/gpu/role_plan.json", BASE + "/gpu/budget.json"
    roles = {kind: "not_applicable" if kind == "external_submission" else "applicable" for kind in ROLE_EVENT_KINDS}
    atomic_json(root / role_ref, {"role_applicability": roles, "train_rows": [0,100000],
        "development_rows": [100000,150000], "official_validation": "forbidden",
        "test_dev": "forbidden", "test_challenge": "forbidden"})
    atomic_json(root / budget_ref, {"authority_ref": BASE + "/protocol.md", "estimated_wall_hours": 4,
        "estimated_allocated_t4_hours": 8, "maximum_wall_hours": 7, "maximum_allocated_t4_hours": 14,
        "allocated_devices": 2, "baseline_retraining": True,
        "baseline_retraining_reason": "user-authorized matched execution/accuracy test; not a candidate architecture screen",
        "automatic_successor_authorized": False})
    spec = deepcopy(previous)
    spec.update(experiment_id="gptrans-execution40", logical_run_id=RUN,
                arms=[], prospective={"arms":[]})
    recipes, config_arms, plan_refs = {}, {}, []
    for arm_id in ARMS:
        folder = BASE + "/gpu/" + arm_id
        if (root / folder / "rml_plan").exists():
            raise ValueError("Do not refreeze an existing prospective attempt")
        optimized = arm_id == ARMS[1]
        recipe_ref = folder + "/contract.json"
        recipe = read(OLD + "/gpu/" + model + "/contract.json")
        recipe.update(experiment_purpose="delivery_experiment", variant=model,
            epochs=40, schedule_horizon_epochs=60, optimizer_steps=31240, sample_presentations=3998720,
            selection="minimum EMA999 internal50K MAE over epochs0..39; horizon60 prefix",
            execution_backend="optimized-v1" if optimized else "original-v1",
            optimizer_execution={"foreach": optimized, "fused": False},
            ema_execution="tensor-list" if optimized else "per-tensor",
            protected_roles_forbidden=True)
        atomic_json(root / recipe_ref, recipe)
        recipes[arm_id] = recipe_ref
        arm = deepcopy(prior_arm)
        arm.update(arm_id=arm_id, scientific_role="candidate" if optimized else "reference")
        arm["training"]["recipe"]["sha256"] = normalized_source_sha256(root / recipe_ref)
        for role in arm["data"]["roles"]:
            role["usage_sha256"] = sha256_file(root / role_ref)
        tid = "TC-gptrans-execution40-" + ("optimized" if optimized else "original") + "-100k-s42"
        plan = read(OLD + "/gpu/" + model + "/plan_input.json")
        t = plan["trajectory"]
        t.update(trajectory_id=tid, family_id="gptrans-execution40", question=
            "Does the opt-in execution backend reduce complete forty-pass runtime without material accuracy regression?" if optimized else
            "What is the matched original-backend forty-pass speed/accuracy control?")
        t["hypothesis"].update(hypothesis_id="H-"+tid, observed_deficiency=t["question"],
            supporting_evidence_ids=["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42"], alternative_explanations=[
                "CPU contention and first-use costs can change timing; foreach roundoff can change trajectories."],
            changed_mechanism="execution-only backend" if optimized else "fresh matched timing control",
            cheapest_falsifier="one paired seed42 dual40 test explicitly authorized by the user",
            related_closed_family_ids=["gptrans-runtime-profiling"], expected_native_cost_ref="cost-"+tid,
            decision_changed_if_positive="consider an independently qualified execution backend, not architecture promotion",
            decision_changed_if_negative="retain the original backend",
            historical_unknowns=["full forty-pass numerical/selection drift is unmeasured"])
        prelaunch = assess_comparison_prelaunch(candidate_id=tid,
            candidate_plan={"comparison_identity": {}, "source_config_status":"frozen", "source_commit_or_archive":commit},
            reference_id=None, reference_bundle=None, experiment_purpose="delivery_experiment",
            intervention_group_id="execution40", declared_intervention_fields=[],
            role_applicability_plan=roles, trace_plan={k:True for k in TRACE_FIELD_DECLARATIONS},
            runtime_qualification_plan={"status":"declared", "runtime_certificate_required":True,
                                        "qualification_scope":"per-arm T4 deterministic optimizer-inclusive repeat"})
        from .comparison_readiness import validate_server_comparison_prelaunch
        validate_server_comparison_prelaunch(prelaunch, experiment_purpose="delivery_experiment", reference_bundle=None)
        prelaunch_ref = folder + "/comparison_readiness_prelaunch.json"
        atomic_json(root / prelaunch_ref, prelaunch)
        t["state_at_start"].update(source_commit=commit, source_config_identity=canonical_fingerprint(arm),
            contract_refs=[recipe_ref,BASE+"/protocol.md",prelaunch_ref], reference_ids=[],
            parent_trajectory_ids=["TC-gptrans-g1-bond-local-100k-s42"],
            prior_trajectory_ids=["TC-gptrans-g1-bond-local-100k-s42"], prior_evidence_ids=["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42"],
            role_snapshot_refs=[role_ref],budget_snapshot_ref=budget_ref)
        t["actions"]=[{"action_id":"A001","type":"single_mechanism_screen","source_commit":commit,
            "run_ids":[RUN+":"+arm_id],"attempt_ids":["v1"],"evidence_refs":[recipe_ref],"cost_event_ids":["cost-"+tid]}]
        t["result"]={"evidence_ids":[],"evidence_refs":[]}
        t["decision"].update(decision_ref=BASE+"/protocol.md",outcome="ACTIVE",
            next_allowed_actions=["one isolated forty-pass execution assay then saved-output acceptance"],
            reopen_conditions=["terminal evidence or infrastructure diagnosis"])
        plan["decision_state"].update(active_reference_ids=[],available_actions=["RUN_EXECUTION40","DEFER"],
            chosen_action="RUN_EXECUTION40",policy_id="gptrans-execution40",budget_snapshot_ref=budget_ref,
            role_snapshot_refs=[role_ref],state_timestamp=datetime.now(timezone.utc).isoformat(),source_commit=commit)
        plan["costs"][0].update(cost_event_id="cost-"+tid,trajectory_id=tid,run_id=RUN+":"+arm_id,
                                platform="kaggle2",evidence_ref=budget_ref)
        for unit in ("device_hours","wall_hours"):
            plan["costs"][0]["measurement"][unit]={"status":"estimated","value":4}
        plan_ref=folder+"/plan_input.json"
        atomic_json(root / plan_ref,plan)
        plan_refs.append(plan_ref)
        spec["arms"].append(arm)
        spec["prospective"]["arms"].append({"arm_id":arm_id,"trajectory_id":tid,
            "plan_spec_ref":plan_ref,"plan_spec_sha256":sha256_file(root / plan_ref),"output":folder+"/rml_plan"})
        config_arms[arm_id]={**facts,"trajectory_id":tid}
    frozen=ExperimentSpec(spec)
    frozen.write(root / BASE / "gpu/spec.json")
    config={**prior_config,"spec_identity":frozen.identity,"arms":config_arms,"path_manifest_sha256":None,
        "platform_id":"kaggle2-t4-gptrans-execution40","requested_kernel":"kaseichou/molgap-gptrans-execution40-dual-s42",
        "output_subdirectory":"gptrans_execution40","maximum_wall_seconds":25200,
        "execution_study":{"epochs":40,"schedule_horizon_epochs":60,"model_variant":model,
            "physical_batch":128,"precision":"fp32","tf32_enabled":False},
        "material_gate_eV":None}
    for k in ("reference_bundle_ref","reference_bundle_sha256"):
        config.pop(k,None)
    atomic_json(root / BASE / "gpu/screen_config.json",config)
    workflow=read(OLD+"/gpu/release_workflow.json")
    workflow.update(spec_identity=frozen.identity,recipe_files=recipes,
        initial_states={a:facts["initial_file"] for a in ARMS},
        artifacts={facts["initial_file"]:UploadArtifact.from_file(initial).to_workflow(),
            "target_transform.json":workflow["artifacts"]["target_transform.json"]},
        kernel_metadata=BASE+"/gpu/kernel-metadata.json",
        dataset_metadata={"title":"MolGap GPTrans Execution40 Source", "id":"kaseichou/molgap-gptrans-execution40-source",
                          "licenses":[{"name":"other"}],"isPrivate":True},
        required_modules=["molgap.gptrans_execution_study","molgap.gptrans_execution_backend",
                          "molgap.pcqm_gptrans_step_profile","molgap.pcqm_gptrans_v4"])
    for arm_id, plan_ref in zip(ARMS, plan_refs):
        workflow["artifacts"][arm_id + "_plan.json"] = UploadArtifact.from_file(root / plan_ref).to_workflow()
    # Reuse the reviewed family payload, not a repository-wide sensitive-file scan.
    sources = [p for p in workflow["source_paths"] if p.startswith("src/molgap/")]
    sources += ["src/molgap/gptrans_execution_backend.py", "src/molgap/gptrans_execution_study.py",
                "src/molgap/gptrans_execution_prepare.py", "src/molgap/gptrans_execution_acceptance.py"]
    workflow["source_paths"]=sorted(set(sources))+[
        *recipes.values(), *plan_refs, workflow["entry_template"],BASE+"/gpu/kernel-metadata.json",BASE+"/gpu/screen_config.json"]
    atomic_json(root / BASE / "gpu/release_workflow.json",workflow)
    return {"spec_identity":frozen.identity,"source_commit":commit,"submission_performed":False}
