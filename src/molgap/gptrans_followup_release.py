"""Question-specific declarations over the standard prospective release path."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from .comparison_readiness import ROLE_EVENT_KINDS, TRACE_FIELD_DECLARATIONS, assess_comparison_prelaunch
from .evidence_pointers import load_json_object
from .experiment_spec import ExperimentSpec
from .experiment_package import _name
from .experiment_staging import UploadArtifact
from .research_memory.candidate_reference import enroll_candidate_reference
from .server_acceptance import write_server_comparison_prelaunch
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, sha256_file
from .v4_runtime import normalized_source_sha256

BASE = "experiments/pcqm_gptrans_input_ema_100k"
OLD = "experiments/pcqm_gptrans_author_alignment"
MODES = ("degree_path_bond_mean", "degree_scale_ema999")
RUN = "gptrans-g1-path-ema-dual-s42"


def freeze_followup(root: Path, *, base=BASE, modes=MODES, run=RUN, terminal_reference=False, study=None):
    BASE, MODES, RUN = base, modes, run
    root = root.resolve()
    read = lambda ref: load_json_object(root / ref)
    if (study or {}).get("frozen_reference_bundle_ref"):
        from .server_acceptance import validate_server_scientific_prelaunch
        bundle_ref = study["frozen_reference_bundle_ref"]
        bundle = read(bundle_ref)
    elif terminal_reference:
        from .research_memory.candidate_reference import enroll_terminal_candidate_reference
        reference = (study or {}).get("terminal_reference", {})
        prior = reference.get("experiment_ref", "experiments/pcqm_gptrans_input_ema_100k")
        reference_arm = reference.get("arm", "degree_scale_ema999")
        bundle = enroll_terminal_candidate_reference(root,
            finalized_ref=prior + "/gpu/" + reference_arm + "/rml_plan/rml_finalized",
            readiness_ref=prior + "/gpu/" + reference_arm + "/results/comparison_readiness.json",
            acceptance_ref=prior + reference.get("acceptance_suffix", "/gpu/results/acceptance.json"), acceptance_arm=reference_arm,
            destination=BASE + "/reference", bundle_id=("reference-" + study["experiment_id"] + "-100k-s42-v1") if study else "reference-gptrans-g1-ema999-100k-s42-v1")
    else:
        bundle = enroll_candidate_reference(root, OLD + "/gpu/degree_scale/rml_plan/candidate_qualification.json",
            BASE + "/reference", "reference-gptrans-g1-100k-s42-v1")
    bundle_ref = (study or {}).get("frozen_reference_bundle_ref", BASE + "/reference/reference_bundle.json")
    identity = bundle["comparison_identity"]
    accepted = read(OLD + "/verification_recovery/acceptance_v2.json")
    initial = root / "platforms/_records/kaggle/training/gptrans_author_inputs_verification_recovery_v2/gptrans_author_inputs/degree_initial_state.pt"
    if sha256_file(initial) != accepted["degree_initial_file_sha256"]:
        raise ValueError("Accepted G1 initialization changed")
    old_spec = ExperimentSpec.from_json((root / OLD / "gpu/spec.json").read_text()).to_dict()
    declaration = deepcopy(old_spec)
    declaration.update(experiment_id=study["experiment_id"] if study else "gptrans-g1-recipe-path" if terminal_reference else "gptrans-g1-input-ema", logical_run_id=RUN, arms=[], prospective={"arms": []})
    budget_ref, role_ref = BASE + "/gpu/budget.json", BASE + "/gpu/role_plan.json"
    single_reason = study.get("single_arm_reason") if study else None
    wall_cap = (study or {}).get("maximum_wall_hours", 6)
    estimate = (study or {}).get("estimated_wall_hours", 4)
    if len(MODES) == 1 and not single_reason:
        raise ValueError("Single-arm study needs an explicit allocation justification")
    atomic_json(root / budget_ref, {"estimated_wall_hours": estimate, "estimated_allocated_t4_hours": estimate * 2,
        "maximum_wall_hours": wall_cap, "maximum_allocated_t4_hours": wall_cap * 2, "allocated_devices": 2,
        "baseline_retraining": False, "automatic_successor_authorized": False,
        "single_arm_reason": single_reason, "authority_ref": BASE + "/protocol.md"})
    roles = {kind: "not_applicable" if kind == "external_submission" else "applicable" for kind in ROLE_EVENT_KINDS}
    scale_only = tuple(MODES) == ("scale_ema",)
    matched_scale = scale_only and bool((study or {}).get("frozen_reference_bundle_ref"))
    atomic_json(root / role_ref, {"role_applicability": roles, "train_rows": [0,500000] if scale_only else [0,100000], "development_rows": [500000,550000] if scale_only else [100000,150000],
        "official_validation": "forbidden", "test_dev": "forbidden", "test_challenge": "forbidden"})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    implementation = normalized_source_sha256(root / "src/molgap/gptrans_author_variants.py")
    source_arm = next(a for a in old_spec["arms"] if a["arm_id"] == "degree_scale")
    source_plan = read(OLD + "/gpu/degree_scale/plan_input.json")
    arms_config, recipes = {}, {}
    capacity_inputs = {}
    for mode in MODES:
        folder = BASE + "/gpu/" + mode
        if (root / folder / "rml_plan").exists():
            raise ValueError("Prospective plan is immutable; reconcile before refreezing")
        ema = mode == "degree_scale_ema999"
        suffix = {"degree_group_decay_ema999": "group-decay", "degree_path_endpoints_ema999": "path-endpoints"}.get(mode, "ema999" if ema else "path-mean")
        if study:
            suffix = study["arms"][mode]["suffix"]
        trajectory = "TC-gptrans-g1-" + suffix + ("-500k-s42" if mode == "scale_ema" else "-100k-s42")
        recipe_ref = folder + "/contract.json"
        recipe = read("experiments/pcqm_gptrans_v5_audit_reference/contract.json")
        recipe.update(initial_state_sha256=accepted["degree_initial_file_sha256"], ema_decay=.999 if ema or terminal_reference else .9999,
            platform=(study or {}).get("platform_id", "kaggle3"), independent_models=1, variant=mode, reference_id=bundle["reference_id"])
        from .gptrans_capacity import MODES as capacity_modes
        transition = mode == "degree_pair_transition_ema999"
        local_control = mode == "degree_bond_local_cap_ema999"
        scale = mode == "scale_ema"
        model_mode = (study or {}).get("scale_model_variant", "degree_scale_ema999") if scale else mode
        capacity = model_mode in capacity_modes or transition or local_control
        prepared_initial, prepared_file = initial, "degree_initial_state.pt"
        prepared_tensor_sha = source_arm["initialization"]["state_sha256"]
        expected_parameters = recipe["model_parameters"]
        if capacity:
            from .pcqm_gptrans_v4 import capacity_module
            module = capacity_module(model_mode)
            freeze_initial, configuration = module.freeze_initial, module.configuration
            load_initial, architecture_identity = module.load_initial, module.architecture_identity
            prepared_file = model_mode + "_initial.pt"
            prepared_initial = root / "platforms/_records/kaggle/initializations" / prepared_file
            retained_facts = (study or {}).get("retained_initial_facts") if scale else None
            if retained_facts:
                facts = dict(retained_facts)
                if (sha256_file(prepared_initial) != facts.pop("file_sha256")
                        or facts["architecture_identity"] != architecture_identity(model_mode)):
                    raise ValueError("Retained scale initialization/architecture differs")
            elif prepared_initial.exists():
                from .pcqm_gptrans_v4 import _state_sha256
                model = load_initial(model_mode, prepared_initial)
                facts = {"parameters": sum(p.numel() for p in model.parameters()),
                    "state_sha256": _state_sha256(model), "architecture_identity": architecture_identity(model_mode)}
            else:
                facts = freeze_initial(model_mode, initial, prepared_initial)
            prepared_tensor_sha, expected_parameters = facts["state_sha256"], facts["parameters"]
            recipe.update(initial_state_sha256=sha256_file(prepared_initial), model_parameters=expected_parameters,
                          capacity_configuration=configuration(model_mode), architecture_sha256=facts["architecture_identity"])
            capacity_inputs[prepared_file] = UploadArtifact.from_file(prepared_initial).to_workflow()
        if terminal_reference:
            recipe.update(optimizer_parameter_groups="bias-and-1d-no-decay-v1" if "group_decay" in mode else "single-group",
                          endpoint_path_encoding="all-shortest-first-minus-last-half-v1" if "path_endpoints" in mode else "none")
        if mode == "degree_decay001_ema999":
            recipe["weight_decay"] = .01
            recipe["optimizer"] = "AdamW-foreach-false-lr0.001-weight-decay0.01"
        if mode == "degree_pair_depth_scale_ema999":
            recipe["pair_residual_scale"] = 12 ** -0.5
        if mode in {"degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999"}:
            recipe["readout_intervention"] = mode
        if scale:
            from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
            recipe.update(format="molgap-gptrans-scale-ema-contract-v1", benchmark_id="pcqm4mv2-fixed500k-ema-equal-update-v1",
                manifest_sha256=FIXED_500K_MANIFEST_SHA256, fixed_train_rows=[0,500000], fixed_development_rows=[500000,550000],
                optimizer_steps=46860, sample_presentations=5998080, observation_rungs=60, optimizer_steps_per_rung=781,
                sampler="seed42-continuous-fixed500k-step-v1", ema_decays=[.9999,.999],
                experiment_purpose="transfer_study", comparison_class="PAIRED_ENDPOINT",
                selection="each EMA view independently selects its minimum internal50k MAE over the same60 rungs",
                reference_scope="100K context only; no accepted500K same-contract comparator")
            recipe.update(dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1@" + FIXED_500K_MANIFEST_SHA256,
                model_id="gptrans-g1-core12x256-pair32", data_role_fingerprint=canonical_fingerprint({"train":[0,500000],"development":[500000,550000]}),
                row_order_fingerprint=canonical_fingerprint({"sampler":"seed42-continuous-fixed500k-step-v1","steps":46860,"batch":128}))
            if matched_scale:
                recipe.update(model_variant=model_mode, ema_decay=.999, experiment_purpose="architecture_comparison",
                    comparison_class="prospective_only", reference_scope="qualified retained same-budget500K EMA999 control",
                    model_id=facts["architecture_identity"], directional_transfer_gate_eV=.001)
        atomic_json(root / recipe_ref, recipe)
        recipes[mode] = recipe_ref
        arm = deepcopy(source_arm)
        arm.update(arm_id=mode, addons=[{"name": mode, "version": "1", "config": {}, "source_sha256": implementation}])
        if capacity:
            arm["initialization"]["state_sha256"] = prepared_tensor_sha
            arm["addons"][0]["source_sha256"] = normalized_source_sha256(Path(module.__file__))
        if scale:
            arm.update(family={"name":"gptrans_scale_ema","version":"1"}, addons=[], addon_semantics="baseline",
                scientific_role="ablation")
            arm["data"]["dataset"]["sha256"] = FIXED_500K_MANIFEST_SHA256
            arm["data"]["split"] = {"name":"fixed500k-internal50k","version":"1","sha256":FIXED_500K_MANIFEST_SHA256}
            arm["data"]["roles"] = [{"role":role, "membership_sha256":canonical_fingerprint({"rows":bounds}),
                "row_order_sha256":canonical_fingerprint({"source_order":bounds}), "usage_sha256":sha256_file(root / recipe_ref)}
                for role,bounds in (("train",[0,500000]),("development",[500000,550000]))]
            arm["training"]["recipe"]["name"] = "gptrans_scale_ema_v1"
            arm["training"]["sampler"] = {"name":"seed42-continuous-fixed500k-step-v1","version":"1",
                "sha256":canonical_fingerprint({"seed":42,"train_rows":500000,"steps":46860,"batch":128,"tail":"drop32-per-cycle"})}
        if "readout" in mode:
            arm["addons"][0]["source_sha256"] = normalized_source_sha256(root / "src/molgap/gptrans_readout.py")
        arm["training"]["recipe"]["sha256"] = normalized_source_sha256(root / recipe_ref)
        arm_role_ref = recipe_ref if scale else role_ref
        for role in arm["data"]["roles"]:
            role["usage_sha256"] = sha256_file(root / arm_role_ref)
        candidate = dict(identity)
        if scale:
            candidate.update(benchmark_identity="pcqm4mv2-fixed500k-ema-equal-update-v1",
                dataset_identity="pcqm4mv2-ogb-fixed500k-v1@"+FIXED_500K_MANIFEST_SHA256,
                data_role_identity=canonical_fingerprint({"train":[0,500000],"development":[500000,550000]}),
                row_membership_identity=FIXED_500K_MANIFEST_SHA256,
                row_order_identity=arm["training"]["sampler"]["sha256"],
                evaluation_role_identity="fixed500k:development-500000-550000", selection_role_identity="fixed500k:development-500000-550000")
            if matched_scale:
                candidate["architecture_config_identity"] = facts["architecture_identity"]
                purpose, fields = "architecture_comparison", ["architecture_config_identity"]
            else:
                purpose, fields = "transfer_study", []
        elif capacity:
            candidate["architecture_config_identity"] = facts["architecture_identity"]
            purpose, fields = "architecture_comparison", ["architecture_config_identity"]
        elif terminal_reference and mode in {"degree_group_decay_ema999", "degree_decay001_ema999"}:
            from .pcqm_gptrans_v4 import _scientific_fields
            candidate.update(optimizer_identity=_scientific_fields(mode)["optimizer_fingerprint"], optimizer_mode="adamw-bias-and-1d-no-decay-v1" if mode == "degree_group_decay_ema999" else "adamw-single-group-wd001-v1")
            purpose, fields = "optimizer_comparison", ["optimizer_identity", "optimizer_mode"]
        elif terminal_reference and mode != "degree_path_bond_mean_ema999":
            module_key, module = (("readout_module", "gptrans_readout.py") if "readout" in mode else
                                 ("pair_scale_module", "gptrans_pair_scale.py") if mode == "degree_pair_depth_scale_ema999"
                                  else ("endpoint_module", "gptrans_endpoint_paths.py"))
            candidate["architecture_config_identity"] = canonical_fingerprint({"core": recipe["architecture_sha256"], "variant": mode, "implementation": implementation,
                module_key: normalized_source_sha256(root / "src/molgap" / module)})
            purpose, fields = "mechanism_comparison", ["architecture_config_identity"]
        elif ema:
            candidate.update(ema_decay=.999, checkpoint_selection_identity="best-development-ema999-60epochs")
            purpose, fields = "ema_comparison", ["ema_decay", "checkpoint_selection_identity"]
        else:
            candidate["architecture_config_identity"] = canonical_fingerprint({"core": recipe["architecture_sha256"], "variant": mode, "implementation": implementation})
            purpose, fields = "mechanism_comparison", ["architecture_config_identity"]
        prelaunch = assess_comparison_prelaunch(candidate_id=trajectory,
            candidate_plan={"comparison_identity": candidate, "source_config_status": "frozen", "source_commit_or_archive": commit},
            reference_id=None if scale and not matched_scale else bundle["reference_id"], reference_bundle=None if scale and not matched_scale else bundle, experiment_purpose=purpose,
            intervention_group_id="gptrans-g1-followup", mechanism_id=mode,
            declared_intervention_fields=fields, role_applicability_plan=roles,
            trace_plan={k: True for k in TRACE_FIELD_DECLARATIONS}, runtime_qualification_plan={"status": "declared",
                "runtime_certificate_required": True, "qualification_scope": identity["runtime_certificate_scope"]})
        if scale and not matched_scale:
            from .comparison_readiness import validate_server_comparison_prelaunch
            # A transfer study has no matched500K reference. Use the existing
            # explicitly noncausal gate; do not relax the scientific helper.
            validate_server_comparison_prelaunch(prelaunch, experiment_purpose=purpose, reference_bundle=None)
            atomic_json(root / folder / "comparison_readiness_prelaunch.json", prelaunch)
        else:
            write_server_comparison_prelaunch(root / folder / "comparison_readiness_prelaunch.json",
                comparison_prelaunch=prelaunch, experiment_purpose=purpose, reference_bundle=bundle,
                repo_root=root, reference_bundle_path=root / bundle_ref)
        plan = deepcopy(source_plan)
        question = (study["arms"][mode]["question"] if study else {"degree_group_decay_ema999": "Does exempting bias and 1D tensors from AdamW decay improve G1 EMA999 without adding inference capacity?",
                     "degree_path_endpoints_ema999": "Does all-shortest-path endpoint bond contrast improve G1 EMA999 while avoiding atom-index tie choices?"}[mode]
                    if terminal_reference or study else "Does reducing G1 EMA lag improve selected predictions without changing live optimization?" if ema
                    else "Does accepted chemical path mean add information to degree-scaled G1?")
        cost = "cost-" + trajectory
        t = plan["trajectory"]
        t.update(trajectory_id=trajectory, family_id="gptrans-g1-" + mode, question=question)
        t["hypothesis"].update(hypothesis_id="H-" + trajectory, observed_deficiency=question,
            supporting_evidence_ids=[bundle["reference_id"], "pcqm-gptrans-author-path-bond-mean-100k-s42"],
            alternative_explanations=(["Optimizer grouping may change convergence without improving generalization; endpoint contrast may duplicate existing pair information."] if terminal_reference else ["EMA lag can coexist with overfitting; chemical path content may duplicate the degree-scaled core."]),
            changed_mechanism=mode, cheapest_falsifier="one matched seed42 screen against accepted G1; no baseline retraining",
            expected_native_cost_ref=cost, related_closed_family_ids=["gptrans-pair-prenorm", "gptrans-runtime-profiling"],
            decision_changed_if_positive="controller interpretation only; no automatic scale or seed release",
            decision_changed_if_negative="close this exact follow-up hypothesis",
            historical_unknowns=["training stochasticity unmeasured; material gate is a policy choice"])
        t["state_at_start"].update(source_commit=commit, source_config_identity=canonical_fingerprint(arm),
            contract_refs=[recipe_ref, BASE + "/protocol.md", folder + "/comparison_readiness_prelaunch.json"],
            reference_ids=[bundle["reference_id"]], parent_trajectory_ids=["TC-gptrans-g1-ema999-100k-s42" if terminal_reference else "TC-gptrans-author-degree-scale-100k-s42"],
            prior_trajectory_ids=["TC-gptrans-author-path-bond-mean-100k-s42"],
            prior_evidence_ids=[bundle["reference_id"], "pcqm-gptrans-author-path-bond-mean-100k-s42"],
            role_snapshot_refs=[role_ref], budget_snapshot_ref=budget_ref)
        if study:
            facts = study["arms"][mode]
            t["hypothesis"].update(supporting_evidence_ids=study["supporting_evidence_ids"],
                alternative_explanations=facts["alternative_explanations"])
            t["state_at_start"].update(prior_trajectory_ids=study["prior_trajectory_ids"],
                prior_evidence_ids=study["supporting_evidence_ids"])
            if study.get("reference_parent_trajectory_id"):
                t["state_at_start"]["parent_trajectory_ids"] = [study["reference_parent_trajectory_id"]]
                t["hypothesis"]["cheapest_falsifier"] = "one matched seed42 screen against the accepted frozen local-bond model; no baseline retraining"
        if scale:
            t["state_at_start"]["role_snapshot_refs"] = [recipe_ref]
            t["hypothesis"]["cheapest_falsifier"] = "one live500K trajectory with two retained EMA views at equal updates;100K reference is context only"
            if matched_scale:
                t["hypothesis"]["cheapest_falsifier"] = "one matched500K local stream against retained same-budget control; no duplicate baseline"
        t["actions"] = [{"action_id": "A001", "type": "single_mechanism_screen", "source_commit": commit,
            "run_ids": [RUN + ":" + mode], "attempt_ids": ["v1"], "evidence_refs": [recipe_ref], "cost_event_ids": [cost]}]
        t["decision"].update(decision_ref=BASE + "/protocol.md", next_allowed_actions=["one isolated T4 arm then full saved-artifact acceptance"])
        plan["decision_state"].update(active_reference_ids=[bundle["reference_id"]], available_actions=["RUN_G1_FOLLOWUP", "DEFER"],
            chosen_action="RUN_G1_FOLLOWUP", policy_id="gptrans-g1-followup", budget_snapshot_ref=budget_ref,
            role_snapshot_refs=[arm_role_ref], state_timestamp=datetime.now(timezone.utc).isoformat(), source_commit=commit)
        plan["costs"][0].update(cost_event_id=cost, trajectory_id=trajectory, run_id=RUN + ":" + mode,
            platform=(study or {}).get("platform_id", "kaggle3"), evidence_ref=budget_ref)
        for unit in ("device_hours", "wall_hours"):
            plan["costs"][0]["measurement"][unit] = {"status": "estimated", "value": estimate}
        plan_ref = folder + "/plan_input.json"
        atomic_json(root / plan_ref, plan)
        declaration["arms"].append(arm)
        declaration["prospective"]["arms"].append({"arm_id": mode, "trajectory_id": trajectory,
            "plan_spec_ref": plan_ref, "plan_spec_sha256": sha256_file(root / plan_ref), "output": folder + "/rml_plan"})
        arms_config[mode] = {"trajectory_id": trajectory, "initial_file": prepared_file,
            "initial_file_sha256": sha256_file(prepared_initial), "comparison_identity": candidate,
            "experiment_purpose": purpose, "declared_intervention_fields": fields}
        if capacity:
            arms_config[mode].update(expected_parameters=expected_parameters, initial_tensor_sha256=prepared_tensor_sha)
    spec = ExperimentSpec(declaration)
    spec.write(root / BASE / "gpu/spec.json")
    config = read(OLD + "/gpu/screen_config.json")
    config.update(spec_identity=spec.identity, arms=arms_config, platform_id="kaggle3-t4-gptrans-g1-followup",
        reference_bundle_ref=bundle_ref, reference_bundle_sha256=sha256_file(root / bundle_ref), material_gate_eV=.003)
    if terminal_reference:
        config.update(path_manifest_sha256=(config["path_manifest_sha256"] if "degree_path_bond_mean_ema999" in MODES else None), requested_kernel="nvoid912/molgap-gptrans-g1-group-path-dual-s42",
            output_subdirectory="gptrans_recipe_path_screen")
    if study:
        config.update(requested_kernel=study["kernel"], output_subdirectory=study["output_subdirectory"])
        config["platform_id"] = study.get("platform_id", config["platform_id"])
        if "scale_ema" in MODES:
            scale_tid = arms_config["scale_ema"]["trajectory_id"]
            config["scale_study"] = {"logical_run_id":RUN+":scale_ema", "physical_arm_trajectory_id":scale_tid,
                "initial_file_sha256":arms_config["scale_ema"]["initial_file_sha256"],
                "training_estimate_cap_hours":study["training_estimate_cap_hours"],
                "view_trajectories":{view:scale_tid for view in ("ema9999","ema999")},
                "role_plan":BASE+"/gpu/scale_ema/contract.json", "profiling_only":False,
                "expected_optimizer_steps":46860, "expected_sample_presentations":5998080}
            if matched_scale:
                config["scale_study"].update(model_variant=model_mode,
                    initial_file=arms_config["scale_ema"]["initial_file"], expected_parameters=expected_parameters,
                    phase_profiling_steps=4)
                config["dataset_manifest_sha256"] = FIXED_500K_MANIFEST_SHA256
                config["directional_transfer_gate_eV"] = .001
        config.update(maximum_wall_seconds=int(wall_cap * 3600),
                      training_estimate_cap_hours=study.get("training_estimate_cap_hours", 4.5))
    if single_reason:
        config["single_arm_reason"] = single_reason
    atomic_json(root / BASE / "gpu/screen_config.json", config)
    workflow = read(OLD + "/gpu/release_workflow.json")
    workflow.update(spec_identity=spec.identity, recipe_files=recipes,
        initial_states={mode: "degree_initial_state.pt" for mode in MODES},
        artifacts={"degree_initial_state.pt": UploadArtifact.from_file(initial).to_workflow(),
            "target_transform.json": UploadArtifact.from_file(root / bundle["target_transform_asset_ref"]).to_workflow()},
        entry_template=BASE + "/gpu/run.py", kernel_metadata=BASE + "/gpu/kernel-metadata.json",
        dataset_metadata={"title": "MolGap GPTrans G1 Group Path Source" if terminal_reference else "MolGap GPTrans G1 Path EMA Source", "id": "nvoid912/molgap-gptrans-g1-group-path-source" if terminal_reference else "nvoid912/molgap-gptrans-g1-path-ema-source",
            "licenses": [{"name": "other"}], "isPrivate": True})
    if study:
        workflow["dataset_metadata"].update(title=study["dataset_title"], id=study["dataset"])
        workflow["entry_template"] = study.get("entry_template", workflow["entry_template"])
    if capacity_inputs:
        workflow["artifacts"] = {**capacity_inputs, "target_transform.json": workflow["artifacts"]["target_transform.json"]}
        workflow["initial_states"] = {mode: arms_config[mode]["initial_file"] for mode in MODES}
        workflow["required_modules"].append("molgap.gptrans_capacity")
        if "degree_pair_transition_ema999" in MODES:
            workflow["required_modules"].append("molgap.gptrans_pair_transition")
        if "degree_bond_local_cap_ema999" in MODES:
            workflow["required_modules"].append("molgap.gptrans_local_control")
        if "scale_ema" in MODES:
            workflow["artifacts"]["degree_initial_state.pt"] = UploadArtifact.from_file(initial).to_workflow()
    if "scale_ema" in MODES:
        workflow["required_modules"].extend(["molgap.gptrans_scale_profile", "molgap.gptrans_scale_ema"])
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "src/molgap"], cwd=root).decode().split("\0")
    sources = []
    for p in tracked:
        if p.endswith(".py") and "/archive/" not in p and p not in (study or {}).get("source_exclusions", []):
            try:
                sources.append(_name(p))
            except ValueError:
                continue
    workflow["source_paths"] = sources + [
        *recipes.values(), workflow["entry_template"], BASE + "/gpu/kernel-metadata.json", BASE + "/gpu/screen_config.json"]
    if terminal_reference and any(mode in {"degree_path_endpoints_ema999", "degree_group_decay_ema999"} for mode in MODES):
        workflow["required_modules"].append("molgap.gptrans_endpoint_paths")
    if "degree_pair_depth_scale_ema999" in MODES:
        workflow["required_modules"].append("molgap.gptrans_pair_scale")
    if any("readout" in mode for mode in MODES):
        workflow["required_modules"].append("molgap.gptrans_readout")
    atomic_json(root / BASE / "gpu/release_workflow.json", workflow)
    return {"spec_identity": spec.identity, "prelaunch_validated": True, "compute_released": False}
