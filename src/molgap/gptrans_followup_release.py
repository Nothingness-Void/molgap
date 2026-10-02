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
    if terminal_reference:
        from .research_memory.candidate_reference import enroll_terminal_candidate_reference
        prior = "experiments/pcqm_gptrans_input_ema_100k"
        bundle = enroll_terminal_candidate_reference(root,
            finalized_ref=prior + "/gpu/degree_scale_ema999/rml_plan/rml_finalized",
            readiness_ref=prior + "/gpu/degree_scale_ema999/results/comparison_readiness.json",
            acceptance_ref=prior + "/gpu/results/acceptance.json", acceptance_arm="degree_scale_ema999",
            destination=BASE + "/reference", bundle_id=("reference-" + study["experiment_id"] + "-100k-s42-v1") if study else "reference-gptrans-g1-ema999-100k-s42-v1")
    else:
        bundle = enroll_candidate_reference(root, OLD + "/gpu/degree_scale/rml_plan/candidate_qualification.json",
            BASE + "/reference", "reference-gptrans-g1-100k-s42-v1")
    bundle_ref = BASE + "/reference/reference_bundle.json"
    identity = bundle["comparison_identity"]
    accepted = read(OLD + "/verification_recovery/acceptance_v2.json")
    initial = root / "platforms/_records/kaggle/training/gptrans_author_inputs_verification_recovery_v2/gptrans_author_inputs/degree_initial_state.pt"
    if sha256_file(initial) != accepted["degree_initial_file_sha256"]:
        raise ValueError("Accepted G1 initialization changed")
    old_spec = ExperimentSpec.from_json((root / OLD / "gpu/spec.json").read_text()).to_dict()
    declaration = deepcopy(old_spec)
    declaration.update(experiment_id=study["experiment_id"] if study else "gptrans-g1-recipe-path" if terminal_reference else "gptrans-g1-input-ema", logical_run_id=RUN, arms=[], prospective={"arms": []})
    budget_ref, role_ref = BASE + "/gpu/budget.json", BASE + "/gpu/role_plan.json"
    atomic_json(root / budget_ref, {"estimated_wall_hours": 4, "estimated_allocated_t4_hours": 8,
        "maximum_wall_hours": 6, "maximum_allocated_t4_hours": 12, "allocated_devices": 2,
        "baseline_retraining": False, "automatic_successor_authorized": False, "authority_ref": BASE + "/protocol.md"})
    roles = {kind: "not_applicable" if kind == "external_submission" else "applicable" for kind in ROLE_EVENT_KINDS}
    atomic_json(root / role_ref, {"role_applicability": roles, "train_rows": [0,100000], "development_rows": [100000,150000],
        "official_validation": "forbidden", "test_dev": "forbidden", "test_challenge": "forbidden"})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    implementation = normalized_source_sha256(root / "src/molgap/gptrans_author_variants.py")
    source_arm = next(a for a in old_spec["arms"] if a["arm_id"] == "degree_scale")
    source_plan = read(OLD + "/gpu/degree_scale/plan_input.json")
    arms_config, recipes = {}, {}
    for mode in MODES:
        folder = BASE + "/gpu/" + mode
        if (root / folder / "rml_plan").exists():
            raise ValueError("Prospective plan is immutable; reconcile before refreezing")
        ema = mode == "degree_scale_ema999"
        suffix = {"degree_group_decay_ema999": "group-decay", "degree_path_endpoints_ema999": "path-endpoints"}.get(mode, "ema999" if ema else "path-mean")
        if study:
            suffix = study["arms"][mode]["suffix"]
        trajectory = "TC-gptrans-g1-" + suffix + "-100k-s42"
        recipe_ref = folder + "/contract.json"
        recipe = read("experiments/pcqm_gptrans_v5_audit_reference/contract.json")
        recipe.update(initial_state_sha256=accepted["degree_initial_file_sha256"], ema_decay=.999 if ema or terminal_reference else .9999,
            platform="kaggle3", independent_models=1, variant=mode, reference_id=bundle["reference_id"])
        if terminal_reference:
            recipe.update(optimizer_parameter_groups="bias-and-1d-no-decay-v1" if "group_decay" in mode else "single-group",
                          endpoint_path_encoding="all-shortest-first-minus-last-half-v1" if "path_endpoints" in mode else "none")
        if mode == "degree_pair_depth_scale_ema999":
            recipe["pair_residual_scale"] = 12 ** -0.5
        atomic_json(root / recipe_ref, recipe)
        recipes[mode] = recipe_ref
        arm = deepcopy(source_arm)
        arm.update(arm_id=mode, addons=[{"name": mode, "version": "1", "config": {}, "source_sha256": implementation}])
        arm["training"]["recipe"]["sha256"] = normalized_source_sha256(root / recipe_ref)
        for role in arm["data"]["roles"]:
            role["usage_sha256"] = sha256_file(root / role_ref)
        candidate = dict(identity)
        if terminal_reference and mode == "degree_group_decay_ema999":
            from .pcqm_gptrans_v4 import _scientific_fields
            candidate.update(optimizer_identity=_scientific_fields(mode)["optimizer_fingerprint"], optimizer_mode="adamw-bias-and-1d-no-decay-v1")
            purpose, fields = "optimizer_comparison", ["optimizer_identity", "optimizer_mode"]
        elif terminal_reference:
            module_key, module = (("pair_scale_module", "gptrans_pair_scale.py") if mode == "degree_pair_depth_scale_ema999"
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
            reference_id=bundle["reference_id"], reference_bundle=bundle, experiment_purpose=purpose,
            intervention_group_id="gptrans-g1-followup", mechanism_id=mode,
            declared_intervention_fields=fields, role_applicability_plan=roles,
            trace_plan={k: True for k in TRACE_FIELD_DECLARATIONS}, runtime_qualification_plan={"status": "declared",
                "runtime_certificate_required": True, "qualification_scope": identity["runtime_certificate_scope"]})
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
        t["actions"] = [{"action_id": "A001", "type": "single_mechanism_screen", "source_commit": commit,
            "run_ids": [RUN + ":" + mode], "attempt_ids": ["v1"], "evidence_refs": [recipe_ref], "cost_event_ids": [cost]}]
        t["decision"].update(decision_ref=BASE + "/protocol.md", next_allowed_actions=["one isolated T4 arm then full saved-artifact acceptance"])
        plan["decision_state"].update(active_reference_ids=[bundle["reference_id"]], available_actions=["RUN_G1_FOLLOWUP", "DEFER"],
            chosen_action="RUN_G1_FOLLOWUP", policy_id="gptrans-g1-followup", budget_snapshot_ref=budget_ref,
            role_snapshot_refs=[role_ref], state_timestamp=datetime.now(timezone.utc).isoformat(), source_commit=commit)
        plan["costs"][0].update(cost_event_id=cost, trajectory_id=trajectory, run_id=RUN + ":" + mode,
            platform="kaggle3", evidence_ref=budget_ref)
        for unit in ("device_hours", "wall_hours"):
            plan["costs"][0]["measurement"][unit] = {"status": "estimated", "value": 4}
        plan_ref = folder + "/plan_input.json"
        atomic_json(root / plan_ref, plan)
        declaration["arms"].append(arm)
        declaration["prospective"]["arms"].append({"arm_id": mode, "trajectory_id": trajectory,
            "plan_spec_ref": plan_ref, "plan_spec_sha256": sha256_file(root / plan_ref), "output": folder + "/rml_plan"})
        arms_config[mode] = {"trajectory_id": trajectory, "initial_file": "degree_initial_state.pt",
            "initial_file_sha256": accepted["degree_initial_file_sha256"], "comparison_identity": candidate,
            "experiment_purpose": purpose, "declared_intervention_fields": fields}
    spec = ExperimentSpec(declaration)
    spec.write(root / BASE / "gpu/spec.json")
    config = read(OLD + "/gpu/screen_config.json")
    config.update(spec_identity=spec.identity, arms=arms_config, platform_id="kaggle3-t4-gptrans-g1-followup",
        reference_bundle_ref=bundle_ref, reference_bundle_sha256=sha256_file(root / bundle_ref), material_gate_eV=.003)
    if terminal_reference:
        config.update(path_manifest_sha256=None, requested_kernel="nvoid912/molgap-gptrans-g1-group-path-dual-s42",
            output_subdirectory="gptrans_recipe_path_screen")
    if study:
        config.update(requested_kernel=study["kernel"], output_subdirectory=study["output_subdirectory"])
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
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "src/molgap"], cwd=root).decode().split("\0")
    sources = []
    for p in tracked:
        if p.endswith(".py") and "/archive/" not in p:
            try:
                sources.append(_name(p))
            except ValueError:
                continue
    workflow["source_paths"] = sources + [
        *recipes.values(), BASE + "/gpu/run.py", BASE + "/gpu/kernel-metadata.json", BASE + "/gpu/screen_config.json"]
    if terminal_reference:
        workflow["required_modules"].append("molgap.gptrans_endpoint_paths")
    if "degree_pair_depth_scale_ema999" in MODES:
        workflow["required_modules"].append("molgap.gptrans_pair_scale")
    atomic_json(root / BASE / "gpu/release_workflow.json", workflow)
    return {"spec_identity": spec.identity, "prelaunch_validated": True, "compute_released": False}
