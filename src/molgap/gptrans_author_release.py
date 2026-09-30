"""Freeze G1/G2 declarations; standard planner/stager own publication.

This is the question-specific evidence translation, not a new packager, model,
submission service or terminal finalizer. It does not execute models.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .comparison_readiness import ROLE_EVENT_KINDS, TRACE_FIELD_DECLARATIONS, assess_comparison_prelaunch
from .experiment_package import _name
from .experiment_spec import ExperimentSpec
from .server_acceptance import write_server_comparison_prelaunch
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, sha256_file
from .v4_runtime import normalized_source_sha256

BASE = "experiments/pcqm_gptrans_author_alignment"
GPU = BASE + "/gpu"
REFERENCE = "pcqm-gptrans-v5-audit-reference-s42"
BUNDLE = "experiments/pcqm_gptrans_v5_audit_reference/results/terminal/reference_bundle.json"
RECIPE = "experiments/pcqm_gptrans_v5_audit_reference/contract.json"
RUN = "gptrans-author-inputs-dual-s42"
MODES = ("degree_scale", "path_bond_mean")


def freeze_screen(root: Path, cpu_output: Path):
    root, cpu_output = Path(root).resolve(), Path(cpu_output).resolve()
    def read(ref):
        return json.loads((root / ref).read_text())
    accepted = read(BASE + "/verification_recovery/acceptance_v2.json")
    if (accepted.get("accepted") is not True or accepted.get("cpu_source_rederivation_verified") is not True
            or any(accepted.get(k) is not False for k in (
                "labels_read", "training_executed", "model_inference_executed",
                "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))):
        raise ValueError("Independent CPU evidence is not qualified")
    paths = accepted["path_acceptance"]
    if sha256_file(cpu_output / "paths/manifest.json") != paths["sidecar_manifest_sha256"]:
        raise ValueError("Accepted path mount changed")
    initial = root / "platforms/_records/kaggle/packages/gptrans_t_v4_source_814d104/initial_state.pt"
    degree = cpu_output / "degree_initial_state.pt"
    if sha256_file(degree) != accepted["degree_initial_file_sha256"]:
        raise ValueError("Accepted degree initial state changed")
    reference, recipe = read(BUNDLE), read(RECIPE)
    if sha256_file(initial) != recipe["initial_state_sha256"]:
        raise ValueError("Frozen reference initialization changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    implementation = normalized_source_sha256(root / "src/molgap/gptrans_author_variants.py")
    now = datetime.now(timezone.utc).isoformat()
    budget_ref, roles_ref = GPU + "/budget.json", GPU + "/role_plan.json"
    atomic_json(root / budget_ref, {"maximum_wall_hours": 6, "maximum_allocated_t4_hours": 12,
        "allocated_devices": 2, "independent_candidates": 2, "seed": 42,
        "baseline_retraining": False, "automatic_successor_authorized": False,
        "authority_ref": BASE + "/dual_arm_protocol.md"})
    role_declaration = {kind: "not_applicable" if kind == "external_submission" else "applicable"
                        for kind in ROLE_EVENT_KINDS}
    atomic_json(root / roles_ref, {"role_applicability": role_declaration,
        "training_source_idx": [0, 100000], "development_source_idx": [100000, 150000],
        "official_validation": "forbidden", "test_dev": "forbidden", "test_challenge": "forbidden"})
    def ref(name, digest):
        return {"name": name, "version": "1", "sha256": digest}
    arms, prospective, arm_config = [], [], {}
    identity = reference["comparison_identity"]
    for mode in MODES:
        trajectory = "TC-gptrans-author-" + mode.replace("_", "-") + "-100k-s42"
        arm = {"arm_id": mode, "scientific_role": "candidate", "family": {"name": "gptrans_t", "version": "1"},
            "base": ref("gptrans_t_frozen_core", recipe["architecture_sha256"]),
            "initialization": {"kind": "frozen_state", "seed": 42,
                "state_sha256": accepted["degree_initial_state_sha256"] if mode == "degree_scale" else "8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6"},
            "data": {"dataset": ref("pcqm4mv2", recipe["manifest_sha256"]),
                "split": ref("fixed100k-internal50k", recipe["manifest_sha256"]),
                "roles": [{"role": role, "membership_sha256": canonical_fingerprint({"range": bounds}),
                    "row_order_sha256": canonical_fingerprint({"range": bounds, "order": "ascending"}),
                    "usage_sha256": sha256_file(root / roles_ref)}
                    for role, bounds in (("train", [0, 100000]), ("development", [100000, 150000]))],
                "feature_schema": "ogb-atom9-bond3-shortest-path-cap20", "feature_sha256": identity["feature_identity"],
                "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": ref("pcqm_gptrans_v4", normalized_source_sha256(root / RECIPE)),
                "overrides": {}, "objective": ref("normalized-gap-l1", canonical_fingerprint({"loss": recipe["loss"]})),
                "sampler": ref("seed-plus-epoch-global-randperm-v1", canonical_fingerprint({"seed": 42, "epochs": 60})),
                "transform": ref("fixed-train-100k-mean-sample-std", identity["target_transform_asset_sha256"])},
            "addons": [{"name": mode, "version": "1", "config": {}, "source_sha256": implementation}],
            "addon_semantics": "ordered"}
        candidate_identity = {**identity, "architecture_config_identity": canonical_fingerprint({
            "core": recipe["architecture_sha256"], "variant": mode, "implementation": implementation})}
        prelaunch = assess_comparison_prelaunch(candidate_id=trajectory,
            candidate_plan={"comparison_identity": candidate_identity, "source_config_status": "frozen",
                            "source_commit_or_archive": commit}, reference_id=REFERENCE, reference_bundle=reference,
            experiment_purpose="mechanism_comparison", intervention_group_id="gptrans-author-input-attribution",
            mechanism_id=mode, declared_intervention_fields=["architecture_config_identity"],
            role_applicability_plan=role_declaration, trace_plan={k: True for k in TRACE_FIELD_DECLARATIONS},
            runtime_qualification_plan={"status": "declared", "runtime_certificate_required": True,
                                       "qualification_scope": identity["runtime_certificate_scope"]})
        folder = GPU + "/" + mode
        write_server_comparison_prelaunch(root / folder / "comparison_readiness_prelaunch.json",
            comparison_prelaunch=prelaunch, experiment_purpose="mechanism_comparison",
            reference_bundle=reference, repo_root=root, reference_bundle_path=root / BUNDLE)
        cost_id = "cost-" + trajectory
        run_id = RUN + ":" + mode
        question = ("Does reducing only the initial degree-table scale improve the unchanged GPTrans core?"
                    if mode == "degree_scale" else "Does chemical content on existing shortest paths improve the unchanged GPTrans core?")
        proposal = {"trajectory": {"trajectory_id": trajectory, "record_mode": "prospective", "owner": "server", "track": "C",
            "family_id": "gptrans-author-input-" + mode.replace("_", "-"), "question": question,
            "hypothesis": {"hypothesis_id": "H-" + trajectory,
                "observed_deficiency": "Input audit found dominant degree-table energy and absent multihop chemical edge content.",
                "supporting_evidence_ids": [REFERENCE], "alternative_explanations": [
                    "LayerNorm may erase the scale change; path-mean pooling may lose chemical sequence information."],
                "changed_mechanism": mode, "cheapest_falsifier": "one matched seed42 100K screen against the immutable accepted reference",
                "related_closed_family_ids": [], "expected_native_cost_ref": cost_id,
                "decision_changed_if_positive": "independent saved-prediction acceptance and controller review; no automatic scale-up",
                "decision_changed_if_negative": "close this exact input hypothesis without a seed or width retry",
                "historical_unknowns": ["training stochasticity is unmeasured; the 0.003 gate is not a measured variance"]},
            "state_at_start": {"source_commit": commit, "source_config_identity": canonical_fingerprint(arm),
                "contract_refs": [RECIPE, BASE + "/dual_arm_protocol.md", folder + "/comparison_readiness_prelaunch.json"],
                "reference_ids": [REFERENCE], "parent_trajectory_ids": ["TC-gptrans-v5-audit-reference-100k"],
                "prior_trajectory_ids": [], "prior_evidence_ids": [REFERENCE],
                "role_snapshot_refs": [roles_ref], "budget_snapshot_ref": budget_ref},
            "actions": [{"action_id": "A001", "type": "single_mechanism_screen", "source_commit": commit,
                "run_ids": [run_id], "attempt_ids": ["v1"], "evidence_refs": [RECIPE], "cost_event_ids": [cost_id]}],
            "result": {"evidence_ids": [], "evidence_refs": []},
            "decision": {"decision_ref": BASE + "/dual_arm_protocol.md", "outcome": "ACTIVE",
                "next_allowed_actions": ["one isolated T4 arm, then complete saved-artifact acceptance"],
                "reopen_conditions": ["terminal evidence or an infrastructure fault requiring controller diagnosis"]}},
            "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": [REFERENCE],
                "available_actions": ["RUN_AUTHOR_INPUT_SCREEN", "DEFER"], "chosen_action": "RUN_AUTHOR_INPUT_SCREEN",
                "policy_id": "gptrans-author-input-screen", "policy_version": "v1",
                "budget_snapshot_ref": budget_ref, "role_snapshot_refs": [roles_ref], "state_timestamp": now, "source_commit": commit},
            "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cost_id, "trajectory_id": trajectory,
                "action_id": "A001", "run_id": run_id, "attempt_id": "v1", "category": "training",
                "platform": "kaggle2", "hardware": "one-of-two-independent-T4", "evidence_ref": budget_ref,
                "measurement": {"device_hours": {"status": "estimated", "value": 6},
                    "wall_hours": {"status": "estimated", "value": 6},
                    "cpu_hours": {"status": "measurement_missing", "value": None},
                    "queue_hours": {"status": "measurement_missing", "value": None}}}]}
        plan_ref = folder + "/plan_input.json"
        atomic_json(root / plan_ref, proposal)
        prospective.append({"arm_id": mode, "trajectory_id": trajectory, "plan_spec_ref": plan_ref,
                            "plan_spec_sha256": sha256_file(root / plan_ref), "output": folder + "/rml_plan"})
        arms.append(arm)
        arm_config[mode] = {"trajectory_id": trajectory,
            "initial_file": "degree_initial_state.pt" if mode == "degree_scale" else "initial_state.pt",
            "initial_file_sha256": accepted["degree_initial_file_sha256"] if mode == "degree_scale" else recipe["initial_state_sha256"]}
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": "gptrans-author-input-attribution",
        "logical_run_id": RUN, "arms": arms, "prospective": {"arms": prospective},
        "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2,
            "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "evidence": {"policy": ref("molgap-v5", sha256_file(root / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")),
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": "molgap-experiment-terminal-descriptor-v1"})
    atomic_json(root / GPU / "spec.json", spec.to_dict())
    transform = root / BASE / "recovered_reference/target_transform.json"
    atomic_json(root / GPU / "screen_config.json", {"spec_identity": spec.identity,
        "arms": arm_config, "dataset_manifest_sha256": recipe["manifest_sha256"],
        "path_manifest_sha256": paths["sidecar_manifest_sha256"],
        "target_transform_sha256": identity["target_transform_asset_sha256"],
        "cpu_acceptance_ref": BASE + "/verification_recovery/acceptance_v2.json",
        "cpu_acceptance_sha256": sha256_file(root / BASE / "verification_recovery/acceptance_v2.json"),
        "maximum_wall_seconds": 21600, "training_estimate_cap_hours": 4.5,
        "recording_owner": "pcqm_gptrans_v4.native_v5", "family_output_hooks_enabled": False,
        "reference_bundle_ref": BUNDLE, "reference_bundle_sha256": sha256_file(root / BUNDLE)})
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "src/molgap"], cwd=root).split(b"\0")
    sources = []
    for raw in tracked:
        if not raw.endswith(b".py") or b"/archive/" in raw:
            continue
        try:
            sources.append(_name(raw.decode()))
        except ValueError:
            continue
    # Generated question files are committed before prepare-release freezes them.
    for file in (RECIPE, GPU + "/run.py", GPU + "/kernel-metadata.json", GPU + "/screen_config.json"):
        sources.append(file)
    workflow = {"format": "molgap-release-workflow-v1", "spec_identity": spec.identity,
        "source_paths": sources, "artifacts": {
            "initial_state.pt": {"path": str(initial), "sha256": recipe["initial_state_sha256"]},
            "degree_initial_state.pt": {"path": str(degree), "sha256": accepted["degree_initial_file_sha256"]},
            "target_transform.json": {"path": str(transform), "sha256": identity["target_transform_asset_sha256"]}},
        "recipe_files": {mode: RECIPE for mode in MODES},
        "initial_states": {mode: arm_config[mode]["initial_file"] for mode in MODES},
        "required_modules": ["molgap.gptrans_author_screen", "molgap.gptrans_screen_adapter", "molgap.pcqm_gptrans_v4"],
        "entry_template": GPU + "/run.py", "kernel_metadata": GPU + "/kernel-metadata.json",
        "dataset_metadata": {"title": "MolGap GPTrans Author Dual Source", "id": "kaseichou/molgap-gptrans-author-dual-source",
                             "licenses": [{"name": "other"}], "isPrivate": True}, "pickle_inputs": []}
    atomic_json(root / GPU / "release_workflow.json", workflow)
    return {"spec_identity": spec.identity, "arms": list(MODES), "compute_released": False}
