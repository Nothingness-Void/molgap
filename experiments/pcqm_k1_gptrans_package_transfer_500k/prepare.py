"""Freeze declarations, then reuse shared prospective/package/release owners."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from molgap.experiment_package import build_experiment_source_package, verify_experiment_source_package
from molgap.experiment_preflight import check_release_inputs
from molgap.experiment_prospective import plan_prospective
from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
from molgap.experiment_spec import ExperimentSpec, FAMILIES
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
REL = EXP.relative_to(ROOT).as_posix()
RUN = "molgap-k1-gptrans-500k-pair-s42-v1"
POLICY = "pcqm-k1-gptrans-package-transfer-500k-launch"
ARMS = ("k1_pretrained_consistency", "gptrans_g1_bond_local_ema999")
MANIFEST = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
STAGING = ROOT / "platforms/_records/kaggle/staging/pcqm_k1_gptrans_package_transfer_500k"
TRUSTED_PICKLE = Path("D:/文档/molgap-exp/molgap-500k-v4-evidence/data/cache/pcqm4mv2_500k_v4/train/train_shard_0000.pt")
REQUIRED_MODULES = ["molgap.pcqm_500k_v4_evidence", "molgap.pcqm_composed_500k", "molgap.qm9_neural_atom",
                    "molgap.gptrans_capacity", "molgap.k1_pretrained_combo", "molgap.k1_screen_training", "molgap.pcqm_wedge"]
# Explicit reviewed legacy-loader and composed-model dependencies, in addition
# to the shared inventory; this is not a whole-src archive or import discovery.
EXTRA_SOURCE = (
    "platforms/kaggle/run_legacy_500k_pair.py", "src/molgap/pcqm_500k_v4_evidence.py",
    "src/molgap/pcqm_composed_500k.py", "src/molgap/pcqm_k1_scale_runner.py",
    "src/molgap/pcqm_k1_scale.py",
    "src/molgap/futility_gate.py", "src/molgap/k1_pretrained_combo.py",
    "src/molgap/gptrans_capacity.py",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False), encoding="utf-8", newline="")


def ref(name, value):
    return {"name": name, "version": "1", "sha256": canonical_fingerprint(value)}


def pinned(inputs, entry):
    path = inputs / entry["file"]
    if not path.resolve().is_relative_to(inputs.resolve()) or not path.is_file() or sha256_file(path) != entry["sha256"]:
        raise ValueError("Missing, escaped or changed independently pinned input: " + str(path))
    return path


def declare(source_commit, inputs):
    from molgap.pcqm_composed_500k import scientific_contract
    pins = read(inputs / "initial_pins.json")
    initial = {aid: pinned(inputs, pins["arms"][aid]) for aid in ARMS}
    transform = pinned(inputs, pins["target_transform"])
    write(EXP / "role_plan.json", {
        "train": "source_idx[0,500000): supervised training, train-only GPU qualification; no teacher",
        "development": "source_idx[500000,550000): selection and fixed50:50 mixture analysis; existing development history",
        "official_validation": "untouched", "test_dev": "untouched", "test_challenge": "untouched",
        "manifest_sha256": MANIFEST,
        "authority": "Desktop user approved the teacher-free two-arm500K60-pass question on2026-10-05; see plan.md",
    })
    write(EXP / "budget.json", {"training_allocated_T4_hours_cap": 48, "max_stage_seconds": 32400,
        "stage_epochs": 60, "qualification_cost": "measured separately; unknown before execution",
        "resume": "Only the same frozen question with verified independently retained checkpoints; no automatic successor",
        "estimates": {ARMS[0]: 18.69, ARMS[1]: 18.80}, "estimate_unit": "T4 hours; rough scaling, not measured"})
    policy = read(ROOT / "research_memory/policies/pcqm-k1-dropout-consistency-kaggle3-100k-launch.1.json")
    policy.update(policy_id=POLICY, created_from_source_digest=sha256_file(EXP / "plan.md"),
        comparability_selector={"scientific_contract": "pcqm-k1-gptrans-package-transfer-500k-v1"},
        action_rule={"action": "QUALIFY_THEN_TRAIN_PAIR_500K", "field": "evidence_review_complete", "operator": "eq", "threshold": 1})
    write(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    arms, bindings = [], []
    for device, aid in enumerate(ARMS):
        family = ("neural_atom_k1", "3") if device == 0 else ("gptrans_t", "2")
        owner = FAMILIES[family]
        recipe = scientific_contract(aid)
        recipe.update(initial_state_sha256=pins["arms"][aid]["sha256"],
            target_transform_artifact_sha256=None if device == 0 else pins["target_transform"]["sha256"],
            source_module_sha256=normalized_source_sha256(ROOT / "src/molgap/pcqm_composed_500k.py"))
        recipe_path = f"{REL}/training_recipe_{aid}.json"
        write(ROOT / recipe_path, recipe)
        roles = []
        for role, start, stop, usage in (("train", 0, 500000, "supervision-and-train-only-qualification"), ("development", 500000, 550000, "selection-and-fixed-equal-fusion")):
            interval = {"source_idx_start": start, "source_idx_stop": stop, "manifest_sha256": MANIFEST}
            roles.append({"role": role, "membership_sha256": canonical_fingerprint(interval),
                "row_order_sha256": canonical_fingerprint({**interval, "order": "ascending-source_idx"}),
                "usage_sha256": canonical_fingerprint({"role": role, "usage": usage, "role_plan_sha256": sha256_file(EXP / "role_plan.json")})})
        arm = {"arm_id": aid, "scientific_role": "candidate", "family": {"name": family[0], "version": family[1]},
            "base": ref("composed-500k-source", {"source_module_sha256": recipe["source_module_sha256"], "arm": aid}),
            "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": pins["arms"][aid]["state_sha256"]},
            "data": {"dataset": {"name": "pcqm4mv2", "version": "1", "sha256": MANIFEST},
                "split": ref("accepted-fixed-500k-train-development", {"train": [0, 500000], "development": [500000, 550000], "manifest_sha256": MANIFEST}),
                "roles": roles, "feature_schema": owner.feature_schema, "feature_sha256": MANIFEST, "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": {"name": owner.recipe, "version": "1", "sha256": sha256_file(ROOT / recipe_path)}, "overrides": {},
                "objective": ref("normalized-gap-l1-dropout-consistency" if device == 0 else "normalized-gap-l1", {"arm": aid, "contract": recipe}),
                "sampler": ref(owner.sampler, {"seed": 42, "epochs": 60, "batch": 128, "drop_last": 32, "steps": 234360, "presentations": 29998080}),
                "transform": ref(owner.transform, {"manifest": MANIFEST} if device == 0 else {"artifact_sha256": pins["target_transform"]["sha256"]})},
            "addons": [], "addon_semantics": "baseline"}
        arms.append(arm)
        tid = "TB-" + aid.replace("_", "-") + "-kaggle1-500k-s42-v1"
        cid, run = "cost-" + tid + "-expected-training", RUN + ":" + aid
        action_path = f"{REL}/action_inputs_{aid}.json"
        write(ROOT / action_path, {"trajectory_id": tid, "state_timestamp": "2026-10-05", "evidence_review_complete": 1, "evidence_ids": ["pcqm-matched-500k-v4-three-arm"]})
        state = {"source_commit": source_commit, "source_config_identity": canonical_fingerprint(arm),
            "prior_evidence_ids": ["pcqm-matched-500k-v4-three-arm"], "parent_trajectory_ids": ["TB-matched-500k-v4-three-arm"], "reference_ids": [],
            "budget_snapshot_ref": f"{REL}/budget.json", "role_snapshot_refs": [f"{REL}/role_plan.json"],
            "contract_refs": [f"{REL}/protocol.md", f"{REL}/plan.md", recipe_path, f"{REL}/role_plan.json", f"{REL}/budget.json"]}
        plan = {"action_inputs_ref": action_path, "decision_state": {
            "known_trajectory_ids": ["TB-matched-500k-v4-three-arm"], "known_evidence_ids": ["pcqm-matched-500k-v4-three-arm"], "active_reference_ids": [],
            "available_actions": ["QUALIFY_THEN_TRAIN_PAIR_500K", "NO_TRAIN"], "chosen_action": "QUALIFY_THEN_TRAIN_PAIR_500K",
            "policy_id": POLICY, "policy_version": "1", "budget_snapshot_ref": state["budget_snapshot_ref"],
            "role_snapshot_refs": state["role_snapshot_refs"], "source_commit": source_commit, "state_timestamp": "2026-10-05"},
            "trajectory": {"schema": "molgap-trajectory-v1", "record_mode": "prospective", "owner": "desktop", "track": "B",
                "trajectory_id": tid, "family_id": "k1-gptrans-package-transfer-500k", "question": "Does the exact " + aid + " composition transfer to500K/60passes and complement the paired recipe?",
                "state_at_start": state, "hypothesis": {"hypothesis_id": "H-" + tid,
                    "observed_deficiency": "Selected100K composition lacks a matched500K60-pass transfer endpoint; historical500K comparisons remain contextual.",
                    "changed_mechanism": "Exact composed recipe at500K/60passes; no teacher, architecture or fusion weight search.",
                    "alternative_explanations": ["Native recipe and initialization differences", "Development selection reuse", "Single-seed training variation"],
                    "cheapest_falsifier": "CPU release and all-arm training-only T4 deterministic optimizer qualification before any training arm.",
                    "decision_changed_if_positive": "Review fixed50:50 gain>=1meV with positive paired-row95% lower bound; no automatic adoption or official evaluation.",
                    "decision_changed_if_negative": "Close with retained exposure/objective/curve/cost attribution; no automatic sweep.",
                    "supporting_evidence_ids": ["pcqm-matched-500k-v4-three-arm"], "related_closed_family_ids": ["k1-dropout-consistency", "gptrans-capacity-relations"],
                    "historical_unknowns": ["Training stochasticity", "Strict cross-contract historical qualification", "Historical pretraining native costs"],
                    "expected_native_cost_ref": cid},
                "actions": [{"action_id": "A001", "type": "authorized_teacher_free_pair_500k", "attempt_ids": ["kaggle1-package-transfer-001"],
                    "run_ids": [run], "cost_event_ids": [cid], "evidence_refs": [f"{REL}/protocol.md", f"{REL}/plan.md"], "source_commit": source_commit}],
                "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/protocol.md", "next_allowed_actions": ["A001"], "reopen_conditions": []},
                "result": {"evidence_ids": [], "evidence_refs": []}},
            "costs": [{"schema": "molgap-cost-event-v1", "trajectory_id": tid, "cost_event_id": cid, "action_id": "A001",
                "attempt_id": "kaggle1-package-transfer-001", "run_id": run, "category": "training", "platform": "kaggle",
                "hardware": f"Tesla T4; assignedGPU{device} in2T4 pair", "evidence_ref": f"{REL}/budget.json",
                "measurement": {"device_hours": {"status": "estimated", "value": 18.69 if device == 0 else 18.80},
                    "wall_hours": {"status": "estimated", "value": 18.69 if device == 0 else 18.80},
                    "cpu_hours": {"status": "measurement_missing", "value": None}, "queue_hours": {"status": "measurement_missing", "value": None}}}]}
        plan_path = f"{REL}/training_plan_{aid}.json"
        write(ROOT / plan_path, plan)
        bindings.append({"arm_id": aid, "trajectory_id": tid, "plan_spec_ref": plan_path, "plan_spec_sha256": sha256_file(ROOT / plan_path), "output": f"{REL}/kaggle1_v1/{aid}"})
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": "pcqm-k1-gptrans-package-transfer-500k", "logical_run_id": RUN,
        "arms": arms, "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2, "cpu_cores": 4, "memory_gib": 32, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings}, "evidence": {"policy": ref("molgap-v5", {"common": sha256_file(ROOT / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")}),
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": "molgap-experiment-terminal-descriptor-v1"})
    write(EXP / "experiment_spec.json", spec.to_dict())
    return spec, pins, initial, transform


def prepare_continuation(previous, resume_root, output, pickle_input):
    """Keep prospective/scientific inputs frozen while rebinding executable source."""
    import torch
    previous, resume_root, output = map(Path, (previous, resume_root, output))
    old = verify_experiment_source_package(previous / "package")
    spec = ExperimentSpec.from_json((previous / "package/experiment_spec.json").read_text())
    if spec.identity != ExperimentSpec.from_json((EXP / "experiment_spec.json").read_text()).identity:
        raise ValueError("Continuation cannot change the owning frozen Spec")
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_experiment_source_package(spec, ROOT, old["relative_allowlist"], output / "package")
    source, kernel, checkpoints = output / "source_dataset", output / "kernel", output / "checkpoint_dataset"
    shutil.copytree(previous / "source_dataset", source)
    shutil.copytree(previous / "kernel", kernel)
    checkpoints.mkdir()
    checkpoint_slug = "molgap-k1-gptrans-500k-resume-v2-s42"
    checkpoint_id = "nothingnessvoid/" + checkpoint_slug
    launch = read(source / "legacy_500k_launch.json")
    launch.update(source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"],
                  package_identity=manifest["package_identity"])
    launch["dataset_sources"] = list(launch["dataset_sources"]) + [checkpoint_id]
    distributions = None
    resume_summary = {}
    for arm in launch["arms"]:
        aid = arm["arm_id"]
        prior = resume_root / aid
        stage = read(prior / "stage_manifest.json")
        if stage["arm"] != aid or stage["source_sha256"] != old["archive_sha256"]:
            raise ValueError("Continuation must use this exact previous source/arm")
        destination = checkpoints / aid
        destination.mkdir()
        for name, checksum in stage["artifacts"].items():
            path = prior / name
            if not path.resolve().is_relative_to(prior.resolve()) or sha256_file(path) != checksum:
                raise ValueError("Retained resume artifact changed: " + name)
            shutil.copyfile(path, destination / name)
        shutil.copyfile(prior / "stage_manifest.json", destination / "stage_manifest.json")
        state = torch.load(prior / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        runtime = read(prior / "runtime.json")
        if (state["arm"] != aid or state["source_sha256"] != old["archive_sha256"]
                or state["next_epoch"] != stage["next_epoch"] or len(state["trace"]) != state["next_epoch"]
                or state["next_batch_index"] != 0 or state["global_step"] != 3906 * state["next_epoch"]
                or state["next_schedule_epoch"] != state["next_epoch"]
                or state["runtime_software"] != runtime["installed_distributions_sha256"]
                or state["contract"] != stage["contract"]):
            raise ValueError("Retained checkpoint cursor/scientific/runtime mismatch")
        if distributions is not None and distributions != runtime["installed_distributions"]:
            raise ValueError("Paired continuation runtimes differ")
        distributions = runtime["installed_distributions"]
        arm["resume"] = {"mount": checkpoint_slug, "manifest_sha256": sha256_file(prior / "stage_manifest.json"),
                         "source_sha256": old["archive_sha256"], "next_epoch": state["next_epoch"]}
        resume_summary[aid] = dict(arm["resume"], checkpoint_sha256=sha256_file(prior / "last_checkpoint.pt"),
                                  global_step=state["global_step"], runtime_software=state["runtime_software"])
    launch["runtime_distributions"] = distributions
    for path in (output / "package").iterdir():
        shutil.copyfile(path, source / ("source_payload.bin" if path.name == "source.tar.gz" else path.name))
    write(source / "legacy_500k_launch.json", launch)
    entry = (ROOT / "platforms/kaggle/run_legacy_500k_pair.py").read_text(encoding="utf-8").replace(
        "EXPECTED_LAUNCH_SHA256 = None", "EXPECTED_LAUNCH_SHA256 = " + repr(sha256_file(source / "legacy_500k_launch.json")))
    (kernel / "run.py").write_text(entry, encoding="utf-8", newline="\n")
    metadata = read(kernel / "kernel-metadata.json")
    metadata["dataset_sources"] = launch["dataset_sources"]
    write(kernel / "kernel-metadata.json", metadata)
    write(checkpoints / "dataset-metadata.json", {"id": checkpoint_id, "title": "MolGap K1 GPTrans 500K Resume V2 S42",
                                                "licenses": [{"name": "other"}], "isPrivate": True})
    write(output / "continuation_binding.json", {"prior_package_identity": old["package_identity"],
        "prior_source_sha256": old["archive_sha256"], "spec_identity": spec.identity,
        "resume_dataset": checkpoint_id, "arms": resume_summary,
        "prospective_preserved": launch["prospective_sha256"]})
    report = check_release_inputs(spec, output / "package", expected_package_identity=manifest["package_identity"],
        recipe_files={a["arm_id"]: a["recipe"] for a in launch["arms"]},
        initial_states={a["arm_id"]: source / a["initial_state"] for a in launch["arms"]},
        required_modules=launch["required_modules"], pickle_inputs=[pickle_input],
        entry_script=kernel / "run.py", input_root=source, launch_config=source / "legacy_500k_launch.json",
        kernel_metadata=kernel / "kernel-metadata.json")
    write(output / "release_report.json", report)
    if report["errors"]:
        raise RuntimeError("Continuation local release failed")
    print(json.dumps({"status": report["status"], "package_identity": manifest["package_identity"], "resume": resume_summary}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--inputs", type=Path, default=STAGING / "inputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--resume-prepared", action="store_true",
                        help="Verify an existing frozen package and published records after RML rebuild recovery")
    parser.add_argument("--pickle-input", type=Path, default=TRUSTED_PICKLE)
    parser.add_argument("--continuation-from", type=Path)
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.source_commit != actual:
        raise ValueError("Source commit must equal this checkout's frozen HEAD")
    if args.continuation_from is not None:
        if args.resume_root is None or args.output is None:
            raise ValueError("Continuation requires --resume-root and a fresh --output")
        prepare_continuation(args.continuation_from, args.resume_root, args.output, args.pickle_input)
        return
    spec, pins, initial, transform = declare(actual, args.inputs)
    if args.declare_only:
        print(json.dumps({"status": "DECLARATIONS_PREPARED", "spec_identity": spec.identity}))
        return
    if args.output is None or (args.output.exists() and not args.resume_prepared):
        raise ValueError("Provide a fresh --output directory")
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume_prepared)
    recipes = {aid: f"{REL}/training_recipe_{aid}.json" for aid in ARMS}
    names = sorted(set(SHARED_SOURCE_FILES) | set(EXTRA_SOURCE) | set(recipes.values()) | {f"{REL}/protocol.md", f"{REL}/plan.md", f"{REL}/role_plan.json", f"{REL}/budget.json"})
    manifest = (verify_experiment_source_package(output / "package", repo_root=ROOT)
                if args.resume_prepared else build_experiment_source_package(spec, ROOT, names, output / "package"))
    if manifest["source_commit"] != actual:
        raise ValueError("Package source commit changed")
    if args.resume_prepared:
        if manifest["spec_identity"] != spec.identity:
            raise ValueError("Recovered package Spec changed")
        for arm, binding in zip(spec.to_dict()["arms"], spec.to_dict()["prospective"]["arms"]):
            trajectory = read(ROOT / binding["output"] / "trajectory.json")
            if (trajectory["record_mode"] != "prospective" or trajectory["trajectory_id"] != binding["trajectory_id"]
                    or trajectory["state_at_start"]["source_config_identity"] != canonical_fingerprint(arm)
                    or trajectory["state_at_start"]["source_commit"] != actual):
                raise ValueError("Published prospective identity changed")
        from molgap.research_memory.compiler import rebuild_research_memory
        rebuild_research_memory(ROOT)
        report, code = {"status": "EXISTING_PROSPECTIVE_VERIFIED_RML_REBUILT", "spec_identity": spec.identity}, 0
    else:
        report, code = plan_prospective(spec, ROOT)
    write(output / "prospective_report.json", report)
    if code:
        raise RuntimeError("Prospective publication requires reconciliation: " + report["status"])
    source, kernel = output / "source_dataset", output / "kernel"
    source.mkdir()
    kernel.mkdir()
    for path in (output / "package").iterdir():
        shutil.copyfile(path, source / ("source_payload.bin" if path.name == "source.tar.gz" else path.name))
    launch_arms = []
    for device, (arm, prospective) in enumerate(zip(spec.to_dict()["arms"], spec.to_dict()["prospective"]["arms"])):
        aid = arm["arm_id"]
        init_name = "inputs/" + pins["arms"][aid]["file"]
        destination = source / init_name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(initial[aid], destination)
        binding_name, trajectory_name = f"bindings/{aid}.json", f"prospective/{aid}/trajectory.json"
        write(source / binding_name, {"spec_identity": spec.identity, "trajectory_id": prospective["trajectory_id"], "source_config_identity": canonical_fingerprint(arm)})
        (source / trajectory_name).parent.mkdir(parents=True)
        shutil.copyfile(ROOT / prospective["output"] / "trajectory.json", source / trajectory_name)
        launch_arms.append({"arm_id": aid, "device": device, "recipe": recipes[aid], "source_config_identity": canonical_fingerprint(arm),
            "initial_state": init_name, "initial_state_sha256": pins["arms"][aid]["sha256"], "binding": binding_name,
            "binding_sha256": sha256_file(source / binding_name), "trajectory": trajectory_name, "trajectory_sha256": sha256_file(source / trajectory_name), "trajectory_id": prospective["trajectory_id"]})
    shutil.copyfile(transform, source / "inputs/target_transform.json")
    shutil.copyfile(args.inputs / "initial_pins.json", source / "inputs/initial_pins.json")
    launch = {"format": "molgap-legacy-500k-pair-v1", "spec_identity": spec.identity, "package_identity": manifest["package_identity"],
        "required_modules": REQUIRED_MODULES,
        "run_reference": "nothingnessvoid/" + RUN, "account": "nothingnessvoid", "accelerator": "NvidiaTeslaT4", "device_count": 2,
        "dataset_sources": ["nothingnessvoid/molgap-k1-gptrans-500k-source-s42-v1", "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1"],
        "source_mount": "molgap-k1-gptrans-500k-source-s42-v1", "graph_mount": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
        "prospective_sha256": {a["arm_id"]: a["trajectory_sha256"] for a in launch_arms},
        "source_commit": actual, "source_archive_sha256": manifest["archive_sha256"], "stage_epochs": 60, "max_stage_seconds": 32400,
        "target_transform": "inputs/target_transform.json", "target_transform_sha256": pins["target_transform"]["sha256"], "arms": launch_arms}
    write(source / "legacy_500k_launch.json", launch)
    entry = (ROOT / "platforms/kaggle/run_legacy_500k_pair.py").read_text(encoding="utf-8").replace("EXPECTED_LAUNCH_SHA256 = None", "EXPECTED_LAUNCH_SHA256 = " + repr(sha256_file(source / "legacy_500k_launch.json")))
    (kernel / "run.py").write_text(entry, encoding="utf-8", newline="\n")
    write(source / "dataset-metadata.json", {"id": "nothingnessvoid/molgap-k1-gptrans-500k-source-s42-v1", "title": "MolGap K1 GPTrans 500K Source S42 V1", "licenses": [{"name": "other"}], "isPrivate": True})
    write(kernel / "kernel-metadata.json", {"id": "nothingnessvoid/" + RUN, "title": "MolGap K1 GPTrans 500K Pair S42 V1", "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": launch["dataset_sources"], "competition_sources": [], "kernel_sources": [], "model_sources": []})
    report = check_release_inputs(spec, output / "package", expected_package_identity=manifest["package_identity"], recipe_files=recipes,
        initial_states={a["arm_id"]: source / a["initial_state"] for a in launch_arms}, required_modules=REQUIRED_MODULES,
        pickle_inputs=[args.pickle_input], entry_script=kernel / "run.py", input_root=source, launch_config=source / "legacy_500k_launch.json", kernel_metadata=kernel / "kernel-metadata.json")
    write(output / "legacy_500k_binding.json", {"launch_sha256": sha256_file(source / "legacy_500k_launch.json"), "entry_sha256": sha256_file(kernel / "run.py"), "metadata_sha256": sha256_file(kernel / "kernel-metadata.json"), "prospective": {a["arm_id"]: a["trajectory_sha256"] for a in launch_arms}, "workflow_registry": "UNSUPPORTED; owning500Klegacy adapter"})
    write(output / "release_report.json", report)
    if report["errors"]:
        raise RuntimeError("Local release inputs failed; inspect release_report.json")
    print(json.dumps({"status": "LOCAL_LEGACY500K_PAYLOAD_PREPARED", "package_identity": manifest["package_identity"], "spec_identity": spec.identity, "output": str(output)}))


if __name__ == "__main__":
    main()
