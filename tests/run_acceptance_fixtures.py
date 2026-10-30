"""Frozen synthetic two-arm run shared by local lifecycle integration tests."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

from molgap.common.paths import MachinePaths
from molgap.experiment_launch import canonical_json
from molgap.experiment_package import build_experiment_source_package
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from test_experiment_launch import build, envelope, known
from test_experiment_workflow import _candidate_pair_spec


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode())


def frozen_acceptance_run(tmp_path, repo):
    profile = tmp_path / "machine-acceptance.json"
    write(profile, {"format": "molgap-machine-paths-v1", "python": sys.executable,
        **{name: str(tmp_path / name) for name in ("data_root", "artifact_root", "scratch_root")}})
    paths = MachinePaths.from_file(profile)
    declaration = _candidate_pair_spec().to_dict()
    config_source = repo / "experiments/synthetic/config.json"
    config_source.parent.mkdir(parents=True, exist_ok=True)
    config_bytes = canonical_json({"format": "molgap-scientific-experiment-v1",
        "experiment_id": declaration["experiment_id"], "question": "synthetic acceptance fixture"}).encode()
    config_source.write_bytes(config_bytes)
    subprocess.run(["git", "add", "experiments/synthetic/config.json"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic acceptance config"],
                   cwd=repo, check=True, capture_output=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    generated_root = "experiments/" + declaration["experiment_id"] + "/generated/" + canonical_fingerprint({
        "input": hashlib.sha256(config_bytes).hexdigest(), "commit": source})
    for binding in declaration["prospective"]["arms"]:
        binding["trajectory_id"] = "trajectory-" + binding["arm_id"].replace("_", "-")
        binding["plan_spec_ref"] = generated_root + "/" + binding["arm_id"] + "-plan.json"
    generated, entries = {}, []
    def add(name, value):
        raw = canonical_json(value).encode()
        generated[name] = raw
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
    expected = {"epochs": 2, "optimizer_steps": 4, "sample_presentations": 8,
                "development_rows": 3,
                "source_idx_sha256": hashlib.sha256(struct.pack("<3q", 101, 103, 108)).hexdigest(),
                "target_sha256": hashlib.sha256(b"\0" * 12).hexdigest(), "precision": "fp32"}
    target = add(generated_root + "/target.json", {"development_target_sha256": expected["target_sha256"],
                                      "target_encoding": "little-endian-float32-contiguous-raw-bytes"})
    for arm, binding in zip(declaration["arms"], declaration["prospective"]["arms"]):
        arm_id = arm["arm_id"]
        contract = add(generated_root + "/" + arm_id + "-recipe.json", {
            "acceptance_requirements": expected, "development_role_identity": "development"})
        arm["training"]["recipe"]["sha256"] = contract["sha256"]
        run_key = declaration["logical_run_id"] + ":" + arm_id + ":downstream"
        proposal = {"trajectory": {"trajectory_id": binding["trajectory_id"],
            "state_at_start": {"source_commit": source, "source_config_identity": canonical_fingerprint(arm)},
            "actions": [{"action_id": "unrelated", "run_ids": ["unrelated"], "source_commit": source},
                {"action_id": "matching", "run_ids": [run_key], "attempt_ids": [arm_id + "-v7"],
                 "source_commit": source}]}}
        binding["plan_spec_sha256"] = add(binding["plan_spec_ref"], proposal)["sha256"]
        entries.append({"arm_id": arm_id, "adapter": "k1-screen-v1" if arm_id == "k1_candidate" else "gptrans-v1",
                        "expected": expected, "contract": contract,
                        "reference_artifacts": {"target_manifest": target}})
    spec = ExperimentSpec(declaration)
    plan_name = generated_root + "/acceptance.json"
    add(plan_name, {"format": "molgap-family-acceptance-plan-v1", "spec_identity": spec.identity, "arms": entries})
    workflow = {"spec_identity": spec.identity, "acceptance_plan": plan_name,
                "platform": {"source_dataset": "synthetic-owner/source"}}
    run = paths.run_directory(declaration["logical_run_id"], "v1")
    package = run / "prepared/package"
    manifest = build_experiment_source_package(spec, repo,
        ["src/example.py", "README.md", "experiments/synthetic/config.json", *generated], package, generated_records=generated,
        configuration_source={"path": "experiments/synthetic/config.json", "sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
                              "compiler": "molgap-scientific-experiment-v1"})
    write(run / "inputs/experiment_spec.json", spec.to_dict())
    write(run / "inputs/workflow_plan.json", workflow)
    binding = {"format": "molgap-experiment-input-v1", "input_identity": "f" * 64,
        "spec_identity": spec.identity, "workflow_identity": canonical_fingerprint(workflow),
        "authority_ref": "authority.json", "monitor_paths": ["experiments/question/STATUS.md"],
        "source_commit": source}
    write(run / "inputs/input_binding.json", binding)
    write(run / "prepared/workflow_report.json", {"status": "PREPARED_FOR_PLATFORM",
        "spec_identity": spec.identity, "package_identity": manifest["package_identity"], "source_commit": source})
    context = spec, package, manifest["package_identity"]
    response = envelope(context)
    response["platform_version"] = known("7")
    response["physical_runs"] = known([{"arm_ids": list(a["arm_id"] for a in declaration["arms"]),
        "run_identity": "kernel1/version7", "canonical_reference": "synthetic-owner/workload",
        "platform_version": known("7")}])
    receipt = build(context, response)
    write(run / "operations/launch_receipt.json", receipt)
    write(run / "operations/submission_state.json", {"phase": "BOUND", "binding": {
        "package_identity": manifest["package_identity"], "kernel": "synthetic-owner/workload",
        "dataset": "synthetic-owner/source", "authority_ref": binding["authority_ref"],
        "monitor_paths": binding["monitor_paths"]},
        "kernel_response": {"kernel": "synthetic-owner/workload", "version_number": 7}})
    return SimpleNamespace(profile=profile, paths=paths, repo=repo, run=run, prepared=run / "prepared",
        package=package, spec=spec, manifest=manifest, receipt=receipt, expected=expected,
        generated=generated, original=(profile, repo, declaration["logical_run_id"], "v1"))


def frozen_shared_acceptance_run(tmp_path, repo, *, retain_training_target=True,
                                 comparison_purpose="architecture_comparison"):
    """Freeze current shared-family contracts for the joined diagnostic path."""
    from molgap.experiment_package import build_experiment_source_package
    from molgap.experiment_spec import ExperimentSpec
    from test_experiment_launch import build as build_receipt, envelope, known
    from test_experiment_workflow import _arm, _spec_payload
    from test_family_training_runner import contract_file
    from test_family_training_session import events, make_session
    import torch

    profile = tmp_path / "machine-shared-acceptance.json"
    write(profile, {"format": "molgap-machine-paths-v1", "python": sys.executable,
        **{name: str(tmp_path / name) for name in ("data_root", "artifact_root", "scratch_root")}})
    paths = MachinePaths.from_file(profile)
    arm_specs, contracts = [], {}
    training_target_sha256 = None
    for family, version, arm_id in (
            ("neural_atom_k1", "3", "k1_candidate"),
            ("gptrans_t", "2", "gptrans_candidate")):
        if comparison_purpose in {"initialization_comparison", "single_module_ablation"}:
            family, version = "neural_atom_k1", "3"
        session = make_session(family, development=True, schedule_kind="cosine_step", events=events())
        if comparison_purpose == "initialization_comparison" and arm_id == "gptrans_candidate":
            from dataclasses import replace
            session.recipe = replace(session.recipe, initialization_sha256="3" * 64)
        if comparison_purpose == "single_module_ablation" and arm_id == "gptrans_candidate":
            from dataclasses import replace
            from molgap.experiment_execution import build_addon_declaration
            addon = build_addon_declaration((family, version), "k1_readout_scale",
                repo_root=Path(__file__).resolve().parents[1])
            session.recipe = replace(session.recipe, mode="composed", addons=(addon,))
        try:
            contract_root = tmp_path / "native-contracts" / arm_id
            contract_root.mkdir(parents=True)
            contract_path, record = contract_file(contract_root, session)
            training_targets = torch.cat([graph.y for graph in session.loaders.data.train])
            training_target_sha256 = hashlib.sha256(
                training_targets.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
            arm = _arm(family, version, arm_id,
                recipe_sha256=hashlib.sha256(canonical_json(record).encode()).hexdigest(),
                state_sha256=session.recipe.initialization_sha256)
            if comparison_purpose == "single_module_ablation":
                from molgap.family_training_runner import family_arm_bindings
                arm["training"]["objective"] = family_arm_bindings(session.recipe)["objective"]
                arm["addons"] = list(session.recipe.addons)
                arm["addon_semantics"] = "ordered" if arm["addons"] else "baseline"
            arm["data"]["dataset"]["sha256"] = session.recipe.dataset.manifest_sha256
            roles = {item["role"]: item for item in arm["data"]["roles"]}
            roles["train"]["usage_sha256"] = record["train_role_identity"]
            roles.setdefault("development", {"role": "development", "membership_sha256": "a" * 64,
                "row_order_sha256": "b" * 64, "usage_sha256": record["development_role_identity"]})
            roles["development"]["usage_sha256"] = record["development_role_identity"]
            # Real Specs freeze the development membership as the contract's source-index digest.
            roles["development"]["membership_sha256"] = roles["development"]["row_order_sha256"] = (
                record["acceptance_requirements"]["source_idx_sha256"])
            arm["data"]["roles"] = [roles["train"], roles["development"]]
            arm_specs.append(arm)
            contracts[arm_id] = (record, canonical_json(record).encode("utf-8"))
        finally:
            session.close()

    arm_specs[0]["scientific_role"] = "reference"
    declaration = _spec_payload(arm_specs, platform="kaggle")
    declaration["prospective"]["same_run_replay"] = {
        "reference_arm_id": "k1_candidate", "candidate_arm_ids": ["gptrans_candidate"]}
    config_source = repo / "experiments/synthetic/config.json"
    config_source.parent.mkdir(parents=True, exist_ok=True)
    config_bytes = canonical_json({"format": "molgap-scientific-experiment-v1",
        "experiment_id": declaration["experiment_id"], "question": {"changed_mechanism": "synthetic-intervention"},
        "arms": [{"arm_id": arm["arm_id"], "purpose": comparison_purpose}
                 for arm in declaration["arms"]]}).encode()
    config_source.write_bytes(config_bytes)
    # A current package carries the real runner (with the data-open recorder) for the role-audit path.
    runner = repo / "src/molgap/family_training_runner.py"
    runner.write_bytes((Path(__file__).resolve().parents[1] / "src/molgap/family_training_runner.py").read_bytes())
    subprocess.run(["git", "add", "experiments/synthetic/config.json", "src/molgap/family_training_runner.py"],
                   cwd=repo, check=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic acceptance config"],
                   cwd=repo, check=True, capture_output=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    generated_root = "experiments/" + declaration["experiment_id"] + "/generated/" + canonical_fingerprint({
        "input": hashlib.sha256(config_bytes).hexdigest(), "commit": source})
    for binding in declaration["prospective"]["arms"]:
        binding["trajectory_id"] = "trajectory-" + binding["arm_id"].replace("_", "-")
        binding["plan_spec_ref"] = generated_root + "/" + binding["arm_id"] + "-plan.json"
    generated, entries = {}, []

    def add(name, raw):
        generated[name] = raw
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    target = add(generated_root + "/target.json", canonical_json({
        **({"training_float32_target_sha256": training_target_sha256} if retain_training_target else {}),
        "development_target_sha256": contracts["k1_candidate"][0]["acceptance_requirements"]["target_sha256"],
        "target_encoding": "little-endian-float64-contiguous-raw-bytes"}).encode())
    for arm, binding in zip(declaration["arms"], declaration["prospective"]["arms"]):
        arm_id = arm["arm_id"]
        record, raw = contracts[arm_id]
        contract = add(generated_root + "/" + arm_id + "-recipe.json", raw)
        if hashlib.sha256(raw).hexdigest() != arm["training"]["recipe"]["sha256"]:
            raise AssertionError("Frozen shared recipe digest differs from Spec")
        run_key = declaration["logical_run_id"] + ":" + arm_id + ":downstream"
        proposal = {"trajectory": {"trajectory_id": binding["trajectory_id"],
            "state_at_start": {"source_commit": source, "source_config_identity": canonical_fingerprint(arm)},
            "actions": [{"action_id": "matching", "run_ids": [run_key],
                "attempt_ids": [declaration["logical_run_id"] + "-v7"],
                "source_commit": source}]}}
        binding["plan_spec_sha256"] = add(binding["plan_spec_ref"], canonical_json(proposal).encode())["sha256"]
        entries.append({"arm_id": arm_id, "adapter": "k1-session-v1" if arm["family"]["name"] == "neural_atom_k1" else "gptrans-session-v1",
            "expected": record["acceptance_requirements"], "contract": contract,
            "reference_artifacts": {"target_manifest": target}})

    spec = ExperimentSpec(declaration)
    plan_name = generated_root + "/acceptance.json"
    add(plan_name, canonical_json({"format": "molgap-family-acceptance-plan-v1",
        "spec_identity": spec.identity, "arms": entries}).encode())
    workflow = {"spec_identity": spec.identity, "acceptance_plan": plan_name,
        "platform": {"source_dataset": "synthetic-owner/source"}}
    run = paths.run_directory(declaration["logical_run_id"], "v1")
    package = run / "prepared/package"
    manifest = build_experiment_source_package(spec, repo,
        ["src/example.py", "README.md", "experiments/synthetic/config.json",
         "src/molgap/family_training_runner.py", *generated],
        package, generated_records=generated,
        configuration_source={"path": "experiments/synthetic/config.json",
            "sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
            "compiler": "molgap-scientific-experiment-v1"})
    write(run / "inputs/experiment_spec.json", spec.to_dict())
    write(run / "inputs/workflow_plan.json", workflow)
    binding = {"format": "molgap-experiment-input-v1", "input_identity": "f" * 64,
        "spec_identity": spec.identity, "workflow_identity": canonical_fingerprint(workflow),
        "authority_ref": "authority.json", "monitor_paths": ["experiments/question/STATUS.md"],
        "source_commit": source}
    write(run / "inputs/input_binding.json", binding)
    write(run / "prepared/workflow_report.json", {"status": "PREPARED_FOR_PLATFORM",
        "spec_identity": spec.identity, "package_identity": manifest["package_identity"], "source_commit": source})
    context = spec, package, manifest["package_identity"]
    response = envelope(context)
    response["platform_version"] = known("7")
    response["physical_runs"] = known([{"arm_ids": [arm["arm_id"] for arm in declaration["arms"]],
        "run_identity": "kernel1/version7", "canonical_reference": "synthetic-owner/workload",
        "platform_version": known("7")}])
    receipt = build_receipt(context, response)
    write(run / "operations/launch_receipt.json", receipt)
    write(run / "operations/submission_state.json", {"phase": "BOUND", "binding": {
        "package_identity": manifest["package_identity"], "kernel": "synthetic-owner/workload",
        "dataset": "synthetic-owner/source", "authority_ref": binding["authority_ref"],
        "monitor_paths": binding["monitor_paths"]},
        "kernel_response": {"kernel": "synthetic-owner/workload", "version_number": 7}})
    return SimpleNamespace(profile=profile, paths=paths, repo=repo, run=run, prepared=run / "prepared",
        package=package, spec=spec, manifest=manifest, receipt=receipt,
        expected={arm["arm_id"]: item["expected"] for arm, item in zip(declaration["arms"], entries)},
        training_target_sha256=training_target_sha256,
        generated=generated, original=(profile, repo, declaration["logical_run_id"], "v1"))
