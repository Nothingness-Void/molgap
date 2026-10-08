"""Retained bounded500K preparation/continuation adapter from owner031a890b."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from molgap.experiment_package import build_experiment_source_package, verify_experiment_source_package
from molgap.experiment_preflight import check_release_inputs
from molgap.experiment_prospective import plan_prospective
from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import sha256_file, retained_resume_artifacts

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


def prepare_continuation(previous, resume_root, output, pickle_input,
                         checkpoint_dataset, max_stage_seconds=None, *, repo_root, experiment_dir,
                         source_dataset=None):
    """Keep prospective/scientific inputs frozen while rebinding executable source."""
    ROOT, EXP = Path(repo_root), Path(experiment_dir)
    import torch
    previous, resume_root, output = map(Path, (previous, resume_root, output))
    old = verify_experiment_source_package(previous / "package")
    if max_stage_seconds is not None and (type(max_stage_seconds) is not int or not 0 < max_stage_seconds <= 32400):
        raise ValueError("Continuation stage bound must be within the approved 9 hours")
    spec = ExperimentSpec.from_json((previous / "package/experiment_spec.json").read_text())
    if spec.identity != ExperimentSpec.from_json((EXP / "experiment_spec.json").read_text()).identity:
        raise ValueError("Continuation cannot change the owning frozen Spec")
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_experiment_source_package(spec, ROOT, old["relative_allowlist"], output / "package")
    source, kernel, checkpoints = output / "source_dataset", output / "kernel", output / "checkpoint_dataset"
    shutil.copytree(previous / "source_dataset", source)
    shutil.copytree(previous / "kernel", kernel)
    checkpoints.mkdir()
    checkpoint_id = checkpoint_dataset
    if not checkpoint_id.startswith("nothingnessvoid/") or checkpoint_id.count("/") != 1:
        raise ValueError("Continuation checkpoint dataset must belong to Kaggle1")
    checkpoint_slug = checkpoint_id.split("/")[1]
    if not checkpoint_slug or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in checkpoint_slug):
        raise ValueError("Invalid checkpoint dataset slug")
    launch = read(source / "legacy_500k_launch.json")
    if source_dataset is not None:
        if (not source_dataset.startswith("nothingnessvoid/") or source_dataset.count("/") != 1
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in source_dataset.split("/")[1])
                or not source_dataset.split("/")[1]):
            raise ValueError("Continuation source dataset must belong to Kaggle1")
        previous_source = read(source / "dataset-metadata.json")
        if source_dataset == previous_source["id"]:
            raise ValueError("Use a fresh source dataset for changed executable bytes")
        launch["dataset_sources"] = [source_dataset if d == previous_source["id"] else d
                                     for d in launch["dataset_sources"]]
        launch["source_mount"] = source_dataset.split("/")[1]
        previous_source.update(id=source_dataset, title="MolGap K1 500K Continuation Source S42")
        write(source / "dataset-metadata.json", previous_source)
    if max_stage_seconds is not None:
        launch["max_stage_seconds"] = max_stage_seconds
    launch.update(source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"],
                  package_identity=manifest["package_identity"])
    prior_resume_mounts = {a["resume"]["mount"] for a in launch["arms"] if "resume" in a}
    launch["dataset_sources"] = [d for d in launch["dataset_sources"]
                                 if d.split("/")[-1] not in prior_resume_mounts]
    launch["dataset_sources"] += [checkpoint_id]
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
        for name, checksum in retained_resume_artifacts(stage, "selected-and-resume-v1").items():
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
                         "source_sha256": old["archive_sha256"], "next_epoch": state["next_epoch"],
                         "retention": "selected-and-resume-v1"}
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
    write(checkpoints / "dataset-metadata.json", {"id": checkpoint_id, "title": "MolGap K1 500K Resume S42",
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


def prepare_payload(repo_root, experiment_dir, spec, pins, initial, transform, output, *,
                    source_files, required_modules, run_id, source_dataset, title,
                    source_title, inputs, pickle_input, resume_prepared=False):
    """Prepare a previously declared two-arm frozen500K question; never submit."""
    ROOT, EXP = Path(repo_root), Path(experiment_dir)
    REL = EXP.relative_to(ROOT).as_posix()
    ARMS = tuple(a["arm_id"] for a in spec.to_dict()["arms"])
    EXTRA_SOURCE, REQUIRED_MODULES, RUN = source_files, required_modules, run_id
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if len(ARMS) != 2 or not source_dataset.startswith("nothingnessvoid/"):
        raise ValueError("This retained adapter requires a Kaggle1 two-arm question")
    if source_dataset == "nothingnessvoid/molgap-k1-gptrans-500k-source-s42-v1":
        raise ValueError("A new question cannot publish over the retained source dataset")
    if Path(output).exists() and not resume_prepared:
        raise ValueError("Provide a fresh output directory")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume_prepared)
    recipes = {aid: f"{REL}/training_recipe_{aid}.json" for aid in ARMS}
    names = sorted(set(SHARED_SOURCE_FILES) | set(EXTRA_SOURCE) | set(recipes.values()) | {f"{REL}/protocol.md", f"{REL}/plan.md", f"{REL}/role_plan.json", f"{REL}/budget.json"})
    manifest = (verify_experiment_source_package(output / "package", repo_root=ROOT)
                if resume_prepared else build_experiment_source_package(spec, ROOT, names, output / "package"))
    if manifest["source_commit"] != actual:
        raise ValueError("Package source commit changed")
    if resume_prepared:
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
    shutil.copyfile(Path(inputs) / "initial_pins.json", source / "inputs/initial_pins.json")
    launch = {"format": "molgap-legacy-500k-pair-v1", "spec_identity": spec.identity, "package_identity": manifest["package_identity"],
        "required_modules": REQUIRED_MODULES,
        "run_reference": "nothingnessvoid/" + RUN, "account": "nothingnessvoid", "accelerator": "NvidiaTeslaT4", "device_count": 2,
        "dataset_sources": [source_dataset, "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1"],
        "source_mount": source_dataset.split("/")[1], "graph_mount": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
        "prospective_sha256": {a["arm_id"]: a["trajectory_sha256"] for a in launch_arms},
        "source_commit": actual, "source_archive_sha256": manifest["archive_sha256"], "stage_epochs": 60, "max_stage_seconds": 32400,
        "target_transform": "inputs/target_transform.json", "target_transform_sha256": pins["target_transform"]["sha256"], "arms": launch_arms}
    write(source / "legacy_500k_launch.json", launch)
    entry = (ROOT / "platforms/kaggle/run_legacy_500k_pair.py").read_text(encoding="utf-8").replace("EXPECTED_LAUNCH_SHA256 = None", "EXPECTED_LAUNCH_SHA256 = " + repr(sha256_file(source / "legacy_500k_launch.json")))
    (kernel / "run.py").write_text(entry, encoding="utf-8", newline="\n")
    write(source / "dataset-metadata.json", {"id": source_dataset, "title": source_title, "licenses": [{"name": "other"}], "isPrivate": True})
    write(kernel / "kernel-metadata.json", {"id": "nothingnessvoid/" + RUN, "title": title, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": launch["dataset_sources"], "competition_sources": [], "kernel_sources": [], "model_sources": []})
    report = check_release_inputs(spec, output / "package", expected_package_identity=manifest["package_identity"], recipe_files=recipes,
        initial_states={a["arm_id"]: source / a["initial_state"] for a in launch_arms}, required_modules=REQUIRED_MODULES,
        pickle_inputs=[pickle_input], entry_script=kernel / "run.py", input_root=source, launch_config=source / "legacy_500k_launch.json", kernel_metadata=kernel / "kernel-metadata.json")
    write(output / "legacy_500k_binding.json", {"launch_sha256": sha256_file(source / "legacy_500k_launch.json"), "entry_sha256": sha256_file(kernel / "run.py"), "metadata_sha256": sha256_file(kernel / "kernel-metadata.json"), "prospective": {a["arm_id"]: a["trajectory_sha256"] for a in launch_arms}, "workflow_registry": "UNSUPPORTED; owning500Klegacy adapter"})
    write(output / "release_report.json", report)
    if report["errors"]:
        raise RuntimeError("Local release inputs failed; inspect release_report.json")
    return {"status": "LOCAL_LEGACY500K_PAYLOAD_PREPARED", "package_identity": manifest["package_identity"], "spec_identity": spec.identity, "output": str(output)}
