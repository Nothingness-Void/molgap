"""Local Kaggle preparation adapter; remote operations remain in the workload skill."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil

from .experiment_family_workflow import _artifact_path
from .experiment_launch import canonical_json, publish_immutable_bytes
from .experiment_package import read_packaged_text

def _metadata(kaggle: dict) -> dict:
    fields = {"account", "kernel", "title", "datasets", "source_dataset", "accelerator"}
    if type(kaggle) is not dict or set(kaggle) != fields:
        raise ValueError("Expected explicit Kaggle account/kernel/title/datasets/source_dataset/accelerator")
    if not re.fullmatch(r"[a-z0-9-]+", kaggle["account"]):
        raise ValueError("Invalid Kaggle account")
    prefix = kaggle["account"] + "/"
    slug = re.sub(r"[^a-z0-9]+", "-", kaggle["title"].lower()).strip("-")
    if not 5 <= len(kaggle["title"]) <= 50 or kaggle["kernel"] != prefix + slug:
        raise ValueError("Kernel title-generated slug must match the requested kernel")
    if (type(kaggle["datasets"]) is not list or not kaggle["datasets"] or
        len(set(kaggle["datasets"])) != len(kaggle["datasets"]) or
        any(not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", d) for d in kaggle["datasets"]) or
        not kaggle["source_dataset"].startswith(prefix) or
        kaggle["source_dataset"] not in kaggle["datasets"]):
        raise ValueError("Expected unique dataset mounts including the account-owned source dataset")
    if kaggle["accelerator"] != "NvidiaTeslaT4":
        raise ValueError("This platform adapter supports the explicit T4 allocation only")
    return {"id": kaggle["kernel"], "title": kaggle["title"], "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": kaggle["datasets"], "competition_sources": [], "kernel_sources": []}



def validate_plan(spec, plan):
    metadata = _metadata(plan)
    platform = spec.to_dict()["platform"]
    if platform["name"] != "kaggle" or platform["accelerator"] not in {"Tesla T4", "T4", "NvidiaTeslaT4"} or platform["device_count"] != 2:
        raise ValueError("Kaggle T4 requires its declared two-device allocation")
    return metadata


def stage_inputs(*, repo_root, output, package, manifest, spec, platform_plan,
                 metadata, initial_states, jobs):
    staged = output / "source_dataset"
    staged.mkdir()
    for path in package.iterdir():
        shutil.copyfile(path, staged / ("source_payload.bin" if path.name == "source.tar.gz" else path.name))
    (staged / "initial_states").mkdir()
    for arm_id, initial in initial_states.items():
        shutil.copyfile(initial, staged / "initial_states" / (arm_id + ".pt"))
    publish_immutable_bytes(staged / "dataset-metadata.json", canonical_json({
        "id": platform_plan["source_dataset"], "title": platform_plan["source_dataset"].split("/")[1],
        "licenses": [{"name": "CC0-1.0"}]}).encode())
    kernel = output / "kernel"
    kernel.mkdir()
    marker = "EXPECTED_LAUNCH_SHA256 = None"
    entry_text = read_packaged_text(package, "platforms/kaggle/run_experiment.py")
    if entry_text.count(marker) != 1:
        raise ValueError("Expected one reviewed launch digest marker in the platform entry")
    publish_immutable_bytes(kernel / "kernel-metadata.json", canonical_json(metadata).encode())
    return {"input_root": staged, "kernel_dir": kernel, "entry_path": kernel / "run.py",
        "metadata_path": kernel / "kernel-metadata.json", "launch_path": staged / "experiment_launch.json",
        "entry_text": entry_text, "accelerator": platform_plan["accelerator"],
        "launch": {"format": "molgap-execution-launch-v1", "spec_identity": spec.identity,
            "expected_package_identity": manifest["package_identity"],
            "expected_source_archive_sha256": manifest["archive_sha256"],
            "account": platform_plan["account"], "run_reference": platform_plan["kernel"],
            "dataset_sources": metadata["dataset_sources"], "accelerator": platform_plan["accelerator"],
            "device_count": spec.to_dict()["platform"]["device_count"], "jobs": jobs}}


def freeze_inputs(stage, trajectories):
    config = stage["launch"]
    config["prospective_sha256"] = {}
    for arm_id, trajectory in trajectories.items():
        destination = stage["input_root"] / "prospective" / arm_id
        destination.mkdir(parents=True)
        # Publish and hash one observation; recovery supplies already captured bytes.
        raw = trajectory if type(trajectory) is bytes else Path(trajectory).read_bytes()
        publish_immutable_bytes(destination / "trajectory.json", raw)
        config["prospective_sha256"][arm_id] = hashlib.sha256(raw).hexdigest()
    publish_immutable_bytes(stage["launch_path"], canonical_json(config).encode())
    entry_text = stage["entry_text"].replace("EXPECTED_LAUNCH_SHA256 = None",
        "EXPECTED_LAUNCH_SHA256 = " + repr(hashlib.sha256(stage["launch_path"].read_bytes()).hexdigest()))
    publish_immutable_bytes(stage["entry_path"], entry_text.encode())


def bind_release(stage, spec, release):
    from .experiment_preflight import check_workflow_binding
    release["inputs"].update(entry_script=str(stage["entry_path"]),
        launch_config=str(stage["launch_path"]), kernel_metadata=str(stage["metadata_path"]))
    release["checks"]["entry_script:kernel"] = hashlib.sha256(stage["entry_path"].read_bytes()).hexdigest()
    release["checks"]["workflow:launch"] = check_workflow_binding(spec,
        launch_config=stage["launch_path"], kernel_metadata=stage["metadata_path"],
        entry_script=stage["entry_path"], package_dir=Path(release["inputs"]["package"]),
        expected_package_identity=release["package_identity"],
        recipe_files=release["inputs"]["recipe_files"],
        initial_states=release["inputs"]["initial_states"], input_root=stage["input_root"])
