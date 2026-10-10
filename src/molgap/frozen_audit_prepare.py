"""Shared immutable NO_TRAIN packaging; owning adapters supply the question."""
import json
from pathlib import Path
import shutil
import subprocess

from .training_reproducibility import atomic_json, sha256_file, canonical_fingerprint


def prepare_package(root, output, *, base, assets, kernel, dataset, plan_spec):
    from .research_memory.plan import plan
    from .v4_bundle import build_v4_source_bundle
    from .frozen_inference_release import FORMAT, check_frozen_inference_release
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Never overwrite a frozen release")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if plan_spec["trajectory"]["state_at_start"]["source_commit"] != commit:
        raise ValueError("Plan source differs from package source")
    prospective = plan(root, plan_spec, root / base / "rml_plan")
    inputs = output / "inputs"
    names = [n for n in subprocess.check_output(["git", "ls-files", "src/molgap"], cwd=root, text=True).splitlines() if n.endswith(".py")]
    names += [base + "/" + n for n in ("run.py", "contract.json", "protocol.md")]
    bundle = build_v4_source_bundle(repo_root=root, relative_paths=names, output_dir=inputs,
                                   source_commit=commit, archive_name="source_payload.bin")
    contract = json.loads((root / base / "contract.json").read_text(encoding="utf-8"))
    for name in [*contract["model_assets"], *contract["reference_payloads"], "target_transform.json"]:
        shutil.copyfile(root / assets / name, inputs / name)
    shutil.copyfile(root / base / "contract.json", inputs / "contract.json")
    meta = json.loads((root / base / "kernel-metadata.json").read_text(encoding="utf-8"))
    release = dict(format=FORMAT, experiment_purpose="NO_TRAIN", source_commit=commit,
        archive_sha256=bundle["archive_sha256"], contract_sha256=sha256_file(inputs / "contract.json"),
        entry_sha256=sha256_file(root / base / "run.py"), metadata_sha256=sha256_file(root / base / "kernel-metadata.json"),
        kernel=kernel, dataset_sources=meta["dataset_sources"],
        prospective_trajectory_id=plan_spec["trajectory"]["trajectory_id"],
        prospective_sha256=sha256_file(root / base / "rml_plan/trajectory.json"),
        files={name: sha256_file(inputs / name) for name in ("source_payload.bin", "SOURCE_FILES.json",
            "contract.json", "target_transform.json", *contract["model_assets"], *contract["reference_payloads"])})
    release["identity"] = canonical_fingerprint(release)
    atomic_json(inputs / "audit_release.json", release)
    atomic_json(inputs / "dataset-metadata.json", dict(id=dataset, title="MolGap GPTrans Source Dependence Inputs", licenses=[dict(name="CC0-1.0")]))
    report = check_frozen_inference_release(inputs, root / base / "run.py", root / base / "kernel-metadata.json")
    atomic_json(output / "release_report.json", report)
    atomic_json(root / base / "release_binding.json", dict(release=release, prospective=prospective,
        inputs=str(inputs), release_report=str(output / "release_report.json")))
    return report["status"]
