"""Declare this authorized recovery; shared owners validate and stage every byte."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

from molgap.experiment_package import read_packaged_text
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_workflow_resume import prepare_resumed_workflow
from molgap.training_reproducibility import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slot", choices=("single", "mean2"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    original = Path("D:/molgap-artifacts/runs") / f"pcqm-k1-init-forward-500k-{args.slot}-s42/v1"
    recovery = Path("D:/molgap-artifacts/recovery/k1-init-forward-500k") / args.slot / "v2"
    package = original / "prepared/package"
    spec = ExperimentSpec.from_json((original / "inputs/experiment_spec.json").read_text())
    manifest = json.loads((package / "package_manifest.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    files = {}
    for name in ("src/molgap/experiment_resume.py", "src/molgap/family_training_runner.py"):
        old = read_packaged_text(package, name).encode()
        new = subprocess.check_output(["git", "show", commit + ":" + name], cwd=repo)
        files[name] = {"original_sha256": hashlib.sha256(old).hexdigest(),
                       "repaired_sha256": hashlib.sha256(new).hexdigest()}
    plan = {"format": "molgap-workflow-resume-v1", "spec_identity": spec.identity,
        "arms": {arm: str(recovery / "bundles" / arm) for arm in ("scratch", "pretrained")},
        "recovery_repair": {"commit": commit, "files": files},
        "allocation_wall_limit_seconds": 18000 if args.slot == "single" else 25200,
        "cleanup_reserve_seconds": 180}
    atomic_json(recovery / "resume_plan.json", plan)
    result = prepare_resumed_workflow(spec, repo, original / "prepared", package,
        manifest["package_identity"], original / "operations/launch_receipt.json",
        recovery / "prepared", plan=plan)
    print(json.dumps({key: result.get(key) for key in
        ("status", "errors", "kernel_dir", "source_dataset_dir", "recovery_repair")}))


if __name__ == "__main__":
    main()
