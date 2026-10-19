"""Finish this exact unsubmitted package after its RML rebuild-only failure.

This is a bounded composition of existing owners, not a prospective retry or
platform submitter. It cannot replace source, plans, trajectories or artifacts.
"""
from pathlib import Path
import json

from molgap.common.io import copy_immutable_file, publish_immutable_bytes
from molgap.experiment_package import verify_experiment_source_package
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_workflow import _workflow_inputs
from molgap.kaggle_workflow import stage_inputs, freeze_inputs, bind_release
from molgap.research_memory.compiler import rebuild_research_memory
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes


def reconcile():
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent / "k1"
    spec = ExperimentSpec(json.loads((here / "experiment_spec.json").read_bytes()))
    plan = json.loads((here / "workflow_plan.json").read_bytes())
    prepared = Path("D:/molgap-artifacts/runs") / spec.to_dict()["logical_run_id"] / "v1/prepared"
    report_path = prepared / "workflow_report.json"
    report = json.loads(report_path.read_bytes())
    if (report["status"] != "RECONCILIATION_REQUIRED" or report.get("stage") != "prospective"
            or report["spec_identity"] != spec.identity or report.get("submitted") is not False
            or (prepared.parent / "operations").exists()):
        raise ValueError("Not this unsubmitted RML-rebuild-only failure")
    publication = next(s["result"] for s in report["stages"] if s["stage"] == "prospective")
    if publication["status"] != "PROSPECTIVE_PLANNED_RML_REBUILD_FAILED":
        raise ValueError("Prospective publication is partial or unknown; do not reconstruct it")
    manifest = verify_experiment_source_package(prepared / "package")
    if manifest["spec_identity"] != spec.identity:
        raise ValueError("Original immutable package differs")
    for binding in spec.to_dict()["prospective"]["arms"]:
        proposal = json.loads((root / binding["plan_spec_ref"]).read_bytes())
        trajectory = json.loads((root / binding["output"] / "trajectory.json").read_bytes())
        if (file_digest(root / binding["plan_spec_ref"]) != binding["plan_spec_sha256"]
                or trajectory["state_at_start"]["source_commit"] != manifest["source_commit"]
                or trajectory["state_at_start"]["source_config_identity"] != proposal["trajectory"]["state_at_start"]["source_config_identity"]
                or trajectory["actions"] != proposal["trajectory"]["actions"]):
            raise ValueError("Published prospective source/action differs")
    release = json.loads((prepared / "release_report.json").read_bytes())
    if release["errors"] or release["source_commit"] != manifest["source_commit"]:
        raise ValueError("Original release check is not valid")
    inputs = _workflow_inputs(spec, root, plan)
    derived = rebuild_research_memory(root)
    publish_immutable_bytes(prepared / "workflow_report_before_reconciliation.json", report_path.read_bytes())
    scratch = prepared / "reconciled_stage_002"
    stage = stage_inputs(repo_root=root, output=scratch, package=prepared / "package",
        manifest=manifest, spec=spec, platform_plan=plan["kaggle"], metadata=inputs["metadata"],
        initial_states=inputs["initial_states"], jobs=inputs["jobs"], acceptance_plan_path=plan["acceptance_plan"])
    freeze_inputs(stage, {b["arm_id"]: root / b["output"] / "trajectory.json"
                         for b in spec.to_dict()["prospective"]["arms"]})
    for directory in ("source_dataset", "kernel"):
        for source in (scratch / directory).rglob("*"):
            if source.is_file():
                copy_immutable_file(source, prepared / source.relative_to(scratch),
                                    expected_sha256=file_digest(source))
    for name, value in (("input_root", prepared / "source_dataset"), ("kernel_dir", prepared / "kernel"),
                        ("entry_path", prepared / "kernel/run.py"), ("metadata_path", prepared / "kernel/kernel-metadata.json"),
                        ("launch_path", prepared / "source_dataset/experiment_launch.json")):
        stage[name] = value
    bind_release(stage, spec, release)
    atomic_write(prepared / "release_report.json", json_bytes(release))
    report.update(status="PREPARED_FOR_PLATFORM", prospective_published=True,
        package_identity=manifest["package_identity"], source_commit=manifest["source_commit"],
        release_report=str(prepared / "release_report.json"), kernel_dir=str(prepared / "kernel"),
        source_dataset_dir=str(prepared / "source_dataset"), accelerator=stage["accelerator"],
        reconciliation="Original published trajectories retained; only failed global rebuild and final freeze completed")
    report["stages"].append({"stage": "reconciled_rml_and_final_binding", "result": {"rml": "REBUILT",
        "derived_files": sorted(derived),
        "release_status": release["status"], "original_prospective_unchanged": True}})
    atomic_write(report_path, json_bytes(report))
    print(json.dumps({"status": report["status"], "prepared": str(prepared), "submitted": False}))


if __name__ == "__main__":
    reconcile()
