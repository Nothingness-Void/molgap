"""Single-process local adapter over frozen preparation and native family owners."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

from .experiment_family_workflow import _json
from .experiment_launch import canonical_json, publish_immutable_bytes, _safe_local
from .research_memory.trace import json_bytes


def run_local_training(prepared, repo_root, data_root, output, *, pause_after_step=None,
                       acceptance_plan=None):
    """Run one observed local process; repeated calls resume the same frozen arm."""
    if pause_after_step is not None and (type(pause_after_step) is not int or pause_after_step < 1):
        raise ValueError("pause_after_step must be a positive absolute optimizer step")
    from .experiment_execution import execute_training_phase, validate_execution_plan
    from .experiment_family_workflow import RunContext, TargetIdentityBinding
    from .experiment_launch import build_launch_receipt, read_launch_receipt
    from .experiment_package import verify_experiment_source_package
    from .experiment_spec import ExperimentSpec
    from .research_memory.trace import file_digest
    from .training_reproducibility import atomic_json

    prepared, repo_root, data_root, output = (Path(p).absolute() for p in
        (prepared, repo_root, data_root, output))
    for path in (prepared, repo_root, data_root, output):
        _safe_local(path)
    package = prepared / "package"
    report = _json(prepared / "prepared_training.json")
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    declaration = spec.to_dict()
    if (declaration["platform"]["name"] != "local" or declaration["platform"]["device_count"] != 1
            or len(declaration["arms"]) != 1):
        raise ValueError("Local training requires one frozen local arm and one physical device")
    if (report.get("status") != "SOURCE_INPUTS_PREPARED" or report.get("spec_identity") != spec.identity):
        raise ValueError("Local training preparation/Spec mismatch")
    manifest = verify_experiment_source_package(package)
    identity = report["expected_package_identity"]
    if manifest["package_identity"] != identity:
        raise ValueError("Local training package differs from preparation")
    job, = validate_execution_plan(spec, report["jobs"])
    if job["module"] != "molgap.family_training_runner":
        raise ValueError("Local execution requires the registered shared family trainer")
    target = None
    if acceptance_plan is not None:
        target = TargetIdentityBinding.from_frozen_package(spec, repo_root, package, acceptance_plan)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "active_process.json"
    started = datetime.now(timezone.utc).isoformat()
    hostname, pid = socket.gethostname(), os.getpid()
    host = hostname.encode("idna").decode("ascii")
    invocation = {"platform": "local", "hostname": hostname, "hostname_idna": host,
        "pid": pid, "started_at": started,
        "spec_identity": spec.identity, "package_identity": identity,
        "pause_after_step": pause_after_step, "scientific_acceptance": "NOT_EVALUATED"}
    # An interrupted adapter leaves this pointer for explicit reconciliation.
    # Cooperative pauses and ordinary failures always release their own lock.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(invocation))
    invocation_path = output / "invocations" / (started.replace(":", "-") + f"-pid{pid}.json")
    try:
        receipt_path = output / "launch_receipt.json"
        if receipt_path.exists():
            receipt = read_launch_receipt(receipt_path, spec, package, expected_package_identity=identity)
            reference = receipt["canonical_platform_reference"]["value"]
            if reference.split("/")[0] != host:
                raise ValueError("Local continuation belongs to a different observed host")
        else:
            base = build_launch_receipt(spec, package, expected_package_identity=identity)
            reference = host + "/" + declaration["logical_run_id"]
            known = lambda value: {"value": value, "missing_reason": None}
            # Version 1 is this local logical launch. Individual process
            # invocations retain actual PID/time identities beside this receipt.
            response = {"format": "molgap-platform-response", "version": 1,
                "mode": "observed", "outcome": "accepted", "conflict_kind": None,
                "binding": base["binding"], "canonical_platform_reference": known(reference),
                "platform_version": known("1"), "physical_runs": known([{
                    "run_identity": f"{host}/pid{pid}/" + started.replace(":", "-").replace("+", "_"),
                    "canonical_reference": reference, "platform_version": known("1"),
                    "arm_ids": [job["arm_id"]]}]), "timestamp": known(started),
                "monitor_paths": {"value": None, "missing_reason": "not_observed"}}
            receipt = build_launch_receipt(spec, package, expected_package_identity=identity,
                response_json=canonical_json(response))
            publish_immutable_bytes(receipt_path, canonical_json(receipt).encode("utf-8"))
        context = RunContext.from_launch(spec, receipt_path, package,
            expected_package_identity=identity, arm_id=job["arm_id"])
        invocation["launch_receipt_sha256"] = file_digest(receipt_path)
        arm_root = output / job["arm_id"]
        resumed = (arm_root / "last_checkpoint.pt").is_file()
        invocation["resume_checkpoint_sha256"] = file_digest(arm_root / "last_checkpoint.pt") if resumed else None
        options = dict(spec=spec, job=job, source_root=repo_root, package_dir=package,
            expected_package_identity=identity, input_root=data_root, account=host,
            run_reference=reference, staged_root=prepared / "inputs",
            prospective_sha256=report["prospective_sha256"][job["arm_id"]])
        if not (arm_root / "output_manifest.json").is_file():
            preflight_root = output / "preflight" / invocation_path.stem
            qualified = execute_training_phase(phase="preflight", output=preflight_root,
                **options, **({"resume_output": arm_root} if resumed else {}))
            if qualified.get("status") != "accepted" and qualified.get("accepted") is not True:
                raise ValueError("Local native preflight did not pass")
            invocation["preflight_output"] = str(preflight_root)
        else:
            preflight_root = arm_root
        result = execute_training_phase(phase="train", output=arm_root, **options,
            target_identity=target, preflight_dir=preflight_root, pause_after_step=pause_after_step)
        invocation.update(status=result.get("status"), context=context.to_dict())
        if result.get("status") not in {"PAUSED", "MECHANICALLY_VERIFIED"}:
            raise ValueError("Local training did not pause or mechanically verify: " + str(result.get("blockers")))
        summary = {"status": result["status"], "platform": "local", "resumed": resumed,
            "logical_run_id": declaration["logical_run_id"],
            "spec": str(package / "experiment_spec.json"), "package": str(package),
            "receipt": str(receipt_path), "expected_package_identity": identity,
            "outputs": {job["arm_id"]: str(arm_root)}, "invocation": str(invocation_path),
            "mechanical": result, "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED"}
        atomic_json(output / "local_training_report.json", summary)
        return summary
    except BaseException as exc:
        invocation.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        invocation["finished_at"] = datetime.now(timezone.utc).isoformat()
        publish_immutable_bytes(invocation_path, json_bytes(invocation))
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("prepared", "repo-root", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pause-after-step", type=int)
    parser.add_argument("--acceptance-plan")
    parser.add_argument("--accept-only", action="store_true")
    parser.add_argument("--outcome", choices=("INFRASTRUCTURE_ONLY",))
    parser.add_argument("--decision-ref")
    parser.add_argument("--role-audit")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.accept_only:
        result = accept_local_training(args.prepared, args.repo_root, args.output,
            acceptance_plan=args.acceptance_plan, outcome=args.outcome,
            decision_ref=args.decision_ref, role_audit=args.role_audit, execute=args.execute)
        path = args.output / "local_acceptance_report.json"
    else:
        if any((args.outcome, args.decision_ref, args.role_audit, args.execute)):
            parser.error("Terminal inputs require --accept-only")
        result = run_local_training(args.prepared, args.repo_root, args.data_root, args.output,
            pause_after_step=args.pause_after_step, acceptance_plan=args.acceptance_plan)
        path = args.output / "local_training_report.json"
    visible = result if args.verbose else {"status": result["status"], "platform": "local",
        "run_id": result.get("logical_run_id"),
        "report": str(path.absolute()), "scientific_acceptance": "NOT_EVALUATED"}
    print(json.dumps(visible, ensure_ascii=False, indent=2))
    return 1 if result["status"] in {"BLOCKED", "INCOMPLETE", "STAGED_DRAFT"} else 0


def accept_local_training(prepared, repo_root, output, *, acceptance_plan=None,
                          outcome=None, decision_ref=None, role_audit=None, execute=False):
    """Use the existing mechanical inspector, terminal staging and RML closure."""
    from .experiment_family_workflow import prepare_terminal_outputs, TargetIdentityBinding
    from .experiment_package import read_packaged_text
    from .experiment_spec import ExperimentSpec
    from .experiment_terminal_staging import stage_verified_terminal_packages
    from .experiment_workflow import accept_workflow
    from .training_reproducibility import atomic_json

    if (outcome is None) != (decision_ref is None):
        raise ValueError("An explicit infrastructure outcome and decision are required together")
    prepared, root, output = (Path(p).absolute() for p in (prepared, repo_root, output))
    package = prepared / "package"
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    report = _json(prepared / "prepared_training.json")
    declaration = spec.to_dict()
    if report.get("status") != "SOURCE_INPUTS_PREPARED" or report.get("spec_identity") != spec.identity:
        raise ValueError("Local acceptance preparation/Spec mismatch")
    if declaration["platform"]["name"] != "local" or len(declaration["arms"]) != 1:
        raise ValueError("Expected a prepared single-arm local run")
    options = dict(package_dir=package, receipt_path=output / "launch_receipt.json",
                   expected_package_identity=report["expected_package_identity"])
    supplied = {}
    for job in report["jobs"]:
        record = json.loads(read_packaged_text(package, job["recipe"]))
        supplied[job["arm_id"]] = {"output_dir": str(output / job["arm_id"]),
                                   "expected": record["acceptance_requirements"]}
    outputs = prepare_terminal_outputs(spec, root, supplied, **options)
    targets = ({arm_id: TargetIdentityBinding.from_frozen_package(spec, root, package, acceptance_plan)
                for arm_id in outputs} if acceptance_plan is not None else {})
    mechanical = accept_workflow(spec, root, supplied, target_identities=targets, **options)
    result = mechanical
    if mechanical["status"] != "BLOCKED" and outcome is not None:
        locations = {binding["arm_id"]: {"trajectory_id": binding["trajectory_id"],
            "run_id": declaration["logical_run_id"] + ":" + binding["arm_id"] + ":downstream",
            "trajectory": binding["output"] + "/trajectory.json",
            "terminal": binding["output"] + "/terminal.json",
            "trace": binding["output"] + "/trace_manifest.json"}
            for binding in declaration["prospective"]["arms"]}
        resolved = {"spec": spec, "options": options, "outputs": outputs,
            "contexts": {arm_id: item["context"] for arm_id, item in outputs.items()},
            "locations": locations, "target_identities": targets, "run_directory": output}
        staged = stage_verified_terminal_packages(None, root, declaration["logical_run_id"], "v1",
            resolved=resolved, reports=mechanical["arms"], outcome=outcome,
            decision_ref=decision_ref, role_audit=role_audit)
        result = {**mechanical, "terminal_staging": staged,
                  "status": staged["status"], "blockers": staged.get("blockers", [])}
        if execute:
            if staged.get("blockers"):
                result.update(status="BLOCKED", executed=False)
            else:
                result.update(accept_workflow(spec, root, supplied, target_identities=targets,
                    locations=staged["locations"], execute=True, **options))
    elif execute:
        raise ValueError("Terminal execution requires explicit outcome and decision")
    result["logical_run_id"] = declaration["logical_run_id"]
    atomic_json(output / "local_acceptance_report.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
