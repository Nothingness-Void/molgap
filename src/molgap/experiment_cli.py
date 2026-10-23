"""Thin local ExperimentSpec CLI; no submission or scientific authority."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

from .experiment_launch import (
    _safe_local, build_launch_receipt, canonical_json,
    reconcile_platform_response, write_launch_receipt,
)
from .experiment_package import build_experiment_source_package
from .experiment_preflight import LOADER_MODE, MODEL_MODE, run_experiment_preflight, check_release_inputs, _atomic
from .experiment_prospective import plan_prospective
from .experiment_runner import run_experiment
from .experiment_spec import ExperimentSpec
from .experiment_workflow import check_experiment_preparation, prepare_experiment_release
from .experiment_terminal import (
    TerminalDescriptor, execute_terminal_descriptor, translate_terminal_descriptor,
)


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, allow_abbrev=False))

    def error(self, message):
        raise ValueError(message)

    def print_help(self, file=None):
        print(canonical_json({"help": self.format_help()}), file=file or sys.stdout)


def _local(value: str) -> Path:
    path = Path(value).absolute()
    # Reuse the launch boundary: no network, traversal, link or reparse paths.
    _safe_local(path)
    return path


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    freeze=commands.add_parser("freeze-input")
    freeze.add_argument("--experiment",required=True,type=_local)
    freeze.add_argument("--repo-root",required=True,type=_local)
    locate=commands.add_parser("locate-run")
    locate.add_argument("--profile",required=True,type=_local)
    locate.add_argument("--repo-root",required=True,type=_local)
    locate.add_argument("--run-id",required=True)
    locate.add_argument("--attempt-id",required=True)
    diagnose=commands.add_parser("diagnose")
    diagnose.add_argument("--control",type=_local)
    diagnose.add_argument("--candidate",type=_local)
    diagnose.add_argument("--run-id")
    diagnose.add_argument("--attempt-id")
    diagnose.add_argument("--profile",type=_local)
    diagnose.add_argument("--repo-root",type=_local,default=Path.cwd())
    diagnose.add_argument("--retrieved-root",type=_local)
    diagnose.add_argument("--outputs",type=_local)
    diagnose.add_argument("--output",type=_local)
    diagnose.add_argument("--threshold-ev",type=float)
    diagnose.add_argument("--noninferiority-margin-ev",type=float)
    diagnose.add_argument("--bootstrap-samples",type=int)
    diagnose.add_argument("--seed",type=int,default=42)
    diagnose.add_argument("--verbose",action="store_true")
    for name in ("validate-spec", "package", "preflight", "run-diagnostic",
                 "launch-receipt", "terminal", "plan-prospective", "check-release",
                 "check-acceptance", "inspect-output", "accept-terminal",
                 "prepare-workflow", "prepare-family-training", "accept-workflow", "build-terminal", "prepare-release", "check-preparation",
                 "workflow-info", "check-workflow", "build-resume", "prepare-resume", "source-dependencies"):
        command = commands.add_parser(name)
        by_run = name in {"inspect-output", "accept-workflow", "build-resume", "prepare-resume"}
        if by_run:
            identity = command.add_mutually_exclusive_group(required=True)
            identity.add_argument("--spec", type=_local)
            identity.add_argument("--run-id")
            command.add_argument("--profile", type=_local)
            command.add_argument("--attempt-id")
        else:
            command.add_argument("--spec", required=True, type=_local)
        if name == "inspect-output":
            command.add_argument("--repo-root", type=_local)
        if name in {"package", "terminal", "plan-prospective", "check-acceptance", "accept-terminal", "prepare-workflow", "prepare-family-training", "accept-workflow", "prepare-release", "check-preparation", "build-terminal", "build-resume", "prepare-resume", "source-dependencies", "check-workflow"}:
            command.add_argument("--repo-root", required=True, type=_local)
        if name in {"package", "preflight", "run-diagnostic", "prepare-workflow", "prepare-family-training", "prepare-release", "build-resume", "prepare-resume"}:
            command.add_argument("--output", required=True, type=_local)
        if name in {"preflight", "launch-receipt", "check-release", "inspect-output", "accept-terminal", "accept-workflow", "build-terminal", "build-resume", "prepare-resume"}:
            command.add_argument("--package", required=not by_run, type=_local)
            command.add_argument("--expected-package-identity", required=not by_run)
        if name in {"build-resume", "prepare-resume"}:
            command.add_argument("--prepared", type=_local)
            command.add_argument("--receipt", type=_local)
            if name == "build-resume":
                command.add_argument("--arm", required=True)
                command.add_argument("--source-output", required=True, type=_local)
            else:
                command.add_argument("--resume-plan", required=True, type=_local)
        elif name in {"prepare-release", "check-preparation"}:
            command.add_argument("--workflow", required=True, type=_local)
        elif name == "build-terminal":
            command.add_argument("--receipt", required=True, type=_local)
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--observations", type=_local)
            source.add_argument("--execution-state", type=_local)
            command.add_argument("--locations", required=True, type=_local)
            command.add_argument("--output", required=True, type=_local)
        elif name in {"prepare-workflow", "prepare-family-training", "check-workflow"}:
            command.add_argument("--plan", required=True, type=_local)
            if name == "prepare-family-training":
                command.add_argument("--auto-dependencies", action="store_true")
        elif name == "accept-workflow":
            command.add_argument("--receipt", type=_local)
            command.add_argument("--outputs", type=_local)
            command.add_argument("--locations", type=_local)
            command.add_argument("--retrieved-root", type=_local)
            command.add_argument("--diagnose", action="store_true")
            command.add_argument("--stage-terminal", action="store_true")
            command.add_argument("--outcome")
            command.add_argument("--decision-ref", type=_local)
            command.add_argument("--role-audit", type=_local)
            command.add_argument("--output", type=_local, help="Optional diagnostic JSON file or report directory")
            command.add_argument("--bootstrap-samples", type=int, default=0)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--execute", action="store_true")
            command.add_argument("--execution-root", type=_local)
            command.add_argument("--continuation-receipt", type=_local)
            command.add_argument("--resume-prepared", type=_local,
                help="Existing prepare-resume directory; derive continuation evidence from its pins")
            command.add_argument("--resume-authority", type=_local,
                help="Explicit recovery decision/authority retained for this continuation")
            command.add_argument("--strict", action="store_true",
                help="Recheck retained native, strict comparison and canonical replay evidence")
            command.add_argument("--strict-inputs", type=_local,
                help="Per-arm local proof/comparison locations; recheck existing strict owners")
            command.add_argument("--physical-runs", type=_local,
                help="Hash-bound physical allocation records; count each job/version once")
        elif name == "check-acceptance":
            command.add_argument("--plan", required=True, type=_local)
        elif name in {"inspect-output", "accept-terminal"}:
            command.add_argument("--receipt", required=not by_run, type=_local)
            if name == "inspect-output":
                command.add_argument("--arm", required=True)
                command.add_argument("--artifact-root", required=True, type=_local)
                command.add_argument("--expectations", required=True, type=_local)
            else:
                command.add_argument("--descriptor", required=True, type=_local)
                command.add_argument("--outputs", required=True, type=_local)
                command.add_argument("--execute", action="store_true")
        elif name == "package":
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--allowlist", nargs="+", action="extend")
            source.add_argument("--auto-dependencies", action="store_true")
        elif name == "source-dependencies":
            command.add_argument("--extra-source", action="append", default=[])
            command.add_argument("--strict-dynamic", action="store_true")
        elif name == "check-release":
            command.add_argument("--dataset-manifest", action="append", default=[], metavar="ARM=LOCAL_MANIFEST")
            command.add_argument("--recipe-file", required=True, action="append", metavar="ARM=PACKAGED_PATH")
            command.add_argument("--initial-state", action="append", default=[], metavar="ARM=LOCAL_PATH")
            command.add_argument("--required-module", required=True, action="append")
            command.add_argument("--pickle-input", action="append", default=[], type=_local)
            command.add_argument("--entry-script", type=_local)
            command.add_argument("--input-root", type=_local)
            command.add_argument("--output-report", type=_local)
        elif name == "preflight":
            command.add_argument("--shard-root", required=True, type=_local)
            command.add_argument("--expected-shard-manifest-sha256", required=True)
            command.add_argument("--mode", choices=(LOADER_MODE, MODEL_MODE), default=LOADER_MODE)
        elif name == "run-diagnostic":
            command.add_argument("--device", required=True, nargs="+", action="extend")
            command.add_argument("--worker", choices=("adapter_probe", "construct"), default="adapter_probe")
        elif name == "launch-receipt":
            command.add_argument("--output-dir", required=True, type=_local)
            command.add_argument("--response", type=_local)
        elif name == "terminal":
            command.add_argument("--descriptor", required=True, type=_local)
            command.add_argument("--execute", action="store_true")
    sandbox = commands.add_parser("sandbox")
    selected = sandbox.add_mutually_exclusive_group(required=True)
    selected.add_argument("--spec", type=_local)
    selected.add_argument("--family", choices=("neural_atom_k1", "gptrans_t"))
    sandbox.add_argument("--arm")
    for name in ("prepare-workflow", "prepare-family-training", "prepare-release", "accept-workflow"):
        commands.choices[name].add_argument("--verbose", action="store_true",
            help="Print the full owner report rather than its summary")
    return parser


def _dispatch(args) -> tuple[dict, int]:
    if args.command == "diagnose":
        from .experiment_attribution import analyze_family_pair
        if args.run_id is not None:
            if args.control is not None or args.candidate is not None:
                raise ValueError("Choose run identity or control/candidate directories")
            if args.profile is None or args.attempt_id is None:
                raise ValueError("Run diagnosis requires --profile and --attempt-id")
            if args.threshold_ev is not None or args.noninferiority_margin_ev is not None:
                raise ValueError("Run diagnosis uses the frozen comparison policy")
            from .experiment_attribution import diagnose_run
            from .experiment_family_workflow import _json
            result = diagnose_run(args.profile, args.repo_root, args.run_id, args.attempt_id,
                retrieved_root=args.retrieved_root,
                supplied=_json(args.outputs) if args.outputs is not None else None,
                output=args.output, n_bootstrap=args.bootstrap_samples or 0, seed=args.seed)
            return result, 1 if result.get("status") in {"BLOCKED", "INSUFFICIENT_EVIDENCE"} else 0
        if not args.control or not args.candidate:
            raise ValueError("Directory diagnosis requires --control and --candidate")
        if any(value is not None for value in (args.profile, args.attempt_id, args.retrieved_root, args.outputs)):
            raise ValueError("Run lookup options require --run-id")
        result = analyze_family_pair(
            args.control,
            args.candidate,
            output_dir=args.output,
            threshold_eV=0.003 if args.threshold_ev is None else args.threshold_ev,
            noninferiority_margin_eV=0.0005 if args.noninferiority_margin_ev is None else args.noninferiority_margin_ev,
            n_bootstrap=10000 if args.bootstrap_samples is None else args.bootstrap_samples,
            seed=args.seed,
        )
        return result, 0
    if args.command == "locate-run":
        from .experiment_input import locate_run
        return locate_run(args.profile,args.repo_root,args.run_id,args.attempt_id),0
    if args.command == "catalog":
        from dataclasses import asdict
        from .experiment_execution import TRAINING_ADAPTERS
        from .experiment_catalog import FAMILIES,ADDONS
        families=[]
        for key,contract in FAMILIES.items():
            adapter=TRAINING_ADAPTERS.get(key)
            families.append({**asdict(contract),"execution_support":
                "DECLARATION_ONLY" if adapter is None else
                "CPU_SYNTHETIC_ONLY" if adapter.cpu_synthetic_only else "REGISTERED_REQUIRES_QUALIFICATION"})
        return {"families":families,"addons":[{"name":name,"version":version,**asdict(c)}
            for (name,version),c in ADDONS.items()],"submitted":False},0
    if args.command == "freeze-input":
        from .experiment_input import freeze_experiment_input
        return freeze_experiment_input(args.experiment,args.repo_root),0
    if args.command == "sandbox":
        from .fast_sandbox import run_arm_sandbox, run_family_sandbox
        if args.spec is not None:
            if not args.arm:
                raise ValueError("A Spec sandbox requires one explicit --arm")
            raw = _read(args.spec)
            spec = ExperimentSpec.from_json(raw)
            if raw != spec.to_json():
                raise ValueError("Expected canonical ExperimentSpec JSON bytes (no newline)")
            result = run_arm_sandbox(spec, args.arm)
        else:
            if args.arm is not None:
                raise ValueError("--arm is only used with a Spec")
            result = run_family_sandbox(args.family)
        return result, 0 if result["status"] == "PASSED" else 1
    if args.command in {"inspect-output", "accept-workflow", "build-resume", "prepare-resume"}:
        frozen = (args.package, args.expected_package_identity, args.receipt)
        if args.command in {"build-resume", "prepare-resume"}:
            frozen += (args.prepared,)
        if args.run_id is not None:
            if args.profile is None or args.attempt_id is None or args.repo_root is None:
                raise ValueError("--run-id requires --profile, --attempt-id and --repo-root")
            if any(value is not None for value in frozen):
                raise ValueError("--run-id resolves the original package/receipt/preparation; do not override them")
            from .experiment_run import (RunReconciliationRequired, inspect_run_output,
                accept_run_workflow, build_run_resume, prepare_run_resume)
            from .experiment_family_workflow import _json
            original = (args.profile, args.repo_root, args.run_id, args.attempt_id)
            try:
                if args.command == "inspect-output":
                    result = inspect_run_output(*original, arm_id=args.arm,
                        artifact_root=args.artifact_root, expectations=_json(args.expectations))
                    return result, 1 if result["status"] == "BLOCKED" else 0
                if args.command == "accept-workflow":
                    options = dict(supplied=_json(args.outputs) if args.outputs is not None else None,
                        locations=_json(args.locations) if args.locations is not None else None,
                        execute=args.execute, execution_root=args.execution_root)
                    if args.continuation_receipt is not None:
                        options["continuation_receipt"] = args.continuation_receipt
                    if args.resume_prepared is not None or args.resume_authority is not None:
                        options.update(resume_prepared=args.resume_prepared, resume_authority=args.resume_authority)
                    if args.strict_inputs is not None:
                        options["strict_inputs"] = _json(args.strict_inputs)
                    elif args.strict:
                        options["strict_inputs"] = {}
                    if args.physical_runs is not None:
                        import json
                        options["physical_runs"] = json.loads(args.physical_runs.read_text(encoding="utf-8"))
                    if args.retrieved_root is not None:
                        options["retrieved_root"] = args.retrieved_root
                    if args.diagnose or args.stage_terminal or args.outcome is not None or args.decision_ref is not None:
                        options.update(diagnose=args.diagnose, stage_terminal=args.stage_terminal,
                            outcome=args.outcome, decision_ref=args.decision_ref,
                            diagnostic_output=args.output, n_bootstrap=args.bootstrap_samples, seed=args.seed)
                    elif args.output is not None or args.bootstrap_samples or args.seed != 42:
                        raise ValueError("Diagnostic options require --diagnose or --stage-terminal")
                    if args.role_audit is not None:
                        audit = _json(args.role_audit)
                        options["role_audit"] = args.role_audit if "context" in audit else audit
                    result = accept_run_workflow(*original, **options)
                    return result, 0 if result["status"] in {"COMPLETE", "MECHANICALLY_VERIFIED"} else 1
                if args.command == "build-resume":
                    return build_run_resume(*original, arm_id=args.arm,
                        source_output=args.source_output, output=args.output), 0
                result = prepare_run_resume(*original, plan=_json(args.resume_plan), output=args.output)
                return result, 0 if result["status"] == "RESUME_INPUTS_PREPARED" else 1
            except RunReconciliationRequired as exc:
                return {"status": "RECONCILIATION_REQUIRED", "reason": str(exc),
                        "executed": False, "network_accessed": False}, 1
        if args.profile is not None or args.attempt_id is not None:
            raise ValueError("--profile and --attempt-id require --run-id")
        if args.command == "accept-workflow":
            if any((args.retrieved_root, args.diagnose, args.stage_terminal, args.outcome,
                    args.decision_ref, args.role_audit, args.output, args.strict_inputs,
                    args.strict, args.continuation_receipt, args.physical_runs,
                    args.resume_prepared, args.resume_authority)):
                raise ValueError("Automatic acceptance requires --run-id")
            if args.outputs is None:
                raise ValueError("Explicit Spec acceptance requires --outputs")
            if args.execute and args.locations is None:
                raise ValueError("Terminal execution requires --locations")
        if any(value is None for value in frozen):
            raise ValueError("--spec requires --package, --expected-package-identity, --receipt and, for recovery, --prepared")
    raw = _read(args.spec)
    spec = ExperimentSpec.from_json(raw)
    if raw != spec.to_json():
        raise ValueError("Expected canonical ExperimentSpec JSON bytes (no newline)")
    if args.command == "validate-spec":
        return {"spec_identity": spec.identity, "spec": spec.to_dict()}, 0
    if args.command == "workflow-info":
        from .experiment_execution import workflow_capabilities
        return workflow_capabilities(spec), 0
    if args.command == "check-workflow":
        from .experiment_workflow import check_workflow_readiness
        from .experiment_family_workflow import _json
        result = check_workflow_readiness(spec, args.repo_root, _json(args.plan))
        return result, 0 if result["status"] == "WORKFLOW_INPUTS_READY" else 1
    if args.command == "source-dependencies":
        from .source_dependencies import registered_dependencies
        closure = registered_dependencies(spec, args.repo_root, args.extra_source,
                                          require_static=args.strict_dynamic)
        return {"spec_identity": spec.identity, **closure.to_dict(),
                "submitted": False, "execution_qualification": "NOT_EVALUATED"}, 0
    if args.command in {"build-resume", "prepare-resume"}:
        from .experiment_workflow_resume import build_workflow_resume, prepare_resumed_workflow
        from .experiment_family_workflow import _json
        options = dict(spec=spec, repo_root=args.repo_root, prepared=args.prepared,
                       package_dir=args.package, expected_package_identity=args.expected_package_identity,
                       receipt_path=args.receipt, output=args.output)
        if args.command == "build-resume":
            return build_workflow_resume(**options, arm_id=args.arm, source_output=args.source_output), 0
        result = prepare_resumed_workflow(**options, plan=_json(args.resume_plan))
        return result, 0 if result["status"] == "RESUME_INPUTS_PREPARED" else 1
    if args.command == "prepare-family-training":
        from .experiment_workflow import prepare_family_training
        from .experiment_family_workflow import _json
        plan = _json(args.plan)
        if args.auto_dependencies:
            from .experiment_source_inventory import registered_source_files
            plan["source_files"] = registered_source_files(spec, plan.get("source_files", ()),
                repo_root=args.repo_root, auto_dependencies=True)
        result = prepare_family_training(spec, args.repo_root, plan, args.output)
        return result, 0 if result["status"] == "SOURCE_INPUTS_PREPARED" else 1
    if args.command == "build-terminal":
        from .experiment_family_workflow import (RunContext, _json, build_incomplete_terminal_descriptor,
                                                 incomplete_observations_from_execution)
        from .experiment_launch import publish_immutable_bytes
        contexts = {arm["arm_id"]: RunContext.from_launch(spec, args.receipt, args.package,
            expected_package_identity=args.expected_package_identity, arm_id=arm["arm_id"])
            for arm in spec.to_dict()["arms"]}
        observations = (_json(args.observations) if args.observations else
                        incomplete_observations_from_execution(spec, _json(args.execution_state)))
        descriptor = build_incomplete_terminal_descriptor(spec, contexts=contexts,
            observations=observations, locations=_json(args.locations))
        publish_immutable_bytes(args.output, descriptor.to_json().encode())
        return {"status": "DESCRIPTOR_BUILT", "descriptor": descriptor.to_dict(), "executed": False}, 0
    if args.command in {"prepare-workflow", "accept-workflow"}:
        from .experiment_workflow import prepare_workflow, accept_workflow
        from .experiment_family_workflow import _json
        if args.command == "prepare-workflow":
            result = prepare_workflow(spec, args.repo_root, _json(args.plan), args.output)
            return result, 0 if result["status"] == "PREPARED_FOR_PLATFORM" else 1
        result = accept_workflow(spec, args.repo_root, _json(args.outputs),
            receipt_path=args.receipt, package_dir=args.package,
            expected_package_identity=args.expected_package_identity,
            locations=_json(args.locations) if args.locations is not None else None, execute=args.execute,
            **({"execution_root": args.execution_root} if args.execution_root is not None else {}))
        return result, 0 if result["status"] in {"COMPLETE", "MECHANICALLY_VERIFIED"} else 1
    if args.command in {"check-acceptance", "inspect-output", "accept-terminal"}:
        from .experiment_family_workflow import (
            RunContext, _json, check_acceptance_plan, inspect_output, close_verified_outputs,
            prepare_terminal_outputs,
        )
        if args.command == "check-acceptance":
            result = check_acceptance_plan(spec, args.repo_root, _json(args.plan))
            return result, 1 if result["status"] == "BLOCKED" else 0
        def context(arm_id):
            return RunContext.from_launch(spec, args.receipt, args.package,
                expected_package_identity=args.expected_package_identity, arm_id=arm_id)
        if args.command == "inspect-output":
            result = inspect_output(args.artifact_root, context=context(args.arm), expected=_json(args.expectations))
            return result, 1 if result["status"] == "BLOCKED" else 0
        outputs = prepare_terminal_outputs(spec, args.repo_root, _json(args.outputs),
            receipt_path=args.receipt, package_dir=args.package,
            expected_package_identity=args.expected_package_identity)
        descriptor_raw = _read(args.descriptor)
        descriptor = TerminalDescriptor.from_json(spec, descriptor_raw)
        if descriptor_raw != descriptor.to_json():
            raise ValueError("Expected canonical TerminalDescriptor JSON")
        result = close_verified_outputs(args.repo_root, spec, descriptor, outputs=outputs, execute=args.execute)
        return result, 0 if result["status"] in {"COMPLETE", "MECHANICALLY_VERIFIED"} else 1
    if args.command == "plan-prospective":
        return plan_prospective(spec, args.repo_root)
    if args.command == "prepare-release":
        return prepare_experiment_release(spec, args.repo_root, _read(args.workflow), args.output)
    if args.command == "check-preparation":
        return check_experiment_preparation(spec, args.repo_root, _read(args.workflow)), 0
    if args.command == "package":
        files = args.allowlist
        if args.auto_dependencies:
            from .experiment_source_inventory import registered_source_files
            files = registered_source_files(spec, repo_root=args.repo_root, auto_dependencies=True)
        return build_experiment_source_package(spec, args.repo_root, files, args.output), 0
    if args.command == "check-release":
        def bindings(values, local=False):
            result = {}
            for entry in values:
                arm, separator, value = entry.partition("=")
                if not separator or not arm or not value or arm in result:
                    raise ValueError("Expected unique ARM=PATH bindings")
                result[arm] = _local(value) if local else value
            return result
        result = check_release_inputs(spec, args.package,
            expected_package_identity=args.expected_package_identity,
            recipe_files=bindings(args.recipe_file), initial_states=bindings(args.initial_state, True),
            required_modules=args.required_module, pickle_inputs=args.pickle_input,
            entry_script=args.entry_script, input_root=args.input_root,
            dataset_manifests=bindings(args.dataset_manifest, True) if args.dataset_manifest else None)
        if args.output_report:
            _atomic(args.output_report, result)
        return result, 1 if result["errors"] else 0
    if args.command == "preflight":
        manifest = args.shard_root / "shard_manifest.json"
        _safe_local(manifest)
        result = run_experiment_preflight(
            spec, args.package, args.output,
            expected_package_identity=args.expected_package_identity,
            shard_manifest=manifest if manifest.exists() else None,
            shard_root=args.shard_root,
            expected_shard_manifest_sha256=args.expected_shard_manifest_sha256,
            mode=args.mode,
        )
        return result, 0 if result["status"] in {
            "LOADER_VERIFIED_ONLY", "MODEL_SMOKE_VERIFIED_ONLY",
        } else 1
    if args.command == "run-diagnostic":
        result = run_experiment(spec, args.output, args.device, worker=args.worker)
        return result, 0 if result["status"] == "SUCCEEDED" else 1
    if args.command == "launch-receipt":
        if args.response is None:
            result = build_launch_receipt(
                spec, args.package, expected_package_identity=args.expected_package_identity,
            )
        else:
            result = reconcile_platform_response(
                spec, args.package, _read(args.response),
                expected_package_identity=args.expected_package_identity,
            )
        write_launch_receipt(
            canonical_json(result), args.output_dir, spec, args.package,
            expected_package_identity=args.expected_package_identity,
        )
        return result, 1 if result["submission_state"] == "REJECTED" else 0
    raw = _read(args.descriptor)
    descriptor = TerminalDescriptor.from_json(spec, raw)
    if raw != descriptor.to_json():
        raise ValueError("Expected canonical TerminalDescriptor JSON bytes (no newline)")
    if args.execute:
        results = execute_terminal_descriptor(args.repo_root, spec, descriptor)
        return {"executed": True, "results": results}, 0 if all(
            result.get("pipeline_status") == "COMPLETE" for result in results
        ) else 1
    return {"executed": False, "arms": translate_terminal_descriptor(
        args.repo_root, spec, descriptor,
    )}, 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        # Keep any library diagnostics off the machine-readable output channel.
        with redirect_stdout(sys.stderr):
            result, code = _dispatch(args)
        if args.command in {"prepare-workflow", "prepare-family-training", "prepare-release"} and not args.verbose:
            from .experiment_input import compact_operation_report
            result = compact_operation_report(result, prepared=args.output)
        if (args.command in {"diagnose", "accept-workflow"} and args.run_id is not None
                and not args.verbose and ((args.command == "diagnose" and args.output is not None)
                                         or "diagnosis" in result or "strict_validation" in result)):
            result = _acceptance_summary(result)
        print(canonical_json(result))
        return code
    except Exception as exc:
        print(canonical_json({"status": "ERROR", "error": {
            "type": type(exc).__name__, "message": str(exc),
        }}))
        return 2


def _acceptance_summary(result):
    """Keep ordinary stdout bounded; detailed evidence remains at its report pointer."""
    summary = {key: result[key] for key in ("status", "run_id", "attempt_id", "spec_identity",
        "executed", "closure_complete", "scientific_acceptance", "scientific_outcome", "replay_readiness", "blockers",
        "strict_mode", "physical_allocation_costs") if key in result}
    if "results" in result:
        summary["closure"] = [{key: row[key] for key in
            ("trajectory_id", "trajectory", "pipeline_status", "finalization_status", "error") if key in row}
            for row in result["results"]]
    if "strict_validation" in result:
        strict = result["strict_validation"]
        summary["strict_validation"] = {key: strict[key] for key in
            ("status", "blockers", "replay_readiness", "ready_for_desktop") if key in strict}
        summary["strict_validation"]["arms"] = {
            arm: {**{key: row[key] for key in ("comparison", "replay", "blockers") if key in row},
                "native_qualification": {key: row.get("native_qualification", {})[key]
                    for key in ("status", "blockers", "missing") if key in row.get("native_qualification", {})}}
            for arm, row in strict.get("arms", {}).items()}
    summary["arms"] = {arm: {key: row[key] for key in ("status", "blockers") if key in row}
                       for arm, row in result.get("arms", {}).items()}
    diagnostic = result.get("diagnosis", result)
    if "comparisons" in diagnostic:
        summary["diagnosis"] = {"status": diagnostic["status"], "reports": diagnostic.get("written_files", {}),
            "comparisons": {arm: {"status": pair["status"], "blockers": pair.get("blockers", []),
                "mean_delta_eV": pair.get("primary_comparison", {}).get("mean_delta_eV"),
                "signals": pair.get("signals", {})} for arm, pair in diagnostic["comparisons"].items()}}
    if "terminal_staging" in result:
        staging = result["terminal_staging"]
        summary["terminal_staging"] = {key: staging[key] for key in ("status", "paths", "blockers") if key in staging}
    if "strict_comparison" in result:
        summary["strict_comparison"] = {key: result["strict_comparison"][key]
            for key in ("directory", "locations", "reference_bundle_ref")}
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
