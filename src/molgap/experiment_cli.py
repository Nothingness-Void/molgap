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
from .experiment_workflow import prepare_experiment_release
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
    for name in ("validate-spec", "package", "preflight", "run-diagnostic",
                 "launch-receipt", "terminal", "plan-prospective", "check-release",
                 "prepare-release", "check-acceptance", "inspect-output", "accept-terminal"):
        command = commands.add_parser(name)
        command.add_argument("--spec", required=True, type=_local)
        if name in {"package", "terminal", "plan-prospective", "prepare-release", "check-acceptance", "accept-terminal"}:
            command.add_argument("--repo-root", required=True, type=_local)
        if name in {"package", "preflight", "run-diagnostic", "prepare-release"}:
            command.add_argument("--output", required=True, type=_local)
        if name in {"preflight", "launch-receipt", "check-release", "inspect-output", "accept-terminal"}:
            command.add_argument("--package", required=True, type=_local)
            command.add_argument("--expected-package-identity", required=True)
        if name == "prepare-release":
            command.add_argument("--workflow", required=True, type=_local)
        elif name == "check-acceptance":
            command.add_argument("--plan", required=True, type=_local)
        elif name in {"inspect-output", "accept-terminal"}:
            command.add_argument("--receipt", required=True, type=_local)
            if name == "inspect-output":
                command.add_argument("--arm", required=True)
                command.add_argument("--artifact-root", required=True, type=_local)
                command.add_argument("--expectations", required=True, type=_local)
            else:
                command.add_argument("--descriptor", required=True, type=_local)
                command.add_argument("--outputs", required=True, type=_local)
                command.add_argument("--execute", action="store_true")
        elif name == "package":
            command.add_argument("--allowlist", required=True, nargs="+", action="extend")
        elif name == "check-release":
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
    return parser


def _dispatch(args) -> tuple[dict, int]:
    raw = _read(args.spec)
    spec = ExperimentSpec.from_json(raw)
    if raw != spec.to_json():
        raise ValueError("Expected canonical ExperimentSpec JSON bytes (no newline)")
    if args.command == "validate-spec":
        return {"spec_identity": spec.identity, "spec": spec.to_dict()}, 0
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
    if args.command == "package":
        return build_experiment_source_package(spec, args.repo_root, args.allowlist, args.output), 0
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
            entry_script=args.entry_script, input_root=args.input_root)
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
        print(canonical_json(result))
        return code
    except Exception as exc:
        print(canonical_json({"status": "ERROR", "error": {
            "type": type(exc).__name__, "message": str(exc),
        }}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
